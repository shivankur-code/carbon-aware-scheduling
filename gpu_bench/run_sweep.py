"""Run the GPU measurement campaign described in a YAML config.

    python -m gpu_bench.run_sweep configs/gpu/g1_powercap.yaml [--quick]

Experiment types (each turns a model ASSUMPTION into a measurement):
  idle       GPU power with no process, and with a model loaded but no traffic
             -> spare idle power
  coldstart  launch-to-first-token time, repeated
             -> repair time (MTTR) for software failures
  load       sweep of power caps x load levels (x tensor-parallel degree)
             -> throughput, power and temperature vs cap; latency/SLO; J/token
  fault      SIGKILL the server under load, restart it, count lost requests
             -> downtime and SLO impact of one failure

Outputs: data/measurements/gpu/<tag>/<experiment>/... and data/measurements/gpu/<tag>/summary.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from gpu_bench import nvml_monitor as nv
from gpu_bench.loadgen import run_load, to_rows
from gpu_bench.summarize import combine, gpu_metrics, request_metrics

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- server lifecycle
def start_server(model: str, tp: int, gpus: list[int], port: int, extra: list[str],
                 log: Path) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus[:tp]))}
    cmd = ["vllm", "serve", model, "--port", str(port), "--tensor-parallel-size", str(tp),
           *extra]
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def first_token_ok(base: str, model: str) -> bool:
    body = json.dumps({"model": model, "prompt": "hello", "max_tokens": 1}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_ready(proc: subprocess.Popen, base: str, model: str, timeout_s: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see server log")
        if first_token_ok(base, model):
            return time.perf_counter() - t0
        time.sleep(0.5)
    raise TimeoutError("server not ready")


def stop_server(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


# ---------------------------------------------------------------- helpers
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def append_summary(path: Path, row: dict) -> None:
    rows = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
    write_csv(path, [*rows, row])


def vllm_version() -> str:
    try:
        out = subprocess.run([sys.executable, "-c", "import vllm;print(vllm.__version__)"],
                             capture_output=True, text=True, timeout=120, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- experiments
def exp_idle(cfg, exp, out, meta):
    gpus = cfg["gpus"]
    for phase in ("no_process", "model_loaded"):
        proc = None
        if phase == "model_loaded":
            proc = start_server(exp["model"], exp.get("tp", 1), gpus, cfg["port"],
                                cfg["server"]["extra_args"], out / "server.log")
            wait_ready(proc, meta["base"], exp["model"], cfg["server"]["ready_timeout_s"])
            time.sleep(exp.get("settle_s", 30))
        t0 = time.time()
        with nv.Monitor(gpus, out / f"nvml_{phase}.csv") as mon:
            time.sleep(exp["duration_s"])
        g = gpu_metrics(mon.rows, t0, time.time())
        append_summary(meta["summary"], {**meta["row"], "experiment": "idle", "phase": phase,
                                         "model": exp["model"], "n_gpus": len(gpus), **g})
        if proc:
            stop_server(proc)


def exp_coldstart(cfg, exp, out, meta):
    for k in range(exp.get("repeats", 3)):
        proc = start_server(exp["model"], exp.get("tp", 1), cfg["gpus"], cfg["port"],
                            cfg["server"]["extra_args"], out / "server.log")
        ready = wait_ready(proc, meta["base"], exp["model"], cfg["server"]["ready_timeout_s"])
        stop_server(proc)
        time.sleep(5)
        append_summary(meta["summary"], {**meta["row"], "experiment": "coldstart",
                                         "model": exp["model"], "tp": exp.get("tp", 1),
                                         "repeat": k, "ready_s": ready})


def exp_load(cfg, exp, out, meta, quick):
    warm, dur = (10, 30) if quick else (exp.get("warmup_s", 120), exp.get("duration_s", 300))
    for model in exp["models"]:
        for tp in exp.get("tp", [1]):
            gpus = cfg["gpus"][:tp]
            proc = start_server(model, tp, cfg["gpus"], cfg["port"],
                                cfg["server"]["extra_args"], out / "server.log")
            try:
                wait_ready(proc, meta["base"], model, cfg["server"]["ready_timeout_s"])
                for cap in exp["caps"]:
                    watts = nv.set_power_cap(gpus, cap) if nv.HAVE_NVML else []
                    for li, load in enumerate(exp["loads"]):
                        wl = {**cfg["workload"], **load}
                        asyncio.run(run_load(meta["base"], model, wl, warm, seed=99))
                        t0 = time.time()
                        run_id = f"{Path(model).name}_tp{tp}_cap{cap:.2f}_load{li}"
                        with nv.Monitor(gpus, out / run_id / "nvml.csv") as mon:
                            reqs = to_rows(asyncio.run(run_load(meta["base"], model, wl, dur,
                                                                seed=li)))
                        t1 = time.time()
                        write_csv(out / run_id / "requests.csv", reqs)
                        r = request_metrics(reqs, t1 - t0, cfg["slo"]["ttft_s"],
                                            cfg["slo"]["tpot_s"])
                        row = combine(r, gpu_metrics(mon.rows, t0, t1))
                        append_summary(meta["summary"], {
                            **meta["row"], "experiment": "load", "model": model, "tp": tp,
                            "n_gpus": tp, "power_cap_frac": cap,
                            "power_cap_w": sum(watts) if watts else "", "load": json.dumps(load),
                            **row})
                        print(f"[load] {run_id}: {row['tokens_per_s']:.0f} tok/s, "
                              f"{row['avg_power_w']:.0f} W, {row['gpu_temp_c']} C, "
                              f"SLO {row['slo_attainment']:.2%}", flush=True)
            finally:
                if nv.HAVE_NVML:
                    nv.set_power_cap(gpus, 1.0)
                stop_server(proc)


def exp_fault(cfg, exp, out, meta):
    model, tp = exp["model"], exp.get("tp", 1)
    log = out / "server.log"
    proc = start_server(model, tp, cfg["gpus"], cfg["port"], cfg["server"]["extra_args"], log)
    wait_ready(proc, meta["base"], model, cfg["server"]["ready_timeout_s"])
    events: dict = {}

    async def killer():
        nonlocal proc
        await asyncio.sleep(exp["kill_at_s"])
        events["kill_t"] = time.time()
        stop_server(proc, hard=True)
        proc = start_server(model, tp, cfg["gpus"], cfg["port"], cfg["server"]["extra_args"],
                            log)
        ready = await asyncio.to_thread(wait_ready, proc, meta["base"], model,
                                        cfg["server"]["ready_timeout_s"])
        events["recovered_t"] = time.time()
        events["restart_ready_s"] = ready

    async def main():
        wl = {**cfg["workload"], **exp["load"]}
        res, _ = await asyncio.gather(run_load(meta["base"], model, wl, exp["duration_s"]),
                                      killer())
        return res

    t0 = time.time()
    with nv.Monitor(cfg["gpus"][:tp], out / "nvml.csv"):
        reqs = to_rows(asyncio.run(main()))
    t1 = time.time()
    stop_server(proc)
    write_csv(out / "requests.csv", reqs)
    r = request_metrics(reqs, t1 - t0, cfg["slo"]["ttft_s"], cfg["slo"]["tpot_s"])
    append_summary(meta["summary"], {**meta["row"], "experiment": "fault", "model": model,
                                     "tp": tp, "downtime_s":
                                     events["recovered_t"] - events["kill_t"],
                                     **events, **r})


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true", help="short windows, smoke test")
    ap.add_argument("--only", nargs="*", help="run only these experiment types")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    info = nv.device_info(cfg["gpus"])
    gpu_name = info[0]["name"].replace(" ", "_") if info else "nogpu"
    tag = cfg.get("tag") or f"{gpu_name}_{dt.datetime.now(dt.UTC):%Y%m%d}"
    base_out = ROOT / "data" / "measurements" / "gpu" / tag
    base_out.mkdir(parents=True, exist_ok=True)
    env = {"gpus": info, "vllm": vllm_version(), "python": sys.version, "config": cfg,
           "started": dt.datetime.now(dt.UTC).isoformat()}
    (base_out / "env.json").write_text(json.dumps(env, indent=2))
    meta = {"base": f"http://127.0.0.1:{cfg['port']}", "summary": base_out / "summary.csv",
            "row": {"tag": tag, "gpu_name": gpu_name, "vllm": env["vllm"]}}
    for exp in cfg["experiments"]:
        if a.only and exp["type"] not in a.only:
            continue
        out = base_out / exp.get("name", exp["type"])
        print(f"=== {exp['type']} -> {out}", flush=True)
        if exp["type"] == "idle":
            exp_idle(cfg, exp, out, meta)
        elif exp["type"] == "coldstart":
            exp_coldstart(cfg, exp, out, meta)
        elif exp["type"] == "load":
            exp_load(cfg, exp, out, meta, a.quick)
        elif exp["type"] == "fault":
            exp_fault(cfg, exp, out, meta)
        else:
            raise ValueError(exp["type"])
    print(f"done: {meta['summary']}")


if __name__ == "__main__":
    main()
