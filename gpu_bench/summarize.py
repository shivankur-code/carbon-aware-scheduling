"""Pure functions turning raw request + NVML samples into one summary row. Unit-tested."""

from __future__ import annotations

import itertools
import math
import statistics


def pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def request_metrics(reqs: list[dict], window_s: float, ttft_slo_s: float,
                    tpot_slo_s: float) -> dict:
    ok = [r for r in reqs if r["ok"]]
    tpot = [(r["e2e_s"] - r["ttft_s"]) / (r["output_tokens"] - 1)
            for r in ok if r["output_tokens"] > 1 and not math.isnan(r["ttft_s"])]
    ttft = [r["ttft_s"] for r in ok]
    meets = [r for r in ok if r["output_tokens"] > 1
             and r["ttft_s"] <= ttft_slo_s
             and (r["e2e_s"] - r["ttft_s"]) / (r["output_tokens"] - 1) <= tpot_slo_s]
    out_tok = sum(r["output_tokens"] for r in ok)
    return {
        "requests": len(reqs),
        "failed": len(reqs) - len(ok),
        "req_per_s": len(ok) / window_s,
        "tokens_per_s": out_tok / window_s,
        "total_tokens_per_s": (out_tok + sum(r["prompt_tokens"] for r in ok)) / window_s,
        "output_tokens": out_tok,
        "ttft_p50_s": pct(ttft, 0.5), "ttft_p99_s": pct(ttft, 0.99),
        "tpot_p50_s": pct(tpot, 0.5), "tpot_p99_s": pct(tpot, 0.99),
        "slo_attainment": len(meets) / len(reqs) if reqs else float("nan"),
    }


def gpu_metrics(samples: list[dict], t0: float, t1: float) -> dict:
    """Energy from the NVML energy counter where present, else trapezoid over power."""
    win = [s for s in samples if t0 <= s["t"] <= t1]
    if not win:
        return {"avg_power_w": float("nan"), "energy_j": float("nan"),
                "gpu_temp_c": float("nan"), "sm_mhz": float("nan")}
    by_gpu: dict[int, list[dict]] = {}
    for s in win:
        by_gpu.setdefault(int(s["gpu"]), []).append(s)
    energy, temps, clocks = 0.0, [], []
    for rows in by_gpu.values():
        rows.sort(key=lambda s: s["t"])
        e0, e1 = rows[0]["energy_j"], rows[-1]["energy_j"]
        if not math.isnan(e0) and not math.isnan(e1) and len(rows) > 1:
            energy += e1 - e0
        else:
            energy += sum((b["t"] - a["t"]) * (a["power_w"] + b["power_w"]) / 2
                          for a, b in itertools.pairwise(rows))
        steady = rows[len(rows) // 2:]  # second half of the window = thermal steady state
        temps.append(statistics.median(s["temp_c"] for s in steady))
        clocks.append(statistics.median(s["sm_mhz"] for s in steady))
    span = max(s["t"] for s in win) - min(s["t"] for s in win)
    return {
        "avg_power_w": energy / span if span > 0 else float("nan"),
        "energy_j": energy,
        "gpu_temp_c": max(temps),
        "sm_mhz": statistics.median(clocks),
    }


def combine(req: dict, gpu: dict) -> dict:
    row = {**req, **gpu}
    tok = req["output_tokens"]
    row["j_per_output_token"] = gpu["energy_j"] / tok if tok else float("nan")
    return row
