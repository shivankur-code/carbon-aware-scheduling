"""Fleet design: carbon breakdown with/without clock locking, and the blast-radius
break-even for bigger tensor-parallel groups.

Usage: python experiments/fleet_design.py [configs/fleet.yaml]
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from calf.optimize.fleet import Design, Env, Gpu, Workload, breakeven_efficiency, evaluate, optimize

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(cfg):
    gpu = Gpu(**cfg["gpu"])
    wls = {w["name"]: Workload(**w) for w in cfg["workloads"]}
    return gpu, wls


def main(cfg_path: str = "configs/fleet.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    gpu, wls = load(cfg)
    w = wls["70B model"]
    rows = []
    # (a) best design per grid, clock locking allowed vs max clock only
    for gname, ci in cfg["grids_kg_per_kwh"].items():
        env = Env(carbon_intensity=ci, **cfg["env"])
        for label, clocks in [("max clock", (1.0,)), ("clock-locked", (1.0, 0.8, 0.39))]:
            best = min((evaluate(Design(tp, c), w, gpu, env) for tp in (4, 8) for c in clocks),
                       key=lambda e: e.total)
            rows.append({"panel": "breakdown", "grid": gname, "variant": label,
                         "tp": best.design.tp, "clock": best.design.clock,
                         "gpus": best.gpus_total, "spare_pct": round(best.spare_pct, 2),
                         "embodied_active_t": best.embodied_active / 1e3,
                         "embodied_spares_t": best.embodied_spares / 1e3,
                         "operational_t": best.operational / 1e3,
                         "spare_idle_t": best.spare_idle / 1e3, "total_t": best.total / 1e3})
    # (b) break-even per-GPU efficiency of TP8 vs TP4 vs recovery time
    env0 = Env(carbon_intensity=cfg["grids_kg_per_kwh"]["mid"], **cfg["env"])
    for gname, ci in cfg["grids_kg_per_kwh"].items():
        for h in cfg["recovery_hours_sweep"]:
            env = replace(env0, carbon_intensity=ci, mttr_h=h)
            r = breakeven_efficiency(w, gpu, env, 4, 8)
            rows.append({"panel": "breakeven", "grid": gname, "recovery_h": h,
                         "tp8_needed_eff_ratio": r})
    # sanity: optimizer agrees with enumeration
    best = optimize(w, gpu, env0)
    print("mid-grid optimum:", best.design, best.gpus_total, "GPUs")

    out = Path("results/fleet_design.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=SURFACE)
    ax = axes[0]
    br = [r for r in rows if r["panel"] == "breakdown"]
    parts = [("embodied_active_t", "embodied, serving GPUs"), ("embodied_spares_t", "embodied, spares"),
             ("operational_t", "operational, serving"), ("spare_idle_t", "operational, spares")]
    labels = [f"{r['grid'].split(' ')[0]}\n{r['variant']}" for r in br]
    bottom = [0.0] * len(br)
    for i, (k, lab) in enumerate(parts):
        vals = [r[k] for r in br]
        ax.bar(range(len(br)), vals, bottom=bottom, color=SERIES[i], label=lab,
               edgecolor=SURFACE, linewidth=1.5, width=0.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, r in enumerate(br):
        ax.text(x, bottom[x] * 1.01, f"{r['total_t']:.0f} t", ha="center", va="bottom",
                fontsize=7, color=INK2)
    ax.set_xticks(range(len(br)), labels, fontsize=7.5)
    ax.set_ylabel("tCO2e per year (70B fleet, ~500 GPUs)")
    ax.set_title("Carbon by source, best design per grid", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    ax = axes[1]
    for i, gname in enumerate(cfg["grids_kg_per_kwh"]):
        sub = [r for r in rows if r["panel"] == "breakeven" and r["grid"] == gname]
        ax.plot([r["recovery_h"] for r in sub], [100 * (r["tp8_needed_eff_ratio"] - 1) for r in sub],
                color=SERIES[i], lw=2, marker="o", ms=4, label=gname)
    ax.set_xscale("log")
    ax.set_xlabel("Recovery time per error (hours, log)")
    ax.set_ylabel("TP8 per-GPU efficiency gain needed (%)")
    ax.set_title("Break-even: 8-GPU vs 4-GPU replicas", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, title="grid", title_fontsize=7.5)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8, axis="y")
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/figures/fleet_design.png", dpi=200, facecolor=SURFACE)
    print("wrote results/fleet_design.csv and docs/figures/fleet_design.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
