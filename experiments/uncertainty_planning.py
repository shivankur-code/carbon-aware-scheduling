"""Promised vs realized: spare sizing under uncertain error rates and recovery times.

Plans: fuzzy alpha-cut plans (alpha = 1 is the deterministic plan at most-likely values,
alpha -> 0 the most conservative) and a scenario chance-constrained plan. Each plan is then
evaluated over Monte Carlo scenarios of error rate, recovery time, embodied carbon and grid
intensity.

Usage: python experiments/uncertainty_planning.py [configs/fleet.yaml]
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from calf.models import reliability as rel
from calf.optimize.fleet import Design, Env, Gpu, Workload, evaluate
from calf.optimize.uncertainty import TFN, fuzzy_eval, sample_scenarios

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(cfg_path: str = "configs/fleet.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    u = cfg["uncertainty"]
    gpu = Gpu(**cfg["gpu"])
    w = next(Workload(**x) for x in cfg["workloads"] if x["name"] == u["workload"])
    ci0 = cfg["grids_kg_per_kwh"][u["grid"]]
    env0 = Env(carbon_intensity=ci0, **cfg["env"])
    params = {k: TFN(*u[k]) for k in ("error_rate", "mttr_h", "embodied_kg", "ci_factor")}
    base = Design(**u["design"])
    target = env0.availability_target

    def ev(design: Design, p: dict) -> object:
        env = replace(env0, error_rate=p["error_rate"], mttr_h=p["mttr_h"],
                      carbon_intensity=ci0 * p["ci_factor"])
        return evaluate(design, w, replace(gpu, embodied_kg=p["embodied_kg"]), env)

    nominal = {k: v.mode for k, v in params.items()}
    replicas = ev(base, nominal).replicas
    scen = sample_scenarios(params, u["scenarios"], seed=7)

    plans = {}
    for a in u["alphas"]:
        lam_hi = params["error_rate"].alpha_cut(a)[1]
        mttr_hi = params["mttr_h"].alpha_cut(a)[1]
        q = rel.unavailability(lam_hi * base.tp, mttr_hi)
        plans[f"fuzzy alpha={a:g}"] = rel.spares_needed(replicas, q, target, replicas * 4)
    # scenario chance constraint: smallest S meeting the target in >= chance_level of scenarios
    qs = np.array([rel.unavailability(s["error_rate"] * base.tp, s["mttr_h"]) for s in scen])
    for s_cnt in range(replicas * 4):
        ok = np.mean([rel.pool_availability(replicas, s_cnt, q) >= target for q in qs])
        if ok >= u["chance_level"]:
            plans[f"scenario chance {u['chance_level']:.0%}"] = s_cnt
            break

    rows = []
    for name, s_cnt in plans.items():
        d = replace(base, spares=s_cnt)
        promised = ev(d, nominal)
        real = [ev(d, s) for s in scen]
        av = np.array([e.availability for e in real])
        tot = np.array([e.total for e in real]) / 1e3
        lo, hi = fuzzy_eval(lambda p, d=d: ev(d, p).total / 1e3, params, 0.5)
        rows.append({"plan": name, "spares": s_cnt, "spare_gpus": s_cnt * base.tp,
                     "promised_availability": promised.availability,
                     "promised_total_t": promised.total / 1e3,
                     "p_miss_target": float(np.mean(av < target)),
                     "realized_availability_p05": float(np.quantile(av, 0.05)),
                     "realized_total_t_mean": float(tot.mean()),
                     "realized_total_t_p95": float(np.quantile(tot, 0.95)),
                     "fuzzy_total_t_alpha05_lo": lo, "fuzzy_total_t_alpha05_hi": hi})
    det = next(r for r in rows if r["plan"] == "fuzzy alpha=1")
    for r in rows:
        r["extra_carbon_vs_det_pct"] = 100 * (r["realized_total_t_mean"] /
                                              det["realized_total_t_mean"] - 1)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    out = Path("results/uncertainty_planning.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    fz = [r for r in rows if r["plan"].startswith("fuzzy")]
    ax.plot([r["extra_carbon_vs_det_pct"] for r in fz], [100 * r["p_miss_target"] for r in fz],
            color=SERIES[0], lw=2, marker="o", ms=7, mec=SURFACE, mew=2, zorder=3,
            label="fuzzy alpha-cut plans")
    for r in fz:
        a = r["plan"].split("=")[1]
        tag = " (deterministic)" if a == "1" else ""
        ax.annotate(f"alpha={a}: {r['spare_gpus']} spares{tag}",
                    (r["extra_carbon_vs_det_pct"], 100 * r["p_miss_target"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=7, color=INK2)
    for r in rows:
        if r["plan"].startswith("scenario"):
            ax.scatter(r["extra_carbon_vs_det_pct"], 100 * r["p_miss_target"], s=160,
                       facecolor="none", edgecolor=SERIES[1], linewidth=2, zorder=4,
                       label=f"{r['plan']} ({r['spare_gpus']} spares)")
    ax.set_xlabel("Extra annual carbon vs deterministic plan (%)")
    ax.set_ylabel(f"Scenarios missing {100 * target:.1f}% availability (%)")
    ax.set_title(f"Promised vs realized, {w.name}, {u['grid']} grid", loc="left", fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig("docs/figures/uncertainty_planning.png", dpi=200, facecolor=SURFACE)
    print("wrote results/uncertainty_planning.csv and docs/figures/uncertainty_planning.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
