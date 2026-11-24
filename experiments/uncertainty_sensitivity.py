"""How robust is the promised-vs-realized finding to the assumed uncertainty ranges?

Sweeps the width of the error-rate range (mode / w .. mode * w) and the upper end of the
recovery-time range, for two most-likely error rates and recovery times taken from Delta
(A100 and H100). For each setting it reports:
  * how often the deterministic plan (spares sized at the most-likely values) misses the
    availability target, and
  * the spare count and carbon premium of a 95% scenario chance-constrained plan.

Usage: python experiments/uncertainty_sensitivity.py [configs/fleet.yaml]
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
from scipy.stats import binom

from calf.models import reliability as rel
from calf.optimize.fleet import Design, Env, Gpu, Workload, evaluate
from calf.optimize.uncertainty import TFN, sample_scenarios

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
WIDTHS = [1.25, 1.5, 2.0, 3.0]          # error-rate range = mode/w .. mode*w
MTTR_HI = [4.0, 8.0, 16.0, 24.0]        # hours; Delta's worst single incident was ~23 node-h
MODES = {"H100 (7.5/GPU-y, 2.2 h)": (7.5, 2.2), "A100 (13.5/GPU-y, 0.88 h)": (13.5, 0.88)}
N_SCEN = 3000
CHANCE = 0.95


def availability(replicas: int, spares: int, q: np.ndarray) -> np.ndarray:
    return binom.cdf(spares, replicas + spares, q)


def main(cfg_path: str = "configs/fleet.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    u = cfg["uncertainty"]
    gpu = Gpu(**cfg["gpu"])
    w = next(Workload(**x) for x in cfg["workloads"] if x["name"] == u["workload"])
    ci = cfg["grids_kg_per_kwh"][u["grid"]]
    d = Design(**u["design"])
    target = cfg["env"]["availability_target"]
    rows = []
    for label, (lam0, mttr0) in MODES.items():
        env0 = Env(carbon_intensity=ci, **{**cfg["env"], "error_rate": lam0, "mttr_h": mttr0})
        det = evaluate(d, w, gpu, env0)
        replicas = det.replicas
        for wd in WIDTHS:
            for mhi in MTTR_HI:
                params = {"lam": TFN(lam0 / wd, lam0, lam0 * wd),
                          "mttr": TFN(5 / 60, mttr0, max(mhi, mttr0))}
                sc = sample_scenarios(params, N_SCEN, seed=11)
                q = np.array([rel.unavailability(s["lam"] * d.tp, s["mttr"]) for s in sc])
                p_miss_det = float(np.mean(availability(replicas, det.spares, q) < target))
                s_ch = det.spares
                while np.mean(availability(replicas, s_ch, q) >= target) < CHANCE:
                    s_ch += 1
                ch = evaluate(replace(d, spares=s_ch), w, gpu, env0)
                rows.append({"modes": label, "error_width": wd, "mttr_hi_h": mhi,
                             "det_spares": det.spares, "det_p_miss": p_miss_det,
                             "chance_spares": s_ch,
                             "chance_carbon_premium_pct": 100 * (ch.total / det.total - 1)})
    out = Path("results/uncertainty_sensitivity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    fig, axes = plt.subplots(2, len(MODES), figsize=(10, 6.6), facecolor=SURFACE)
    for j, label in enumerate(MODES):
        sub = [r for r in rows if r["modes"] == label]
        for i, (key, title, cmap, fmt) in enumerate([
            ("det_p_miss", "Deterministic plan: scenarios missing 99.9%", "Blues", "{:.0%}"),
            ("chance_carbon_premium_pct", "95%-chance plan: extra carbon", "Oranges",
             "+{:.1f}%"),
        ]):
            m = np.array([[next(r[key] for r in sub if r["error_width"] == wd
                                and r["mttr_hi_h"] == mh) for mh in MTTR_HI] for wd in WIDTHS])
            ax = axes[i, j]
            ax.imshow(m, cmap=cmap, aspect="auto", origin="lower",
                      vmin=0, vmax=1 if key == "det_p_miss" else max(6, m.max()))
            for a in range(len(WIDTHS)):
                for b in range(len(MTTR_HI)):
                    v = m[a, b]
                    dark = v > (0.55 if key == "det_p_miss" else 0.55 * max(6, m.max()))
                    ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=8,
                            color=SURFACE if dark else INK)
            ax.set_xticks(range(len(MTTR_HI)), [f"{h:g} h" for h in MTTR_HI])
            ax.set_yticks(range(len(WIDTHS)), [f"x{x:g}" for x in WIDTHS])
            ax.set_xlabel("Upper end of recovery-time range")
            ax.set_ylabel("Error-rate range (mode / w .. mode * w)")
            ax.set_title(f"{title}\n{label}", loc="left", fontsize=8.5, color=INK)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/figures/uncertainty_sensitivity.png", dpi=200, facecolor=SURFACE)
    print("wrote results/uncertainty_sensitivity.csv and docs/figures/uncertainty_sensitivity.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
