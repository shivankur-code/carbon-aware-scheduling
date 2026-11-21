"""Pilot experiment: how failures shift the carbon-optimal GPU lifetime.

Usage: python experiments/pilot_lifetime.py [configs/pilot_lifetime.yaml]
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

from calf.models.lifetime import FleetParams, oblivious_regret

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated default palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(cfg_path: str = "configs/pilot_lifetime.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    base = FleetParams(**cfg["base"])
    s = cfg["sweep"]["carbon_intensity_kg_per_kwh"]
    cis = np.linspace(s["start"], s["stop"], s["num"])
    scales = cfg["sweep"]["weibull_scale_years"]

    rows = []
    for scale in scales:
        for ci in cis:
            r = oblivious_regret(replace(base, carbon_intensity=float(ci),
                                         weibull_scale_years=float(scale)))
            rows.append({"weibull_scale_years": scale, "carbon_intensity": float(ci), **r})

    out_csv = Path(cfg["output"]["csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=SURFACE)
    g = lambda x: np.array(x) * 1000  # kg/kWh -> g/kWh

    for i, scale in enumerate(scales):
        sub = [r for r in rows if r["weibull_scale_years"] == scale]
        x = g([r["carbon_intensity"] for r in sub])
        lab = f"failure-aware, Weibull scale {scale} y"
        axes[0].plot(x, [r["lifetime_aware"] for r in sub], color=SERIES[i], lw=2, label=lab)
        axes[1].plot(x, [r["regret_pct"] for r in sub], color=SERIES[i], lw=2, label=lab)
        axes[2].plot(x, [100 * r["availability_if_no_spares"] for r in sub], color=SERIES[i],
                     lw=2, label=lab)
    sub = [r for r in rows if r["weibull_scale_years"] == scales[0]]
    axes[0].plot(g([r["carbon_intensity"] for r in sub]),
                 [r["lifetime_oblivious"] for r in sub], color=INK2, lw=2, ls="--",
                 label="failure-oblivious")

    titles = ["Carbon-optimal GPU lifetime (years)",
              "Extra carbon from planning without failures (%)",
              "Pool availability of oblivious plan, no spares (%)"]
    for ax, t in zip(axes, titles):
        ax.set_facecolor(SURFACE)
        ax.set_title(t, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Grid carbon intensity (gCO2e/kWh)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.suptitle("PILOT - placeholder parameters, not a finding", fontsize=8, color=INK2,
                 x=0.99, ha="right")
    fig.tight_layout()
    out_fig = Path(cfg["output"]["figure"])
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200, facecolor=SURFACE)
    print(f"wrote {out_csv} and {out_fig}")


if __name__ == "__main__":
    main(*sys.argv[1:])
