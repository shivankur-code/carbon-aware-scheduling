"""Spare-GPU overhead (and its embodied carbon) vs recovery time and blast radius.

Usage: python experiments/recovery_spares.py [configs/recovery_spares.yaml]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from calf.models import reliability as rel

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def spare_fraction(n: int, rate: float, blast: int, mttr_h: float, target: float) -> float:
    units = n // blast
    q = rel.unavailability(rate * blast, mttr_h)
    return rel.spares_needed(units, q, target, max_spares=units * 4) * blast / n


def main(cfg_path: str = "configs/recovery_spares.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    r = cfg["recovery_hours"]
    hours = np.geomspace(r["start"], r["stop"], r["num"])
    rows = []
    for gpu, rate in cfg["error_rates_per_gpu_year"].items():
        for g in cfg["blast_radius"]:
            for h in [*hours, *cfg["reference_points_h"]]:
                rows.append({"gpu": gpu, "errors_per_gpu_year": rate, "blast_radius": g,
                             "recovery_h": float(h),
                             "spare_pct": 100 * spare_fraction(cfg["fleet_gpus"], rate, g, h,
                                                               cfg["availability_target"])})
    out = Path(cfg["output"]["csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    gpus = list(cfg["error_rates_per_gpu_year"])
    fig, axes = plt.subplots(1, len(gpus), figsize=(10, 3.6), facecolor=SURFACE, sharey=True)
    for ax, gpu in zip(axes, gpus):
        for i, g in enumerate(cfg["blast_radius"]):
            sub = sorted((x for x in rows if x["gpu"] == gpu and x["blast_radius"] == g),
                         key=lambda x: x["recovery_h"])
            ax.plot([x["recovery_h"] for x in sub], [x["spare_pct"] for x in sub],
                    color=SERIES[i], lw=2, label=f"blast radius {g} GPU" + ("s" if g > 1 else ""))
        for h, lab in [(5 / 60, "5 min"), (0.88, "0.88 h"), (2.2, "2.2 h"), (24, "1 day")]:
            ax.axvline(h, color=GRID, lw=1, zorder=0)
            ax.text(h, 1.02, lab, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                    ha="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{gpu}: {cfg['error_rates_per_gpu_year'][gpu]} errors/GPU-year",
                     loc="left", fontsize=9, color=INK, pad=14)
        ax.set_xlabel("Recovery time per error (hours, log)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"Extra GPUs for {100 * cfg['availability_target']:.1f}% availability (%)")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(cfg["output"]["figure"], dpi=200, facecolor=SURFACE)
    print(f"wrote {out} and {cfg['output']['figure']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
