"""Power cap as a joint carbon lever: carbon saving vs grid intensity.

Usage: python experiments/derating.py [configs/derating.yaml]
Falls back to the placeholder cap curve when no gpu_bench measurements exist.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from calf.models.derating import PLACEHOLDER_CURVE, CapCurve, joint_optimum
from calf.models.lifetime import FleetParams, annual_carbon, optimal_lifetime

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_curves(cfg) -> list[CapCurve]:
    curves = []
    for summary in sorted(Path(cfg["measurements"]).glob("*/summary.csv")):
        with summary.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("experiment") == "load"]
        want = json.loads(cfg["measured_load"])
        rows = [r for r in rows if json.loads(r["load"]) == want and r.get("tp", "1") == "1"]
        for model in sorted({r["model"] for r in rows}):
            sel = [r for r in rows if r["model"] == model
                   and float(r["slo_attainment"]) >= cfg["min_slo_attainment"]]
            if len(sel) < 2 or max(float(r["power_cap_frac"]) for r in sel) < 0.999:
                continue
            num = [{k: float(r[k]) for k in ("power_cap_frac", "tokens_per_s", "avg_power_w",
                                             "gpu_temp_c")} for r in sel]
            curves.append(CapCurve.from_rows(num, f"{summary.parent.name}:{Path(model).name}"))
    return curves or [PLACEHOLDER_CURVE]


def main(cfg_path: str = "configs/derating.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    base = replace(FleetParams(), **cfg["fleet"])
    s = cfg["carbon_intensity_kg_per_kwh"]
    cis = np.linspace(s["start"], s["stop"], s["num"])
    lifetimes = np.arange(0.5, 15.01, 0.25)
    rows = []
    for curve in load_curves(cfg):
        for ci in cis:
            p = replace(base, carbon_intensity=float(ci))
            l_obl, _ = optimal_lifetime(p, False, lifetimes)
            ref = annual_carbon(l_obl, p, True).total  # oblivious plan, realized, no cap
            _, aware = optimal_lifetime(p, True, lifetimes)
            row = {"curve": curve.source, "carbon_intensity": float(ci),
                   "save_failure_aware_pct": 100 * (1 - aware.total / ref)}
            for ea in cfg["activation_energy_ev"]:
                j = joint_optimum(p, curve, ea, lifetimes=lifetimes)
                row[f"save_joint_ea{ea}_pct"] = 100 * (1 - j.total_kg / ref)
                row[f"cap_ea{ea}"] = j.cap
                row[f"lifetime_ea{ea}"] = j.lifetime_years
            rows.append(row)
    out = Path(cfg["output"]["csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    curve0 = rows[0]["curve"]
    sub = [r for r in rows if r["curve"] == curve0]
    x = np.array([r["carbon_intensity"] for r in sub]) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), facecolor=SURFACE)
    axes[0].plot(x, [r["save_failure_aware_pct"] for r in sub], color=INK2, lw=2, ls="--",
                 label="failure-aware lifetime only")
    for i, ea in enumerate(cfg["activation_energy_ev"]):
        lab = f"+ power cap, Ea = {ea} eV"
        axes[0].plot(x, [r[f"save_joint_ea{ea}_pct"] for r in sub], color=SERIES[i], lw=2,
                     label=lab)
        axes[1].plot(x, [100 * r[f"cap_ea{ea}"] for r in sub], color=SERIES[i], lw=2, label=lab)
    axes[0].set_title("Carbon saved vs failure-oblivious plan (%)", loc="left", fontsize=9)
    axes[1].set_title("Carbon-optimal power cap (% of default limit)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("Grid carbon intensity (gCO2e/kWh)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    note = ("PILOT - placeholder cap curve" if curve0.startswith("ASSUMPTION")
            else f"cap curve measured: {curve0}")
    fig.suptitle(note, fontsize=8, color=INK2, x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(cfg["output"]["figure"], dpi=200, facecolor=SURFACE)
    print(f"wrote {out} and {cfg['output']['figure']} ({len(rows)} rows)")


if __name__ == "__main__":
    main(*sys.argv[1:])
