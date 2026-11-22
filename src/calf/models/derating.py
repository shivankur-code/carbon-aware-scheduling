"""Power capping ("derating") as a joint lever on operational carbon, wear-out and fleet size.

A power cap c (fraction of TDP) changes three measured quantities of an inference GPU:
relative throughput tau(c), relative average power rho(c) and steady temperature T(c).
Measurement harness: ``gpu_bench/``. This module turns them into fleet effects:

* more GPUs are needed for the same work:   N  ->  N / tau(c)
* each GPU draws less power:                P  ->  P * rho(c)
* each GPU ages more slowly (Arrhenius accelerated-life time scaling of the Weibull scale):
  scale -> scale * exp(Ea/k * (1/T(c) - 1/T(1)))

The activation energy Ea is the weakest link (ASSUMPTION). Ea = 0 switches the thermal
aging effect off, which every experiment must report as a lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from calf.models.lifetime import FleetParams, annual_carbon

BOLTZMANN_EV = 8.617333262e-5


@dataclass(frozen=True)
class CapCurve:
    cap: tuple[float, ...]             # power cap / TDP, ascending, last entry = 1.0
    rel_throughput: tuple[float, ...]  # tokens/s relative to cap = 1.0
    rel_power: tuple[float, ...]       # measured average power relative to cap = 1.0
    temp_c: tuple[float, ...]          # steady GPU temperature, deg C
    source: str = "ASSUMPTION placeholder"

    def at(self, c: float) -> tuple[float, float, float]:
        xs = self.cap
        return (
            float(np.interp(c, xs, self.rel_throughput)),
            float(np.interp(c, xs, self.rel_power)),
            float(np.interp(c, xs, self.temp_c)),
        )

    @classmethod
    def from_rows(cls, rows: list[dict], source: str) -> CapCurve:
        """Build from gpu_bench summary rows (keys: power_cap_frac, tokens_per_s,
        avg_power_w, gpu_temp_c), normalized to the uncapped row."""
        rows = sorted(rows, key=lambda r: r["power_cap_frac"])
        ref = rows[-1]
        return cls(
            tuple(r["power_cap_frac"] for r in rows),
            tuple(r["tokens_per_s"] / ref["tokens_per_s"] for r in rows),
            tuple(r["avg_power_w"] / ref["avg_power_w"] for r in rows),
            tuple(r["gpu_temp_c"] for r in rows),
            source,
        )


# Shape only; replace with gpu_bench measurements (E-G1) before drawing any conclusion.
PLACEHOLDER_CURVE = CapCurve(
    cap=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    rel_throughput=(0.78, 0.86, 0.92, 0.96, 0.99, 1.0),
    rel_power=(0.55, 0.64, 0.73, 0.82, 0.91, 1.0),
    temp_c=(58.0, 62.0, 66.0, 70.0, 74.0, 78.0),
)


def arrhenius_life_factor(temp_c: float, ref_temp_c: float, ea_ev: float) -> float:
    """Multiplier on characteristic life when running at temp_c instead of ref_temp_c."""
    t, t0 = temp_c + 273.15, ref_temp_c + 273.15
    return math.exp(ea_ev / BOLTZMANN_EV * (1.0 / t - 1.0 / t0))


def derated_params(p: FleetParams, curve: CapCurve, cap: float, ea_ev: float) -> FleetParams:
    tau, rho, temp = curve.at(cap)
    _, _, temp_full = curve.at(1.0)
    return replace(
        p,
        frontier_gpus=p.frontier_gpus / tau,
        avg_power_kw=p.avg_power_kw * rho,
        weibull_scale_years=p.weibull_scale_years
        * arrhenius_life_factor(temp, temp_full, ea_ev),
    )


@dataclass(frozen=True)
class JointResult:
    cap: float
    lifetime_years: float
    total_kg: float


def joint_optimum(
    p: FleetParams,
    curve: CapCurve,
    ea_ev: float,
    caps: np.ndarray | None = None,
    lifetimes: np.ndarray | None = None,
) -> JointResult:
    """Carbon-minimizing (power cap, lifetime) pair under the failure-aware model."""
    caps = np.round(np.arange(curve.cap[0], 1.0001, 0.05), 3) if caps is None else caps
    lifetimes = np.arange(0.5, 15.01, 0.25) if lifetimes is None else lifetimes
    best = JointResult(1.0, float("nan"), float("inf"))
    for c in caps:
        q = derated_params(p, curve, float(c), ea_ev)
        for L in lifetimes:
            tot = annual_carbon(float(L), q, reliability_aware=True).total
            if tot < best.total_kg:
                best = JointResult(float(c), float(L), tot)
    return best


# SM clock locking during memory-bound LLM decode, from Ma et al., "The Illusion of Power
# Capping in LLM Decode", PPAM 2026 (arXiv 2605.11999), one H200 + vLLM, ~4B models, <=4K
# output tokens. "cap" here is the locked SM clock as a fraction of max (1980 MHz):
#   1590 MHz (0.80): <0.1% throughput change, ~9% less power than 1980 MHz.
#   780 MHz (0.39):  20-32% less decode energy (midpoint 26% used), ~1% throughput loss.
# The paper does not report temperature, so temperatures are held equal: this curve makes
# NO thermal-aging claim (Arrhenius factor = 1 regardless of Ea).
CLOCKLOCK_DECODE_CURVE = CapCurve(
    cap=(0.39, 0.80, 1.0),
    rel_throughput=(0.99, 0.999, 1.0),
    rel_power=(0.74, 1 / 1.09, 1.0),  # 0.74/0.99 -> ~25% less energy per token
    temp_c=(70.0, 70.0, 70.0),
    source="Ma et al. PPAM 2026 (clock locking, memory-bound decode)",
)
