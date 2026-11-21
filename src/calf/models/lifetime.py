"""Steady-state carbon of a GPU fleet as a function of its replacement lifetime L.

The fleet serves a fixed workload. Purchases are staggered, so at any time the fleet holds
GPUs of every age in [0, L]. Each GPU keeps the throughput of the generation it was bought
in; new generations improve throughput-per-watt by 2x every ``doubling_years``.

Annual carbon = embodied (active + spare GPUs, amortized over L)
              + embodied of parts replaced after unrepairable failures
              + operational energy of active GPUs
              + idle energy of spares.

Setting ``reliability_aware=False`` drops spares and failure replacements: this is the
failure-oblivious planner the project compares against (an EcoServe-style lifetime choice
that, as far as we have read, does not model failures).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from calf.models import reliability as rel


@dataclass(frozen=True)
class FleetParams:
    # Every default is an ASSUMPTION for the pilot; see docs/assumptions.md.
    frontier_gpus: float = 1000.0        # GPUs needed if the whole fleet were brand new
    embodied_kg_per_gpu: float = 1000.0  # GPU + its share of server, kgCO2e
    avg_power_kw: float = 0.40           # average draw of an active GPU incl. its server share
    idle_power_kw: float = 0.10          # draw of a powered-on spare
    pue: float = 1.2
    carbon_intensity: float = 0.4        # kgCO2e/kWh
    doubling_years: float = 3.5          # perf/W doubling period of new generations
    weibull_shape: float = 2.0
    weibull_scale_years: float = 8.0
    mttr_hours: float = 72.0
    drain_fraction: float = 0.0          # share of time a GPU is out for planned maintenance
    unrepairable_fraction: float = 0.3   # failures that need a replacement part
    part_embodied_fraction: float = 0.5  # replacement part as a share of GPU embodied carbon
    pool_size: int = 64                  # GPUs per serving pool (one model deployment)
    availability_target: float = 0.999


def mean_relative_throughput(lifetime_years: float, doubling_years: float) -> float:
    """Average throughput per GPU of a staggered fleet relative to an all-new fleet."""
    x = lifetime_years / doubling_years
    if x < 1e-9:
        return 1.0
    return (1.0 - 2.0 ** (-x)) / (x * math.log(2.0))


@dataclass(frozen=True)
class CarbonBreakdown:
    lifetime_years: float
    active_gpus: float
    spare_fraction: float
    embodied: float
    replacement_embodied: float
    operational: float
    spare_idle: float
    pool_availability_no_spares: float

    @property
    def total(self) -> float:
        return self.embodied + self.replacement_embodied + self.operational + self.spare_idle


def annual_carbon(
    lifetime_years: float, p: FleetParams, reliability_aware: bool = True
) -> CarbonBreakdown:
    """Annual fleet carbon (kgCO2e/yr) for replacement lifetime ``lifetime_years``."""
    n_active = p.frontier_gpus / mean_relative_throughput(lifetime_years, p.doubling_years)

    # Spares are sized for the oldest GPUs in the fleet (worst-case age = L).
    h_end = rel.weibull_hazard(lifetime_years, p.weibull_shape, p.weibull_scale_years)
    q_end = rel.unavailability(h_end, p.mttr_hours, p.drain_fraction)
    avail_no_spares = rel.pool_availability(p.pool_size, 0, q_end)

    if reliability_aware:
        s = rel.spares_needed(p.pool_size, q_end, p.availability_target)
        spare_frac = s / p.pool_size
        h_mean = rel.mean_hazard(lifetime_years, p.weibull_shape, p.weibull_scale_years)
        repl = (
            n_active * h_mean * p.unrepairable_fraction
            * p.part_embodied_fraction * p.embodied_kg_per_gpu
        )
    else:
        spare_frac, repl = 0.0, 0.0

    hours = rel.HOURS_PER_YEAR
    embodied = n_active * (1.0 + spare_frac) * p.embodied_kg_per_gpu / lifetime_years
    operational = n_active * p.avg_power_kw * hours * p.pue * p.carbon_intensity
    spare_idle = n_active * spare_frac * p.idle_power_kw * hours * p.pue * p.carbon_intensity
    return CarbonBreakdown(
        lifetime_years, n_active, spare_frac, embodied, repl, operational, spare_idle,
        avail_no_spares,
    )


def optimal_lifetime(
    p: FleetParams,
    reliability_aware: bool = True,
    grid: np.ndarray | None = None,
) -> tuple[float, CarbonBreakdown]:
    """Grid-search the carbon-minimizing lifetime."""
    grid = np.arange(0.5, 12.01, 0.1) if grid is None else grid
    best = min((annual_carbon(float(L), p, reliability_aware) for L in grid), key=lambda b: b.total)
    return best.lifetime_years, best


def oblivious_regret(p: FleetParams) -> dict[str, float]:
    """Carbon penalty of planning without failures, then living with them.

    The failure-oblivious lifetime is evaluated under the true (failure-aware) model,
    i.e. the operator is forced to add the spares the SLO actually needs.
    """
    l_obl, promised = optimal_lifetime(p, reliability_aware=False)
    l_rel, best = optimal_lifetime(p, reliability_aware=True)
    realized = annual_carbon(l_obl, p, reliability_aware=True)
    return {
        "lifetime_oblivious": l_obl,
        "lifetime_aware": l_rel,
        "promised_kg": promised.total,
        "realized_kg": realized.total,
        "aware_kg": best.total,
        "regret_pct": 100.0 * (realized.total - best.total) / best.total,
        "availability_if_no_spares": realized.pool_availability_no_spares,
    }
