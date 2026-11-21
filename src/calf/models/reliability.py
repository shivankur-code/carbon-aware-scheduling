"""Age-dependent failure and spare-capacity model.

Failures follow a Weibull hazard (shape k > 1 means wear-out). A GPU that fails is out of
service for MTTR hours. Spares are sized so that a pool of N active GPUs stays fully
available with a target probability, assuming independent failures (an assumption that
the plan says to relax later with correlated/batch failures).
"""

from __future__ import annotations

from scipy.stats import binom

HOURS_PER_YEAR = 8760.0


def weibull_hazard(age_years: float, shape: float, scale_years: float) -> float:
    """Instantaneous failure rate (failures per GPU-year) at a given age."""
    if age_years <= 0:
        return 0.0 if shape > 1 else (shape / scale_years)
    return (shape / scale_years) * (age_years / scale_years) ** (shape - 1)


def cumulative_hazard(age_years: float, shape: float, scale_years: float) -> float:
    """Expected failures per GPU slot from age 0 to ``age_years`` (renewal-free approx.)."""
    return (max(age_years, 0.0) / scale_years) ** shape


def unavailability(hazard_per_year: float, mttr_hours: float, drain_fraction: float = 0.0) -> float:
    """Steady-state probability that one GPU is down (failure repair + planned drain)."""
    mttr_years = mttr_hours / HOURS_PER_YEAR
    q_fail = hazard_per_year * mttr_years / (1.0 + hazard_per_year * mttr_years)
    return min(1.0, 1.0 - (1.0 - q_fail) * (1.0 - drain_fraction))


def spares_needed(pool_size: int, q: float, target: float, max_spares: int | None = None) -> int:
    """Smallest S such that P(at most S of the N+S GPUs are down) >= target."""
    if q <= 0:
        return 0
    limit = max_spares if max_spares is not None else pool_size * 4
    for s in range(limit + 1):
        if binom.cdf(s, pool_size + s, q) >= target:
            return s
    return limit


def pool_availability(pool_size: int, spares: int, q: float) -> float:
    """P(at least ``pool_size`` GPUs are up) given ``spares`` extra GPUs."""
    return float(binom.cdf(spares, pool_size + spares, q))


def mean_hazard(lifetime_years: float, shape: float, scale_years: float) -> float:
    """Average failure rate over a lifetime [0, L]."""
    if lifetime_years <= 0:
        return 0.0
    return cumulative_hazard(lifetime_years, shape, scale_years) / lifetime_years


__all__ = [
    "HOURS_PER_YEAR",
    "cumulative_hazard",
    "mean_hazard",
    "pool_availability",
    "spares_needed",
    "unavailability",
    "weibull_hazard",
]
