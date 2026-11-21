from dataclasses import replace

import pytest

from calf.models import reliability as rel
from calf.models.lifetime import (
    FleetParams,
    annual_carbon,
    mean_relative_throughput,
    oblivious_regret,
    optimal_lifetime,
)


def test_weibull_shape1_is_constant():
    assert rel.weibull_hazard(1.0, 1.0, 5.0) == pytest.approx(0.2)
    assert rel.weibull_hazard(7.0, 1.0, 5.0) == pytest.approx(0.2)


def test_wearout_hazard_increases():
    assert rel.weibull_hazard(6, 2, 8) > rel.weibull_hazard(2, 2, 8)


def test_spares_meet_target():
    q = 0.01
    s = rel.spares_needed(64, q, 0.999)
    assert rel.pool_availability(64, s, q) >= 0.999
    assert s == 0 or rel.pool_availability(64, s - 1, q) < 0.999


def test_no_failures_no_spares():
    assert rel.spares_needed(64, 0.0, 0.999) == 0


def test_throughput_factor_limits():
    assert mean_relative_throughput(1e-12, 3.5) == pytest.approx(1.0)
    assert mean_relative_throughput(3.5, 3.5) == pytest.approx(0.5 / 0.6931, rel=1e-3)


def test_aware_never_below_oblivious_cost():
    p = FleetParams()
    for L in (1.0, 4.0, 9.0):
        assert annual_carbon(L, p, True).total >= annual_carbon(L, p, False).total


def test_cleaner_grid_means_longer_lifetime():
    p = FleetParams()
    l_clean, _ = optimal_lifetime(replace(p, carbon_intensity=0.02), reliability_aware=False)
    l_dirty, _ = optimal_lifetime(replace(p, carbon_intensity=0.6), reliability_aware=False)
    assert l_clean > l_dirty


def test_regret_nonnegative():
    r = oblivious_regret(replace(FleetParams(), carbon_intensity=0.05, weibull_scale_years=5))
    assert r["regret_pct"] >= -1e-9
    assert r["lifetime_aware"] <= r["lifetime_oblivious"]
