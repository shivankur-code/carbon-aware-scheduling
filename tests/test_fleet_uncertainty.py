from dataclasses import replace

import pytest

from calf.models.derating import CLOCKLOCK_DECODE_CURVE
from calf.optimize.fleet import (
    Design,
    Env,
    Gpu,
    Workload,
    availability_front,
    breakeven_efficiency,
    evaluate,
    min_tp,
    optimize,
)
from calf.optimize.uncertainty import TFN, fuzzy_eval, sample_scenarios

GPU = Gpu("g", 80, 1000, 0.45, 0.10, 1000, 4)
W70 = Workload("70B", 400_000, 140)
ENV = Env(0.25, 7.5, 2.2)


def test_min_tp_memory():
    assert min_tp(W70, GPU) == 4
    assert min_tp(Workload("8B", 1, 16), GPU) == 1


def test_spares_meet_target_and_grow_with_recovery_time():
    fast = evaluate(Design(4, 1.0), W70, GPU, replace(ENV, mttr_h=0.1))
    slow = evaluate(Design(4, 1.0), W70, GPU, replace(ENV, mttr_h=24))
    assert fast.availability >= ENV.availability_target
    assert slow.availability >= ENV.availability_target
    assert slow.spares > fast.spares


def test_clock_lock_cuts_operational_carbon():
    hi = evaluate(Design(4, 1.0), W70, GPU, ENV)
    lo = evaluate(Design(4, 0.39), W70, GPU, ENV)
    assert lo.operational < 0.8 * hi.operational
    assert CLOCKLOCK_DECODE_CURVE.at(0.39)[0] > 0.98


def test_optimize_respects_min_tp_and_front_is_monotone():
    best = optimize(W70, GPU, ENV)
    assert best.design.tp >= 4
    front = availability_front(W70, GPU, ENV)
    totals = [e.total for e in front]
    assert totals == sorted(totals)


def test_breakeven_grows_with_recovery_time():
    fast = breakeven_efficiency(W70, GPU, replace(ENV, mttr_h=0.1), 4, 8)
    slow = breakeven_efficiency(W70, GPU, replace(ENV, mttr_h=72), 4, 8)
    assert 1.0 <= fast < slow


def test_tfn_alpha_cuts_and_vertex_method():
    t = TFN(1, 2, 4)
    assert t.alpha_cut(1) == (2, 2)
    assert t.alpha_cut(0) == (1, 4)
    assert t.centroid() == pytest.approx(7 / 3)
    lo, hi = fuzzy_eval(lambda p: p["a"] * p["b"], {"a": TFN(1, 2, 3), "b": TFN(2, 2, 5)}, 0.0)
    assert (lo, hi) == (2, 15)
    with pytest.raises(ValueError):
        TFN(3, 2, 1)


def test_scenarios_within_support():
    s = sample_scenarios({"x": TFN(1, 2, 3)}, 500, seed=1)
    xs = [d["x"] for d in s]
    assert min(xs) >= 1 and max(xs) <= 3
