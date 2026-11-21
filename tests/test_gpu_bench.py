import pytest

from calf.models.derating import CapCurve, arrhenius_life_factor, derated_params, joint_optimum
from calf.models.lifetime import FleetParams
from gpu_bench.summarize import combine, gpu_metrics, pct, request_metrics


def _req(ttft, e2e, out, ok=True):
    return {"ok": ok, "ttft_s": ttft, "e2e_s": e2e, "output_tokens": out, "prompt_tokens": 10}


def test_pct():
    assert pct([1, 2, 3, 4, 5], 0.5) == 3
    assert pct([], 0.5) != pct([], 0.5)  # nan


def test_request_metrics_slo():
    reqs = [_req(0.5, 0.5 + 99 * 0.05, 100), _req(3.0, 3.0 + 99 * 0.05, 100),
            _req(0.5, 0.5 + 99 * 0.2, 100), _req(0, 0, 0, ok=False)]
    m = request_metrics(reqs, 10.0, ttft_slo_s=2.0, tpot_slo_s=0.1)
    assert m["failed"] == 1
    assert m["slo_attainment"] == pytest.approx(0.25)
    assert m["tokens_per_s"] == pytest.approx(30.0)


def test_gpu_metrics_energy_counter_and_power():
    s = [{"t": t, "gpu": 0, "power_w": 200.0, "energy_j": 200.0 * t, "temp_c": 60 + t,
          "sm_mhz": 1500} for t in range(11)]
    g = gpu_metrics(s, 0, 10)
    assert g["energy_j"] == pytest.approx(2000.0)
    assert g["avg_power_w"] == pytest.approx(200.0)
    row = combine({"output_tokens": 1000}, g)
    assert row["j_per_output_token"] == pytest.approx(2.0)


def test_gpu_metrics_trapezoid_fallback():
    s = [{"t": t, "gpu": 0, "power_w": 100.0, "energy_j": float("nan"), "temp_c": 50,
          "sm_mhz": 1000} for t in range(5)]
    assert gpu_metrics(s, 0, 4)["energy_j"] == pytest.approx(400.0)


def test_arrhenius():
    assert arrhenius_life_factor(70, 70, 0.7) == pytest.approx(1.0)
    assert arrhenius_life_factor(60, 80, 0.7) > 1.0
    assert arrhenius_life_factor(60, 80, 0.0) == pytest.approx(1.0)


def test_capcurve_from_rows_and_derating():
    rows = [{"power_cap_frac": c, "tokens_per_s": 1000 * c ** 0.3, "avg_power_w": 300 * c,
             "gpu_temp_c": 40 + 40 * c} for c in (0.5, 0.75, 1.0)]
    curve = CapCurve.from_rows(rows, "test")
    assert curve.rel_throughput[-1] == pytest.approx(1.0)
    q = derated_params(FleetParams(), curve, 0.5, ea_ev=0.7)
    assert q.frontier_gpus > FleetParams().frontier_gpus
    assert q.weibull_scale_years > FleetParams().weibull_scale_years
    j = joint_optimum(FleetParams(), curve, 0.0)
    assert 0.5 <= j.cap <= 1.0
