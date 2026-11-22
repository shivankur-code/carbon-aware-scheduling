"""Reliability-aware provisioning of an LLM inference fleet.

Design choices per fleet: tensor-parallel degree g (GPUs per model replica), SM clock level c
and the number of spare replicas S. A GPU error takes down its whole replica (blast radius g)
for the recovery time. Spares are sized so that the fleet keeps enough replicas up with the
target probability (binomial k-of-n, independent errors).

The design space is small and discrete, so the optimum is found by exhaustive enumeration
(exact). The epsilon-constraint front over availability targets gives the carbon cost of
reliability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from calf.models import reliability as rel
from calf.models.derating import CLOCKLOCK_DECODE_CURVE, CapCurve

HOURS = rel.HOURS_PER_YEAR


@dataclass(frozen=True)
class Workload:
    name: str
    demand_tok_s: float          # sustained decode tokens/s the fleet must serve
    model_mem_gb: float          # weights in serving precision
    kv_overhead: float = 0.3     # ASSUMPTION: KV cache + activations as a share of weights


@dataclass(frozen=True)
class Gpu:
    name: str
    mem_gb: float
    tput_tp1: float              # ASSUMPTION: decode tokens/s per GPU at TP=1, max clock
    active_kw: float             # ASSUMPTION: average draw incl. server share while serving
    idle_kw: float               # ASSUMPTION: powered spare (incl. server share)
    embodied_kg: float           # ASSUMPTION: GPU + server share, kgCO2e
    lifetime_y: float            # amortization period


@dataclass(frozen=True)
class Env:
    carbon_intensity: float      # kgCO2e/kWh
    error_rate: float            # node-interrupting errors per GPU-year (Cui et al. SC'25)
    mttr_h: float                # recovery time per error, hours (Cui et al. SC'25)
    pue: float = 1.2
    availability_target: float = 0.999


# ASSUMPTION: per-GPU throughput efficiency vs tensor-parallel degree (communication overhead).
# Swept in experiments; a larger group can also be MORE efficient per GPU (more KV memory ->
# bigger batches), which is why break-even analysis is reported instead of a single answer.
TP_EFF = {1: 1.0, 2: 0.92, 4: 0.85, 8: 0.75}


def min_tp(w: Workload, gpu: Gpu) -> int:
    need = w.model_mem_gb * (1.0 + w.kv_overhead)
    g = 1
    while g * gpu.mem_gb < need:
        g *= 2
    return g


@dataclass(frozen=True)
class Design:
    tp: int
    clock: float
    spares: int | None = None    # None = size spares for env.availability_target


@dataclass(frozen=True)
class Evaluation:
    design: Design
    replicas: int
    spares: int
    gpus_total: int
    q_replica: float             # probability a replica is down
    availability: float          # P(enough replicas up)
    embodied_active: float       # kgCO2e / year
    embodied_spares: float
    operational: float
    spare_idle: float

    @property
    def total(self) -> float:
        return self.embodied_active + self.embodied_spares + self.operational + self.spare_idle

    @property
    def spare_pct(self) -> float:
        return 100.0 * self.spares / self.replicas


def evaluate(d: Design, w: Workload, gpu: Gpu, env: Env, tp_eff: dict[int, float] = TP_EFF,
             clock_curve: CapCurve = CLOCKLOCK_DECODE_CURVE) -> Evaluation:
    tau, rho, _ = clock_curve.at(d.clock)
    per_replica = gpu.tput_tp1 * tp_eff[d.tp] * d.tp * tau
    replicas = math.ceil(w.demand_tok_s / per_replica)
    q = rel.unavailability(env.error_rate * d.tp, env.mttr_h)
    spares = (d.spares if d.spares is not None
              else rel.spares_needed(replicas, q, env.availability_target, replicas * 4))
    n_active, n_spare = replicas * d.tp, spares * d.tp
    energy_k = HOURS * env.pue * env.carbon_intensity
    return Evaluation(
        design=d, replicas=replicas, spares=spares, gpus_total=n_active + n_spare,
        q_replica=q, availability=rel.pool_availability(replicas, spares, q),
        embodied_active=n_active * gpu.embodied_kg / gpu.lifetime_y,
        embodied_spares=n_spare * gpu.embodied_kg / gpu.lifetime_y,
        operational=n_active * gpu.active_kw * rho * energy_k,
        spare_idle=n_spare * gpu.idle_kw * energy_k,
    )


def designs(w: Workload, gpu: Gpu, clocks=(1.0, 0.8, 0.39), tps=(1, 2, 4, 8)) -> list[Design]:
    g0 = min_tp(w, gpu)
    return [Design(tp, c) for tp in tps if tp >= g0 for c in clocks]


def optimize(w: Workload, gpu: Gpu, env: Env, **kw) -> Evaluation:
    """Minimum-carbon design meeting env.availability_target (exhaustive, exact)."""
    return min((evaluate(d, w, gpu, env, **kw) for d in designs(w, gpu)), key=lambda e: e.total)


def availability_front(w: Workload, gpu: Gpu, env: Env,
                       targets=(0.99, 0.999, 0.9999, 0.99999), **kw) -> list[Evaluation]:
    """Epsilon-constraint front: min carbon subject to availability >= each target."""
    return [optimize(w, gpu, replace(env, availability_target=t), **kw) for t in targets]


def breakeven_efficiency(w: Workload, gpu: Gpu, env: Env, tp_small: int, tp_big: int,
                         clock: float = 1.0, lo: float = 0.3, hi: float = 2.0) -> float:
    """Per-GPU efficiency the bigger TP group needs (relative to the smaller group's) to match
    its carbon, given the bigger group's larger blast radius. Bisection on the ratio."""
    base = evaluate(Design(tp_small, clock), w, gpu, env, tp_eff={tp_small: 1.0}).total

    def gap(r: float) -> float:
        return evaluate(Design(tp_big, clock), w, gpu, env, tp_eff={tp_big: r}).total - base

    if gap(hi) > 0:
        return float("inf")
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return hi
