"""Planning under uncertainty: triangular fuzzy numbers (alpha-cuts) and Monte Carlo scenarios.

The fuzzy treatment follows the multi-objective fuzzy MILP tradition (triangular fuzzy
parameters, alpha-cut intervals, a satisfaction level alpha chosen by the decision maker):
alpha = 1 plans at the most likely values (deterministic plan); alpha -> 0 plans for the
widest plausible range (most conservative).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TFN:
    """Triangular fuzzy number (lower, mode, upper)."""

    lo: float
    mode: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo <= self.mode <= self.hi:
            raise ValueError(f"need lo <= mode <= hi, got {self}")

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        a = min(max(alpha, 0.0), 1.0)
        return (self.lo + a * (self.mode - self.lo), self.hi - a * (self.hi - self.mode))

    def centroid(self) -> float:
        return (self.lo + self.mode + self.hi) / 3.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.lo == self.hi:
            return np.full(n, self.mode)
        return rng.triangular(self.lo, self.mode, self.hi, n)


def fuzzy_eval(f: Callable[[dict[str, float]], float], params: dict[str, TFN],
               alpha: float) -> tuple[float, float]:
    """Alpha-cut interval of f(params) by the vertex method (exact for monotone f)."""
    names = list(params)
    cuts = [params[k].alpha_cut(alpha) for k in names]
    vals = [f(dict(zip(names, corner))) for corner in itertools.product(*cuts)]
    return min(vals), max(vals)


def sample_scenarios(params: dict[str, TFN], n: int, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    cols = {k: v.sample(rng, n) for k, v in params.items()}
    return [{k: float(cols[k][i]) for k in params} for i in range(n)]
