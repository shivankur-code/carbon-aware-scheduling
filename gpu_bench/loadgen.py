"""Async load generator for an OpenAI-compatible completions endpoint (e.g. `vllm serve`).

Open loop (Poisson arrivals at `rate` req/s) or closed loop (`concurrency` users).
Output length is controlled with max_tokens + ignore_eos (a vLLM extension).
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass

import aiohttp

WORDS = ["carbon", "grid", "power", "energy", "model", "fleet", "token", "latency", "cluster", "server", "memory", "battery", "solar", "wind", "hydro", "cooling", "rack", "spare", "failure", "repair", "lifetime"]


@dataclass
class ReqResult:
    rid: int
    t_send: float
    ttft_s: float
    e2e_s: float
    prompt_tokens: int
    output_tokens: int
    ok: bool
    error: str = ""


def make_prompt(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def sample_lengths(rng: random.Random, cfg: dict) -> tuple[int, int]:
    """Lognormal input/output lengths clipped to [min, max]."""
    def ln(mu_med, sigma, lo, hi):
        return int(min(max(rng.lognormvariate(math.log(mu_med), sigma), lo), hi))
    return (ln(cfg["input_median"], cfg.get("input_sigma", 0.6), 8, cfg["input_max"]),
            ln(cfg["output_median"], cfg.get("output_sigma", 0.6), 4, cfg["output_max"]))


async def one_request(session, url, model, rid, prompt, max_tokens, timeout_s) -> ReqResult:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_send = time.time()
    ttft, usage = float("nan"), {}
    try:
        async with session.post(url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            if r.status != 200:
                return ReqResult(rid, t_send, ttft, time.perf_counter() - t0, 0, 0, False,
                                 f"http {r.status}")
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("choices") and msg["choices"][0].get("text") and math.isnan(ttft):
                    ttft = time.perf_counter() - t0
                if msg.get("usage"):
                    usage = msg["usage"]
        e2e = time.perf_counter() - t0
        return ReqResult(rid, t_send, ttft, e2e, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0), True)
    except Exception as e:  # noqa: BLE001 - record every failure, keep the run going
        return ReqResult(rid, t_send, ttft, time.perf_counter() - t0, 0, 0, False,
                         type(e).__name__)


async def run_load(base_url: str, model: str, cfg: dict, duration_s: float,
                   seed: int = 0) -> list[ReqResult]:
    rng = random.Random(seed)
    url = base_url.rstrip("/") + "/v1/completions"
    timeout_s = cfg.get("timeout_s", 300)
    results: list[ReqResult] = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        t_end = time.perf_counter() + duration_s
        if "concurrency" in cfg:
            rid = 0

            async def user():
                nonlocal rid
                while time.perf_counter() < t_end:
                    rid += 1
                    n_in, n_out = sample_lengths(rng, cfg)
                    results.append(await one_request(session, url, model, rid,
                                                     make_prompt(rng, n_in), n_out, timeout_s))
            await asyncio.gather(*(user() for _ in range(cfg["concurrency"])))
        else:
            tasks, rid = [], 0
            while time.perf_counter() < t_end:
                rid += 1
                n_in, n_out = sample_lengths(rng, cfg)
                tasks.append(asyncio.create_task(one_request(
                    session, url, model, rid, make_prompt(rng, n_in), n_out, timeout_s)))
                await asyncio.sleep(rng.expovariate(cfg["rate"]))
            results.extend(await asyncio.gather(*tasks))
    return results


def to_rows(results: list[ReqResult]) -> list[dict]:
    return [asdict(r) for r in results]
