"""End-to-end check of the load generator against a fake vLLM-style streaming server."""

import asyncio
import json

from aiohttp import web

from gpu_bench.loadgen import run_load
from gpu_bench.summarize import request_metrics


async def _completions(request):
    body = await request.json()
    n = body["max_tokens"]
    resp = web.StreamResponse()
    await resp.prepare(request)
    for _ in range(n):
        await asyncio.sleep(0.001)
        chunk = {"choices": [{"text": " tok"}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": n}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def _run(cfg):
    app = web.Application()
    app.router.add_post("/v1/completions", _completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        return await run_load(f"http://127.0.0.1:{port}", "fake", cfg, duration_s=1.0)
    finally:
        await runner.cleanup()


WL = {"input_median": 20, "input_max": 50, "output_median": 16, "output_max": 32}


def test_closed_loop():
    res = asyncio.run(_run({**WL, "concurrency": 4}))
    assert res and all(r.ok for r in res)
    assert all(r.output_tokens >= 4 and r.ttft_s > 0 for r in res)
    m = request_metrics([r.__dict__ for r in res], 1.0, 2.0, 0.1)
    assert m["slo_attainment"] == 1.0


def test_open_loop_and_connection_failures():
    res = asyncio.run(_run({**WL, "rate": 20}))
    assert len(res) > 5 and all(r.ok for r in res)
    # nothing listening -> every request fails but the run completes
    bad = asyncio.run(run_load("http://127.0.0.1:9", "fake", {**WL, "rate": 20}, 0.3))
    assert bad and not any(r.ok for r in bad)
