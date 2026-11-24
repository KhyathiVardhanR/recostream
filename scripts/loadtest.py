#!/usr/bin/env python3
"""Open-loop load test for the serving API: requests are issued on a fixed
schedule (not waiting for responses), so queueing delay shows up in the
latency percentiles as it would for real traffic.

    .venv/bin/python scripts/loadtest.py --rps 100 250 500 1000 --duration 20
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np


async def run_level(url: str, rps: int, duration: float, users: np.ndarray, conns: int) -> dict:
    limits = httpx.Limits(max_connections=conns, max_keepalive_connections=conns)
    lat: list[float] = []
    errors = 0
    rng = np.random.default_rng(rps)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=10.0) as client:
        async def one(u: int):
            nonlocal errors
            t0 = time.perf_counter()
            try:
                r = await client.get(f"/recommend/{u}", params={"k": 10})
                if r.status_code == 200:
                    lat.append(time.perf_counter() - t0)
                else:
                    errors += 1
            except httpx.HTTPError:
                errors += 1

        # warm up connections
        await asyncio.gather(*(one(int(u)) for u in rng.choice(users, min(conns, 50))))
        lat.clear()
        tasks = []
        start = time.perf_counter()
        n = int(rps * duration)
        for j in range(n):
            target = start + j / rps
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(int(rng.choice(users)))))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
    a = np.array(lat) * 1000
    return {"rps_target": rps, "rps_achieved": len(lat) / elapsed, "errors": errors,
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


async def main_async(a):
    users = np.arange(a.users)
    rows = []
    for rps in a.rps:
        r = await run_level(a.url, rps, a.duration, users, a.connections)
        rows.append(r)
        print(f"[load] {rps:>5} rps target -> {r['rps_achieved']:7.1f} achieved | p50 {r['p50']:6.1f} ms  "
              f"p95 {r['p95']:6.1f} ms  p99 {r['p99']:6.1f} ms  errors {r['errors']}", flush=True)
    lines = [f"\n### Serving latency ({a.label})\n",
             "| Target RPS | Achieved RPS | p50 (ms) | p95 (ms) | p99 (ms) | Errors |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['rps_target']} | {r['rps_achieved']:.0f} | {r['p50']:.1f} | {r['p95']:.1f} | "
                     f"{r['p99']:.1f} | {r['errors']} |")
    if a.out:
        with open(a.out, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rps", type=int, nargs="+", default=[100, 250, 500, 1000])
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--users", type=int, default=162541)
    ap.add_argument("--connections", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "serving.md"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
