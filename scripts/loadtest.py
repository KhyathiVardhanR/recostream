#!/usr/bin/env python3
"""Open-loop load test for the serving API using `oha` (brew install oha).

Requests are issued at a fixed rate regardless of response time, so queueing
delay shows up in the percentiles as it would for real traffic. Each request
targets a random user id.

    .venv/bin/python scripts/loadtest.py --rps 250 500 1000 2000 3000 4000
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_level(base: str, rps: int, duration: int, conns: int, max_user: int) -> dict:
    digits = len(str(max_user))
    url = base.replace(".", "[.]") + "/recommend/[1-9][0-9]{0,%d}" % (digits - 2)
    out = subprocess.run(
        ["oha", "-q", str(rps), "-z", f"{duration}s", "-c", str(conns), "--no-tui", "--output-format", "json",
         "--rand-regex-url", url],
        capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    s, p = r["summary"], r["latencyPercentiles"]
    return {"rps_target": rps, "rps_achieved": s["requestsPerSec"], "success": s["successRate"],
            "p50": p["p50"] * 1000, "p95": p["p95"] * 1000, "p99": p["p99"] * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rps", type=int, nargs="+", default=[250, 500, 1000, 2000, 3000, 4000])
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--connections", type=int, default=1024)
    ap.add_argument("--max-user", type=int, default=99999)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "serving.md"))
    a = ap.parse_args()
    rows = []
    for rps in a.rps:
        r = run_level(a.url, rps, a.duration, a.connections, a.max_user)
        rows.append(r)
        print(f"[load] {rps:>5} rps -> {r['rps_achieved']:7.0f} achieved | p50 {r['p50']:6.1f} ms  "
              f"p95 {r['p95']:6.1f} ms  p99 {r['p99']:6.1f} ms  success {r['success'] * 100:.1f}%", flush=True)
    lines = [f"\n### Serving latency ({a.label})\n",
             "| Target req/s | Achieved req/s | p50 (ms) | p95 (ms) | p99 (ms) | Success |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['rps_target']} | {r['rps_achieved']:.0f} | {r['p50']:.1f} | {r['p95']:.1f} | "
                     f"{r['p99']:.1f} | {r['success'] * 100:.1f}% |")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
