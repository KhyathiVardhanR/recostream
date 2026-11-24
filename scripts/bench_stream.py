#!/usr/bin/env python3
"""Kafka -> Redis feature-pipeline throughput across consumer configurations.

For each partition count, the topic is pre-loaded with N interaction events;
then a fresh consumer group of C worker processes drains it while we measure
sustained events/sec (first event processed -> last event processed).

    .venv/bin/python scripts/bench_stream.py --events 400000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from recostream import data  # noqa: E402
from recostream.stream import consumer, producer  # noqa: E402


def run_config(topic: str, workers: int, batch: int, n_events: int, item_genres: np.ndarray) -> dict:
    os.environ["KAFKA_TOPIC"] = topic
    counter = mp.Value("q", 0)
    group = f"bench-{uuid.uuid4().hex[:8]}"
    procs = [mp.Process(target=consumer.run_worker, args=(i, group, batch, item_genres, 4.0, counter))
             for i in range(workers)]
    for p in procs:
        p.start()
    t_first = t_last = None
    last_val = 0
    deadline = time.time() + 600
    while time.time() < deadline:
        v = counter.value
        now = time.time()
        if v > 0 and t_first is None:
            t_first = now
        if v != last_val:
            t_last, last_val = now, v
        if v >= n_events:
            break
        if t_last and now - t_last > 5:
            break
        time.sleep(0.01)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    elapsed = (t_last - t_first) if t_first and t_last and t_last > t_first else float("nan")
    return {"events": last_val, "seconds": elapsed, "eps": last_val / elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=400000)
    ap.add_argument("--partitions", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--model-npz", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "stream.md"))
    a = ap.parse_args()
    item_genres = np.load(a.model_npz)["item_genres"]
    events = producer.load_events("test", a.events)
    lines = [f"\n### Kafka -> Redis feature updates, {a.events:,} events per run\n",
             "| Partitions | Consumer workers | Batch size | Events/sec | Drain time (s) |",
             "|---:|---:|---:|---:|---:|"]
    for parts in a.partitions:
        topic = f"bench-p{parts}"
        subprocess.run([ROOT / "scripts" / "infra.sh", "topic", topic, str(parts)], check=True)
        prod_eps = producer.replay(events, 0, topic, producer.make_producer(), report_every=1e9)
        print(f"[bench] produced {len(events):,} events to {topic} at {prod_eps:,.0f} ev/s", flush=True)
        for w in a.workers:
            if w > parts:
                continue
            for b in a.batches:
                r = run_config(topic, w, b, len(events), item_genres)
                print(f"[bench] partitions={parts} workers={w} batch={b}: {r['eps']:,.0f} ev/s "
                      f"({r['events']:,} events in {r['seconds']:.1f}s)", flush=True)
                lines.append(f"| {parts} | {w} | {b} | {r['eps']:,.0f} | {r['seconds']:.1f} |")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
