"""Kafka consumer group that updates online features in Redis.

    python -m recostream.stream.consumer --workers 4 --batch 1000

Each worker is a separate process in the same consumer group, so Kafka
spreads partitions across them. Events are processed in batches with two
pipelined Redis round trips per batch, then offsets are committed
(at-least-once delivery).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time

import numpy as np
import orjson
from confluent_kafka import Consumer, KafkaError
from prometheus_client import CollectorRegistry, Counter, Gauge, multiprocess, start_http_server

from .. import feature_store
from ..config import settings

EVENTS = Counter("reco_stream_events_total", "Interaction events processed", ["worker"])
BATCH_SECONDS = Gauge("reco_stream_last_batch_seconds", "Time to process the last batch", ["worker"],
                      multiprocess_mode="max")
ONLINE_HITS = Counter("reco_online_hits_total", "Events whose item was in the user's served top-k")
ONLINE_EVENTS = Counter("reco_online_events_with_recs_total", "Events for users that had recommendations served")


def load_item_genres(path: str | None) -> np.ndarray:
    """Item genres from a local model.npz, or from the production model in S3."""
    path = path or os.environ.get("RECO_MODEL_NPZ")
    if not path:
        from .. import registry
        prod = registry.production()
        if not prod:
            raise SystemExit("no production model; pass --model-npz or train first")
        path = str(registry.download(prod["version"], settings.artifacts_dir / "cache") / "model.npz")
    return np.load(path)["item_genres"]


def run_worker(idx: int, group: str, batch: int, item_genres: np.ndarray, stop_when_idle: float,
               counter=None) -> None:
    c = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "fetch.min.bytes": 1,
        "queued.max.messages.kbytes": 65536,
    })
    c.subscribe([settings.kafka_topic])
    r = feature_store.client()
    w = str(idx)
    idle_since = time.time()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    while running:
        msgs = c.consume(num_messages=batch, timeout=0.2)
        if not msgs:
            if stop_when_idle and time.time() - idle_since > stop_when_idle:
                break
            continue
        idle_since = time.time()
        t0 = time.perf_counter()
        events = []
        for m in msgs:
            if m.error():
                if m.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[consumer {w}] {m.error()}", flush=True)
                continue
            e = orjson.loads(m.value())
            events.append((e["u"], e["i"], e["r"], e["t"]))
        if events:
            res = feature_store.apply_events(r, events, item_genres)
            ONLINE_HITS.inc(res["hits"])
            ONLINE_EVENTS.inc(res["with_recs"])
            EVENTS.labels(w).inc(len(events))
            if counter is not None:
                with counter.get_lock():
                    counter.value += len(events)
        c.commit(asynchronous=True)
        BATCH_SECONDS.labels(w).set(time.perf_counter() - t0)
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--group", default="feature-updater")
    ap.add_argument("--model-npz", default=None, help="model.npz with item_genres (default $RECO_MODEL_NPZ)")
    ap.add_argument("--metrics-port", type=int, default=9300)
    a = ap.parse_args()
    item_genres = load_item_genres(a.model_npz)
    # Workers are separate processes: aggregate their metrics through
    # prometheus_client's multiprocess mode.
    mp_dir = os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", f"/tmp/reco-consumer-metrics-{os.getpid()}")
    os.makedirs(mp_dir, exist_ok=True)
    for f in os.listdir(mp_dir):
        os.remove(os.path.join(mp_dir, f))
    reg = CollectorRegistry()
    multiprocess.MultiProcessCollector(reg)
    start_http_server(a.metrics_port, registry=reg)
    procs = [mp.Process(target=run_worker, args=(i, a.group, a.batch, item_genres, 0), daemon=True)
             for i in range(a.workers)]
    for p in procs:
        p.start()
    print(f"[consumer] {a.workers} workers in group '{a.group}', batch={a.batch}, metrics :{a.metrics_port}")
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
