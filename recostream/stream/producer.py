"""Replays MovieLens interactions into Kafka as a real-time event stream.

    python -m recostream.stream.producer --split test --rate 5000
    python -m recostream.stream.producer --split test --rate 0      # as fast as possible

Events are keyed by user id, so all events of a user land in one partition
and are processed in order.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import orjson
from confluent_kafka import Producer

from .. import data
from ..config import settings


def make_producer(bootstrap: str | None = None) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap or settings.kafka_bootstrap,
        "linger.ms": 5,
        "batch.size": 1 << 20,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 2_000_000,
    })


def encode(u: int, i: int, r: float, t: int) -> bytes:
    return orjson.dumps({"u": u, "i": i, "r": r, "t": t})


def replay(events: np.ndarray, rate: float, topic: str, producer: Producer, report_every: float = 5.0) -> float:
    """events: structured array with fields user, item, rating, ts. rate<=0 = unthrottled."""
    t0 = last = time.time()
    sent = 0
    for row in events:
        u, i = int(row["user"]), int(row["item"])
        while True:
            try:
                producer.produce(topic, key=str(u).encode(), value=encode(u, i, float(row["rating"]), int(row["ts"])))
                break
            except BufferError:
                producer.poll(0.05)
        sent += 1
        if sent % 1000 == 0:
            producer.poll(0)
            if rate > 0:
                ahead = sent / rate - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
            now = time.time()
            if now - last >= report_every:
                print(f"[producer] {sent:,} events, {sent / (now - t0):,.0f} ev/s", flush=True)
                last = now
    producer.flush()
    elapsed = time.time() - t0
    print(f"[producer] sent {sent:,} events in {elapsed:.1f}s ({sent / elapsed:,.0f} ev/s)", flush=True)
    return sent / elapsed


def load_events(split: str, limit: int | None = None) -> np.ndarray:
    ds = data.load()
    df = {"test": ds.test, "val": ds.val, "history": ds.history}[split].sort_values("ts", kind="stable")
    if limit:
        df = df.head(limit)
    return df[["user", "item", "rating", "ts"]].to_records(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "history"])
    ap.add_argument("--rate", type=float, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topic", default=settings.kafka_topic)
    a = ap.parse_args()
    replay(load_events(a.split, a.limit), a.rate, a.topic, make_producer())


if __name__ == "__main__":
    main()
