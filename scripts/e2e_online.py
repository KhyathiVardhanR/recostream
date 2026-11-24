#!/usr/bin/env python3
"""End-to-end online evaluation through the real serving + streaming path.

1. request top-10 recommendations from the API for N sampled test users
   (the API records what it served in Redis)
2. replay those users' actual next interactions (the test split) into Kafka
3. the running consumer group updates features and counts, in real time, how
   many events hit the user's served top-10
4. compare the online hit rate with the offline HitRate@10

Requires: infra up, API on :8000, consumer running (python -m recostream.stream.consumer).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402

from recostream import data, feature_store, registry  # noqa: E402
from recostream.config import settings  # noqa: E402
from recostream.stream import producer  # noqa: E402


async def fetch_recs(url: str, users: np.ndarray, conc: int = 64) -> None:
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(base_url=url, timeout=30) as c:
        async def one(u):
            async with sem:
                r = await c.get(f"/recommend/{int(u)}", params={"k": 10})
                r.raise_for_status()
        await asyncio.gather(*(one(u) for u in users))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rate", type=float, default=5000)
    ap.add_argument("--out", default=str(ROOT / "results" / "online.md"))
    a = ap.parse_args()
    ds = data.load()
    rng = np.random.default_rng(7)
    users = np.sort(rng.choice(np.unique(ds.test.user.values), a.users, replace=False))
    r = feature_store.client()
    for k in ("eval:events", "eval:events_with_recs", "eval:hits"):
        r.delete(k)
    # Reset these users' online features to history+val so the run is
    # repeatable (a previous run streamed their test events already).
    prod = registry.production()
    z = np.load(registry.download(prod["version"], settings.artifacts_dir / "cache") / "model.npz")
    hv = pd.concat([ds.history, ds.val]).sort_values(["user", "ts"], kind="stable")
    feature_store.bootstrap(r, hv, z["user_emb"], z["item_genres"], users=users)

    t0 = time.time()
    asyncio.run(fetch_recs(a.url, users))
    print(f"[e2e] served recommendations to {len(users):,} users in {time.time() - t0:.1f}s", flush=True)

    ev = ds.test[ds.test.user.isin(users)].sort_values("ts", kind="stable")
    # One event per user (their next interaction), so the hit rate matches
    # "did any of the user's next items appear" only through the first event.
    events = ev[["user", "item", "rating", "ts"]].to_records(index=False)
    producer.replay(events, a.rate, "interactions", producer.make_producer())
    deadline = time.time() + 120
    while time.time() < deadline:
        n = int(r.get("eval:events") or 0)
        if n >= len(events):
            break
        time.sleep(0.5)
    events_n, with_recs, hits = (int(x or 0) for x in r.mget("eval:events", "eval:events_with_recs", "eval:hits"))
    lag = time.time() - t0
    # Per-user hit rate (any of the user's next 5 items in the served top-10),
    # comparable to offline HitRate@10.
    hit_users = 0
    served = {}
    p = r.pipeline(transaction=False)
    for u in users:
        p.lrange(f"u:{int(u)}:r", 0, -1)
    for u, recs in zip(users, p.execute()):
        served[int(u)] = {int(x) for x in recs}
    for u, grp in ev.groupby("user"):
        if served.get(int(u), set()) & set(grp.item.values.tolist()):
            hit_users += 1
    online_user_hr = hit_users / len(users)
    print(f"[e2e] {events_n:,} streamed events, {hits:,} hit the served top-10 "
          f"({hits / max(with_recs, 1) * 100:.2f}% of events); per-user online HitRate@10 = {online_user_hr:.4f}")
    lines = ["\n### End-to-end online evaluation\n",
             f"Served top-10 to {len(users):,} random test users through the API, then streamed their real next "
             f"{events_n:,} interactions through Kafka; the consumer scored hits in real time.\n",
             "| Metric | Value |", "|---|---:|",
             f"| Events streamed / processed | {len(events):,} / {events_n:,} |",
             f"| Events whose item was in the served top-10 | {hits:,} ({hits / max(with_recs, 1) * 100:.2f}%) |",
             f"| Online HitRate@10 (user had ≥1 of next 5 items in served top-10) | {online_user_hr:.4f} |",
             f"| Wall time (serve + stream + process) | {lag:.1f} s |"]
    Path(a.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
