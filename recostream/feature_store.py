"""Online feature store in Redis.

Per user (dense id):
    u:{id}:h   LIST   recent item ids, newest first (capped at history_len)
    u:{id}:s   HASH   n (interactions), rs (rating sum), t (last timestamp)
    u:{id}:g   HASH   genre index -> count
    u:{id}:e   STRING float32 MF user embedding (bootstrapped from the model)
    u:{id}:r   LIST   last recommendations served (for online evaluation), TTL
Global:
    pop:trend  ZSET   item -> streamed interaction count (trending)
    eval:*     counters for online recommendation quality

The Kafka consumer writes these incrementally; the serving API reads them in
one pipelined round trip per micro-batch.
"""
from __future__ import annotations

import numpy as np
import redis

from .config import settings
from .features import RECENT_N, UserFeatures

REC_TTL = 3600


def client(url: str | None = None) -> redis.Redis:
    return redis.Redis.from_url(url or settings.redis_url)


def bootstrap(r: redis.Redis, interactions, user_emb: np.ndarray, item_genres: np.ndarray,
              users: np.ndarray | None = None, chunk: int = 2000) -> int:
    """Load features for `users` (default: all) from historical interactions
    (DataFrame sorted by user, ts)."""
    H = settings.history_len
    df = interactions if users is None else interactions[interactions.user.isin(users)]
    g = df.groupby("user", sort=True)
    stats = g.agg(n=("rating", "size"), rs=("rating", "sum"), t=("ts", "max"))
    recent = g.tail(H)
    rec_lists = recent.groupby("user").item.apply(lambda s: s.values[::-1])
    gen = pd_genre_counts(df, item_genres)
    pipe = r.pipeline(transaction=False)
    count = 0
    for u, row in stats.iterrows():
        key = f"u:{u}"
        pipe.delete(f"{key}:h", f"{key}:g")
        items = rec_lists.get(u)
        if items is not None and len(items):
            pipe.rpush(f"{key}:h", *items.tolist())
        pipe.hset(f"{key}:s", mapping={"n": int(row.n), "rs": float(row.rs), "t": int(row.t)})
        gc = gen.get(u)
        if gc:
            pipe.hset(f"{key}:g", mapping=gc)
        pipe.set(f"{key}:e", user_emb[u].astype(np.float32).tobytes())
        count += 1
        if count % chunk == 0:
            pipe.execute()
    pipe.execute()
    return count


def pd_genre_counts(df, item_genres: np.ndarray) -> dict[int, dict[int, int]]:
    out: dict[int, dict[int, int]] = {}
    users = df.user.values
    gi = item_genres[df.item.values]
    # aggregate per user with numpy
    order = np.argsort(users, kind="stable")
    users, gi = users[order], gi[order]
    bounds = np.flatnonzero(np.diff(users)) + 1
    for seg_u, seg in zip(np.split(users, bounds), np.split(gi, bounds)):
        sums = seg.sum(0)
        nz = np.flatnonzero(sums)
        out[int(seg_u[0])] = {int(j): int(sums[j]) for j in nz}
    return out


def apply_events(r: redis.Redis, events: list[tuple[int, int, float, int]], item_genres: np.ndarray) -> dict:
    """Incrementally update features for a batch of (user, item, rating, ts)
    events, and score online quality: an event is a hit if the item was in the
    user's most recently served top-k. Two pipelined round trips per batch."""
    H = settings.history_len
    p = r.pipeline(transaction=False)
    users = [e[0] for e in events]
    for u in users:
        p.lrange(f"u:{u}:r", 0, -1)
    served = p.execute()
    hits = sum(1 for (u, i, _, _), recs in zip(events, served) if recs and str(i).encode() in recs)
    with_recs = sum(1 for recs in served if recs)

    p = r.pipeline(transaction=False)
    for u, i, rating, ts in events:
        key = f"u:{u}"
        p.lpush(f"{key}:h", i)
        p.ltrim(f"{key}:h", 0, H - 1)
        p.hincrby(f"{key}:s", "n", 1)
        p.hincrbyfloat(f"{key}:s", "rs", rating)
        p.hset(f"{key}:s", "t", ts)
        for gidx in np.flatnonzero(item_genres[i]):
            p.hincrby(f"{key}:g", int(gidx), 1)
        p.zincrby("pop:trend", 1, i)
    p.incrby("eval:events", len(events))
    p.incrby("eval:events_with_recs", with_recs)
    p.incrby("eval:hits", hits)
    p.execute()
    return {"events": len(events), "with_recs": with_recs, "hits": hits}


def fetch_users(r: redis.Redis, users: list[int], n_genres: int, emb_dim: int):
    """Read online features for a batch of users in one round trip.
    Returns (UserFeatures, user_emb (n,d), history lists, found mask)."""
    p = r.pipeline(transaction=False)
    for u in users:
        k = f"u:{u}"
        p.lrange(f"{k}:h", 0, settings.history_len - 1)
        p.hmget(f"{k}:s", "n", "rs")
        p.hgetall(f"{k}:g")
        p.get(f"{k}:e")
    res = p.execute()
    return parse_users(res, len(users), n_genres, emb_dim)


def parse_users(res: list, n: int, n_genres: int, emb_dim: int):
    act = np.zeros(n, dtype=np.float32)
    avg = np.full(n, 3.5, dtype=np.float32)
    prof = np.zeros((n, n_genres), dtype=np.float32)
    recent = np.full((n, RECENT_N), -1, dtype=np.int32)
    emb = np.zeros((n, emb_dim), dtype=np.float32)
    hist: list[list[int]] = []
    found = np.zeros(n, dtype=bool)
    for j in range(n):
        h, s, g, e = res[4 * j: 4 * j + 4]
        items = [int(x) for x in h]
        hist.append(items)
        if s[0] is not None:
            cnt = float(s[0])
            act[j] = np.log1p(cnt)
            avg[j] = float(s[1]) / max(cnt, 1.0)
            found[j] = True
        for gk, gv in g.items():
            prof[j, int(gk)] = float(gv)
        if items:
            recent[j, : min(RECENT_N, len(items))] = items[:RECENT_N]
        if e is not None:
            emb[j] = np.frombuffer(e, dtype=np.float32)
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8
    return UserFeatures(act, avg, prof, recent), emb, hist, found


def record_served(pipe, user: int, items: list[int]) -> None:
    key = f"u:{user}:r"
    pipe.delete(key)
    if items:
        pipe.rpush(key, *items)
    pipe.expire(key, REC_TTL)
