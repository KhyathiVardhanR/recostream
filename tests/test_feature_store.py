"""Feature-store tests against a live Redis (skipped if none is running)."""
import numpy as np
import pandas as pd
import pytest
import redis

from recostream import feature_store


@pytest.fixture
def r():
    c = redis.Redis.from_url("redis://127.0.0.1:6379/15")
    try:
        c.ping()
    except redis.ConnectionError:
        pytest.skip("redis not running")
    c.flushdb()
    yield c
    c.flushdb()


def test_bootstrap_apply_and_fetch(r):
    genres = np.eye(3, dtype=np.float32)
    df = pd.DataFrame({"user": [0, 0, 1], "item": [0, 1, 2], "rating": [4.0, 5.0, 3.0], "ts": [1, 2, 3]})
    emb = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert feature_store.bootstrap(r, df, emb, genres) == 2

    uf, e, hist, found = feature_store.fetch_users(r, [0, 1, 7], 3, 4)
    assert found.tolist() == [True, True, False]
    assert hist[0] == [1, 0]  # newest first
    assert np.allclose(e[1], emb[1])
    assert uf.avg_rating[0] == pytest.approx(4.5)

    # serve recs, then stream events: one hit (item 2 was recommended to user 0)
    pipe = r.pipeline()
    feature_store.record_served(pipe, 0, [2, 1])
    pipe.execute()
    res = feature_store.apply_events(r, [(0, 2, 4.0, 10), (1, 0, 2.0, 11)], genres)
    assert res == {"events": 2, "with_recs": 1, "hits": 1}

    uf, _, hist, _ = feature_store.fetch_users(r, [0], 3, 4)
    assert hist[0][:3] == [2, 1, 0]
    assert np.exp(uf.log_activity[0]) - 1 == pytest.approx(3)
    assert r.zscore("pop:trend", 2) == 1.0
