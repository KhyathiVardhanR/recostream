import numpy as np
import pandas as pd
import pytest
import torch

from recostream.features import DENSE_FEATURES, RECENT_N, item_features, pair_features, user_features
from recostream.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from recostream.models.mf import seen_matrix, top_candidates, train_bpr
from recostream.models.ranker import Ranker


def test_ndcg_perfect_and_zero():
    ranked = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    assert ndcg_at_k(ranked, [{1, 2}]) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, [{99}]) == 0.0


def test_ndcg_position_discount():
    # single relevant item at rank 2: 1/log2(3)
    ranked = np.array([[5, 1, 6, 7, 8, 9, 10, 11, 12, 13]])
    assert ndcg_at_k(ranked, [{1}]) == pytest.approx(1 / np.log2(3))


def test_recall_and_hit_rate():
    ranked = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
    rel = [{1, 50}, {99}]
    assert recall_at_k(ranked, rel) == pytest.approx(0.25)
    assert hit_rate_at_k(ranked, rel) == pytest.approx(0.5)


def toy_interactions():
    # user 0 likes items 0-4 (genre 0), user 1 likes 5-9 (genre 1)
    rows = []
    t = 0
    for u in range(40):
        group = u % 2
        for i in range(5):
            rows.append((u, group * 5 + i, 4.0, t))
            t += 1
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def test_mf_learns_clusters_and_excludes_seen():
    df = toy_interactions()
    # hold out item 4 for user 0 and item 9 for user 1
    train = df[~(((df.user == 0) & (df.item == 4)) | ((df.user == 1) & (df.item == 9)))]
    mf = train_bpr(train.user.values, train.item.values, 40, 10, dim=8, epochs=60, batch=64, lr=0.05,
                   pop_neg_frac=0.0)
    U, V = mf.user.weight.detach().numpy(), mf.item.weight.detach().numpy()
    B = mf.item_bias.weight.detach().numpy().reshape(-1)
    seen = seen_matrix(train.user.values, train.item.values, 40, 10)
    cand, _ = top_candidates(U, V, B, np.array([0, 1]), seen, 3)
    assert cand[0, 0] == 4 and cand[1, 0] == 9
    assert not set(cand[0]) & {0, 1, 2, 3}


def test_pair_features_shapes_and_recent_order():
    df = toy_interactions()
    genres = np.zeros((10, 2), dtype=np.float32)
    genres[:5, 0] = 1
    genres[5:, 1] = 1
    emb = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    it = item_features(df, 10, genres, np.full(10, 2000, dtype=np.float32), emb)
    uf = user_features(df, np.array([0, 1]), genres)
    assert uf.recent_items[0, 0] == 4  # newest first
    assert (uf.recent_items[0, 5:] == -1).all() and RECENT_N == 10
    cand = np.array([[0, 5], [5, 0]], dtype=np.int32)
    F = pair_features(uf, cand, np.zeros((2, 2), dtype=np.float32), it)
    assert F.shape == (2, 2, len(DENSE_FEATURES))
    ga = DENSE_FEATURES.index("genre_affinity")
    assert F[0, 0, ga] > F[0, 1, ga]  # user 0 prefers genre 0
    assert F[1, 0, ga] > F[1, 1, ga]
    assert np.isfinite(F).all()


def test_ranker_forward():
    m = Ranker(8, len(DENSE_FEATURES))
    out = m(torch.zeros(3, 8), torch.zeros(3, 20, 8), torch.zeros(3, 20, len(DENSE_FEATURES)))
    assert out.shape == (3, 20)
