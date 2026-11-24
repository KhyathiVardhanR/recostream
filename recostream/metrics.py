"""Ranking metrics for top-k recommendation with binary relevance."""
from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked: np.ndarray, relevant: list[set[int]], k: int = 10) -> float:
    """Mean NDCG@k. ranked: (n_users, >=k) item ids, best first."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    total = 0.0
    for row, rel in zip(ranked[:, :k], relevant):
        if not rel:
            continue
        gains = np.fromiter((1.0 if int(i) in rel else 0.0 for i in row), dtype=np.float64, count=k)
        ideal = discounts[: min(len(rel), k)].sum()
        total += float(gains @ discounts) / ideal
    n = sum(1 for r in relevant if r)
    return total / max(n, 1)


def recall_at_k(ranked: np.ndarray, relevant: list[set[int]], k: int = 10) -> float:
    total, n = 0.0, 0
    for row, rel in zip(ranked[:, :k], relevant):
        if not rel:
            continue
        total += len(rel.intersection(map(int, row))) / min(len(rel), k)
        n += 1
    return total / max(n, 1)


def hit_rate_at_k(ranked: np.ndarray, relevant: list[set[int]], k: int = 10) -> float:
    hits = [bool(rel.intersection(map(int, row))) for row, rel in zip(ranked[:, :k], relevant) if rel]
    return float(np.mean(hits)) if hits else 0.0
