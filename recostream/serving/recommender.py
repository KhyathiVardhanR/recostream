"""Online recommender: candidate generation + neural re-ranking for a batch
of users, with features read from Redis.

Per micro-batch of B users:
  1. one pipelined Redis round trip for all users' online features
  2. MF candidate scores for all items as one (B x d) @ (d x n_items) matmul,
     masking seen items (offline seen-set + online recent history)
  3. dense pair features, then one ranker forward pass over B x 200 pairs
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import feature_store
from ..config import settings
from ..features import DENSE_FEATURES, ItemFeatures, pair_features
from ..models.ranker import Ranker


@dataclass
class Model:
    version: str
    item_emb: np.ndarray
    item_bias: np.ndarray
    items: ItemFeatures
    ranker: Ranker
    popularity_order: np.ndarray
    seen_indptr: np.ndarray
    seen_indices: np.ndarray
    item_ids: np.ndarray
    titles: np.ndarray
    metrics: dict

    @classmethod
    def load(cls, path: Path) -> "Model":
        z = np.load(path / "model.npz", allow_pickle=True)
        items = ItemFeatures(z["item_log_pop"], z["item_recent_pop"], z["item_avg_rating"], z["item_year"],
                             z["item_genres"], z["item_emb_normed"])
        ck = torch.load(path / "ranker.pt", map_location="cpu", weights_only=True)
        ranker = Ranker(ck["emb_dim"], ck["n_dense"])
        ranker.load_state_dict(ck["state_dict"])
        ranker.eval()
        metrics = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else {}
        return cls(path.name, z["item_emb"], z["item_bias"], items, ranker, z["popularity_order"],
                   z["seen_indptr"], z["seen_indices"], z["item_ids"], z["titles"], metrics)


class Recommender:
    def __init__(self, model: Model, redis_client):
        self.m = model
        self.r = redis_client
        self.n_items = model.item_emb.shape[0]
        self.V = torch.from_numpy(model.item_emb)
        self.B = torch.from_numpy(model.item_bias.reshape(-1))
        self.n_genres = model.items.genres.shape[1]
        self.dim = model.item_emb.shape[1]

    def _seen(self, u: int) -> np.ndarray:
        ip = self.m.seen_indptr
        if u + 1 >= len(ip):
            return np.empty(0, dtype=np.int32)
        return self.m.seen_indices[ip[u]:ip[u + 1]]

    @torch.no_grad()
    def recommend_batch(self, users: list[int], k: int) -> tuple[list[list[int]], list[list[float]], dict]:
        timings = {}
        t0 = time.perf_counter()
        uf, emb, hist, found = feature_store.fetch_users(self.r, users, self.n_genres, self.dim)
        timings["features"] = time.perf_counter() - t0

        # Users without an MF embedding (new users) fold in from recent items.
        for j in range(len(users)):
            if not emb[j].any() and hist[j]:
                emb[j] = self.m.item_emb[hist[j][:20]].mean(0)

        t1 = time.perf_counter()
        n_cand = settings.n_candidates
        scores = torch.from_numpy(emb) @ self.V.T + self.B
        for j, u in enumerate(users):
            seen = self._seen(u)
            if len(seen):
                scores[j, torch.from_numpy(seen.astype(np.int64))] = float("-inf")
            if hist[j]:
                scores[j, torch.tensor(hist[j], dtype=torch.int64)] = float("-inf")
        top = torch.topk(scores, n_cand, dim=1)
        cand = top.indices.numpy().astype(np.int32)
        cscore = top.values.numpy().astype(np.float32)
        timings["candidates"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        dense = pair_features(uf, cand, cscore, self.m.items)
        logits = self.m.ranker(torch.from_numpy(emb), self.V[torch.from_numpy(cand.astype(np.int64))],
                               torch.from_numpy(dense))
        best = torch.topk(logits, k, dim=1)
        items = np.take_along_axis(cand, best.indices.numpy(), axis=1)
        timings["rank"] = time.perf_counter() - t2

        out_items, out_scores = [], []
        for j in range(len(users)):
            if not found[j] and not emb[j].any():
                # Cold start with no signal at all: popularity.
                out_items.append([int(x) for x in self.m.popularity_order[:k]])
                out_scores.append([0.0] * k)
            else:
                out_items.append([int(x) for x in items[j]])
                out_scores.append([float(x) for x in best.values[j]])
        return out_items, out_scores, timings
