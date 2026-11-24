"""Behavioral features for the ranker.

The same `pair_features` function is used offline (training/evaluation) and
online (serving, with user features read from Redis), so there is no
training/serving skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECENT_N = 10  # recent items used for history-similarity features
DENSE_FEATURES = [
    "mf_score", "mf_rank", "item_log_pop", "item_recent_pop", "item_avg_rating", "item_year",
    "user_log_activity", "user_avg_rating", "genre_affinity", "hist_sim_mean", "hist_sim_max", "hist_sim_last",
]


@dataclass
class ItemFeatures:
    log_pop: np.ndarray        # (n_items,)
    recent_pop: np.ndarray     # (n_items,) log count in the most recent 5% of history time
    avg_rating: np.ndarray     # (n_items,) Bayesian-shrunk mean rating
    year: np.ndarray           # (n_items,) normalized release year
    genres: np.ndarray         # (n_items, G)
    emb_normed: np.ndarray     # (n_items, d) L2-normalized MF item embeddings

    def to_dict(self) -> dict[str, np.ndarray]:
        return self.__dict__.copy()


@dataclass
class UserFeatures:
    log_activity: np.ndarray   # (n,)
    avg_rating: np.ndarray     # (n,)
    genre_profile: np.ndarray  # (n, G) L2-normalized
    recent_items: np.ndarray   # (n, RECENT_N) item ids, most recent first, -1 = padding


def item_features(history: pd.DataFrame, n_items: int, item_genres: np.ndarray, item_year: np.ndarray,
                  item_emb: np.ndarray) -> ItemFeatures:
    cnt = np.bincount(history.item.values, minlength=n_items).astype(np.float32)
    cutoff = np.quantile(history.ts.values, 0.95)
    recent = history[history.ts >= cutoff]
    rcnt = np.bincount(recent.item.values, minlength=n_items).astype(np.float32)
    rsum = np.bincount(history.item.values, weights=history.rating.values, minlength=n_items)
    global_mean = float(history.rating.mean())
    prior = 20.0
    avg = ((rsum + prior * global_mean) / (cnt + prior)).astype(np.float32)
    year = np.where(item_year > 0, (item_year - 1990.0) / 30.0, 0.0).astype(np.float32)
    norms = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-8
    return ItemFeatures(np.log1p(cnt), np.log1p(rcnt), avg, year, item_genres.astype(np.float32),
                        (item_emb / norms).astype(np.float32))


def user_features(inter: pd.DataFrame, users: np.ndarray, item_genres: np.ndarray) -> UserFeatures:
    """Profiles for `users` from their interactions (sorted by user, ts)."""
    sub = inter[inter.user.isin(users)]
    g = sub.groupby("user")
    cnt = g.size().reindex(users, fill_value=0).values.astype(np.float32)
    avg = g.rating.mean().reindex(users, fill_value=3.5).values.astype(np.float32)
    # genre profile: sum of item genre vectors
    pos = pd.Series(np.arange(len(users)), index=users)
    rows = pos.reindex(sub.user.values).values
    prof = np.zeros((len(users), item_genres.shape[1]), dtype=np.float32)
    np.add.at(prof, rows, item_genres[sub.item.values])
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-8
    # most recent items
    recent = np.full((len(users), RECENT_N), -1, dtype=np.int32)
    last = sub.groupby("user").tail(RECENT_N)
    lrows = pos.reindex(last.user.values).values
    # within each user, tail() keeps time order (oldest..newest); reverse to newest first
    rank = last.groupby("user").cumcount(ascending=False).values
    recent[lrows, rank] = last.item.values
    return UserFeatures(np.log1p(cnt), avg, prof, recent)


def pair_features(u: UserFeatures, cand: np.ndarray, cand_score: np.ndarray, it: ItemFeatures) -> np.ndarray:
    """Dense features for every (user, candidate): (n_users, n_cand, len(DENSE_FEATURES))."""
    n, c = cand.shape
    F = np.empty((n, c, len(DENSE_FEATURES)), dtype=np.float32)
    F[..., 0] = cand_score
    F[..., 1] = np.arange(c, dtype=np.float32)[None, :] / c
    F[..., 2] = it.log_pop[cand]
    F[..., 3] = it.recent_pop[cand]
    F[..., 4] = it.avg_rating[cand] - 3.5
    F[..., 5] = it.year[cand]
    F[..., 6] = u.log_activity[:, None]
    F[..., 7] = u.avg_rating[:, None] - 3.5
    F[..., 8] = np.einsum("ng,ncg->nc", u.genre_profile, it.genres[cand]) / (
        np.sqrt(it.genres[cand].sum(-1)) + 1e-8)
    # similarity of each candidate to the user's most recent items
    mask = u.recent_items >= 0  # (n, R)
    rec = it.emb_normed[np.where(mask, u.recent_items, 0)]  # (n, R, d)
    sims = np.einsum("ncd,nrd->ncr", it.emb_normed[cand], rec)  # (n, c, R)
    sims = np.where(mask[:, None, :], sims, np.nan)
    with np.errstate(invalid="ignore"):
        F[..., 9] = np.nan_to_num(np.nanmean(sims, axis=2))
        F[..., 10] = np.nan_to_num(np.nanmax(np.where(np.isnan(sims), -1, sims), axis=2))
    F[..., 11] = np.where(mask[:, :1], sims[:, :, 0], 0)
    return F
