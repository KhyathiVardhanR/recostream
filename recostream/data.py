"""MovieLens-25M loading and the per-user temporal split.

Every user's interactions are ordered by time and split into
    history | val (next 5) | test (last 5)
for users with at least 20 interactions (shorter histories stay entirely in
`history`). The collaborative-filtering model trains on `history`; the neural
ranker learns to predict `val` from features computed on `history`; final
evaluation predicts `test` from `history + val`, mirroring production where
features are fresher than the last model training.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings

N_HOLDOUT = 5
MIN_INTERACTIONS = 20


@dataclass
class Dataset:
    history: pd.DataFrame  # columns: user, item, rating, ts (int32/float32/int64)
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    item_genres: np.ndarray  # (n_items, n_genres) float32 multi-hot
    genre_names: list[str]
    item_year: np.ndarray  # (n_items,) float32, 0 if unknown
    user_ids: np.ndarray  # dense index -> original MovieLens userId
    item_ids: np.ndarray  # dense index -> original movieId
    titles: np.ndarray


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        raw_dir / "ratings.csv",
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
    )
    movies = pd.read_csv(raw_dir / "movies.csv")
    return ratings, movies


def build(raw_dir: Path | None = None, out_dir: Path | None = None) -> Dataset:
    raw_dir = raw_dir or settings.raw_dir
    out_dir = out_dir or settings.data_dir
    t0 = time.time()
    ratings, movies = load_raw(raw_dir)

    # Dense ids
    user_ids, user_idx = np.unique(ratings.userId.values, return_inverse=True)
    item_ids, item_idx = np.unique(ratings.movieId.values, return_inverse=True)
    df = pd.DataFrame({
        "user": user_idx.astype(np.int32),
        "item": item_idx.astype(np.int32),
        "rating": ratings.rating.values,
        "ts": ratings.timestamp.values,
    })
    df.sort_values(["user", "ts", "item"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)

    # Position from the end within each user
    counts = df.groupby("user").size().values
    pos_from_end = np.concatenate([np.arange(c - 1, -1, -1) for c in counts])
    eligible = np.repeat(counts >= MIN_INTERACTIONS, counts)
    is_test = eligible & (pos_from_end < N_HOLDOUT)
    is_val = eligible & (pos_from_end >= N_HOLDOUT) & (pos_from_end < 2 * N_HOLDOUT)
    history = df[~(is_test | is_val)].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)

    # Item side information
    movies = movies.set_index("movieId").reindex(item_ids)
    genre_lists = movies.genres.fillna("(no genres listed)").str.split("|")
    genre_names = sorted({g for gs in genre_lists for g in gs if g != "(no genres listed)"})
    gidx = {g: i for i, g in enumerate(genre_names)}
    item_genres = np.zeros((len(item_ids), len(genre_names)), dtype=np.float32)
    for i, gs in enumerate(genre_lists):
        for g in gs:
            if g in gidx:
                item_genres[i, gidx[g]] = 1.0
    years = movies.title.fillna("").str.extract(r"\((\d{4})\)\s*$")[0]
    item_year = pd.to_numeric(years, errors="coerce").fillna(0).values.astype(np.float32)

    ds = Dataset(history, val, test, len(user_ids), len(item_ids), item_genres, genre_names, item_year,
                 user_ids, item_ids, movies.title.fillna("").values)
    save(ds, out_dir)
    print(f"[data] {len(df):,} interactions, {ds.n_users:,} users, {ds.n_items:,} items | "
          f"history={len(history):,} val={len(val):,} test={len(test):,} ({time.time() - t0:.1f}s)")
    return ds


def save(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.history.to_parquet(out_dir / "history.parquet")
    ds.val.to_parquet(out_dir / "val.parquet")
    ds.test.to_parquet(out_dir / "test.parquet")
    np.savez(out_dir / "meta.npz", n_users=ds.n_users, n_items=ds.n_items, item_genres=ds.item_genres,
             genre_names=np.array(ds.genre_names), item_year=ds.item_year, user_ids=ds.user_ids,
             item_ids=ds.item_ids, titles=ds.titles)


def load(out_dir: Path | None = None) -> Dataset:
    out_dir = out_dir or settings.data_dir
    if not (out_dir / "meta.npz").exists():
        return build(out_dir=out_dir)
    m = np.load(out_dir / "meta.npz", allow_pickle=True)
    return Dataset(
        pd.read_parquet(out_dir / "history.parquet"), pd.read_parquet(out_dir / "val.parquet"),
        pd.read_parquet(out_dir / "test.parquet"), int(m["n_users"]), int(m["n_items"]), m["item_genres"],
        list(m["genre_names"]), m["item_year"], m["user_ids"], m["item_ids"], m["titles"],
    )
