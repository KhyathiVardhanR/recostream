"""Offline training + evaluation pipeline.

    python -m recostream.train [--epochs 12] [--eval-users 20000] [--no-upload]

1. build/load the temporal split of MovieLens-25M
2. train BPR matrix factorization on `history`
3. generate top-200 MF candidates for ranker-training users and label them
   with each user's `val` interactions; train the neural ranker
4. evaluate on `test` (features from history+val) against a popularity
   baseline and MF alone: NDCG@10, Recall@10, HitRate@10
5. save artifacts, version them in S3, and promote through the evaluation gate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, registry
from .config import settings
from .features import DENSE_FEATURES, item_features, pair_features, user_features
from .metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from .models.mf import device, seen_matrix, top_candidates, train_bpr
from .models.ranker import rerank, train_ranker


def popularity_topk(history: pd.DataFrame, n_items: int, seen, users: np.ndarray, k: int) -> np.ndarray:
    pop = np.bincount(history.item.values, minlength=n_items)
    order = np.argsort(-pop)[: k + 2000]
    out = np.empty((len(users), k), dtype=np.int32)
    for r, u in enumerate(users):
        s = set(seen[u].indices)
        out[r] = [i for i in order if i not in s][:k]
    return out


def relevant_sets(df: pd.DataFrame, users: np.ndarray) -> list[set[int]]:
    g = df.groupby("user").item.apply(set)
    return [g.get(u, set()) for u in users]


def evaluate(name: str, ranked: np.ndarray, rel: list[set[int]], k: int) -> dict:
    m = {"ndcg@10": ndcg_at_k(ranked, rel, k), "recall@10": recall_at_k(ranked, rel, k),
         "hit@10": hit_rate_at_k(ranked, rel, k)}
    print(f"[eval] {name:<12} NDCG@10={m['ndcg@10']:.4f} Recall@10={m['recall@10']:.4f} HR@10={m['hit@10']:.4f}")
    return m


def run(epochs: int, ranker_users: int, eval_users: int, upload: bool, seed: int = 0) -> dict:
    t_all = time.time()
    k, n_cand = settings.k, settings.n_candidates
    ds = data.load()
    rng = np.random.default_rng(seed)

    # ---- 1. collaborative filtering
    mf = train_bpr(ds.history.user.values, ds.history.item.values, ds.n_users, ds.n_items,
                   dim=settings.emb_dim, epochs=epochs, seed=seed)
    U = mf.user.weight.detach().numpy().astype(np.float32)
    V = mf.item.weight.detach().numpy().astype(np.float32)
    B = mf.item_bias.weight.detach().numpy().reshape(-1).astype(np.float32)
    items = item_features(ds.history, ds.n_items, ds.item_genres, ds.item_year, V)

    # ---- 2. ranker training data: predict val from history
    seen_h = seen_matrix(ds.history.user.values, ds.history.item.values, ds.n_users, ds.n_items)
    val_users = np.unique(ds.val.user.values)
    tr_users = np.sort(rng.choice(val_users, size=min(ranker_users, len(val_users)), replace=False))
    cand, cscore = top_candidates(U, V, B, tr_users, seen_h, n_cand)
    uf = user_features(ds.history, tr_users, ds.item_genres)
    dense = pair_features(uf, cand, cscore, items)
    val_rel = relevant_sets(ds.val, tr_users)
    labels = np.array([[1 if int(i) in rel else 0 for i in row] for row, rel in zip(cand, val_rel)], dtype=np.int8)
    keep = labels.sum(1) > 0
    print(f"[ranker] {keep.sum():,}/{len(tr_users):,} users have a val positive in their top-{n_cand} candidates")
    ranker = train_ranker(U, V, tr_users[keep], cand[keep], dense[keep], labels[keep], seed=seed)

    # ---- 3. evaluation on test, features from history + val
    test_users = np.unique(ds.test.user.values)
    ev_users = np.sort(rng.choice(test_users, size=min(eval_users, len(test_users)), replace=False))
    hv = pd.concat([ds.history, ds.val]).sort_values(["user", "ts"], kind="stable")
    seen_hv = seen_matrix(hv.user.values, hv.item.values, ds.n_users, ds.n_items)
    rel = relevant_sets(ds.test, ev_users)
    pop_rank = popularity_topk(ds.history, ds.n_items, seen_hv, ev_users, k)
    cand_e, cscore_e = top_candidates(U, V, B, ev_users, seen_hv, n_cand)
    uf_e = user_features(hv, ev_users, ds.item_genres)
    dense_e = pair_features(uf_e, cand_e, cscore_e, items)
    rr = rerank(ranker, U, V, U[ev_users], cand_e, dense_e, k, dev=device())
    metrics = {
        "popularity": evaluate("popularity", pop_rank, rel, k),
        "mf": evaluate("mf", cand_e[:, :k], rel, k),
        "ranker": evaluate("mf+ranker", rr, rel, k),
        "eval_users": int(len(ev_users)),
        "n_users": ds.n_users, "n_items": ds.n_items,
        "train_interactions": int(len(ds.history)),
    }
    base = metrics["popularity"]["ndcg@10"]
    metrics["ndcg_lift_vs_popularity"] = {
        "mf": metrics["mf"]["ndcg@10"] / base - 1, "ranker": metrics["ranker"]["ndcg@10"] / base - 1}
    print(f"[eval] NDCG@10 lift over popularity: MF {metrics['ndcg_lift_vs_popularity']['mf']*100:+.1f}%, "
          f"MF+ranker {metrics['ndcg_lift_vs_popularity']['ranker']*100:+.1f}%")

    # ---- 4. artifacts + registry
    version = registry.new_version()
    out = settings.artifacts_dir / version
    out.mkdir(parents=True, exist_ok=True)
    pop_order = np.argsort(-np.bincount(ds.history.item.values, minlength=ds.n_items)).astype(np.int32)
    # Serving excludes everything a user has already seen (history + val);
    # newer interactions arrive through the stream into Redis.
    np.savez(out / "model.npz", user_emb=U, item_emb=V, item_bias=B, popularity_order=pop_order,
             seen_indptr=seen_hv.indptr.astype(np.int64), seen_indices=seen_hv.indices.astype(np.int32),
             user_ids=ds.user_ids, item_ids=ds.item_ids, titles=ds.titles, **{f"item_{k_}": v for k_, v in
                                                                              items.to_dict().items()})
    torch.save({"state_dict": ranker.state_dict(), "emb_dim": U.shape[1], "n_dense": len(DENSE_FEATURES)},
               out / "ranker.pt")
    metrics["version"] = version
    metrics["train_seconds"] = round(time.time() - t_all, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if upload:
        registry.upload(version, {"model.npz": out / "model.npz", "ranker.pt": out / "ranker.pt"}, metrics)
        registry.promote_if_better(version, metrics)
    print(f"[train] done in {metrics['train_seconds']}s -> {out}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--ranker-users", type=int, default=80000)
    ap.add_argument("--eval-users", type=int, default=20000)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.epochs, a.ranker_users, a.eval_users, not a.no_upload, a.seed)


if __name__ == "__main__":
    main()
