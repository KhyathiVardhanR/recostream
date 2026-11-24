"""Load the online feature store (Redis) from historical data and the
production model's user embeddings.

    python -m recostream.bootstrap [--model-dir artifacts/vXXXX]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, feature_store, registry
from .config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="local artifact dir (default: production model from S3)")
    a = ap.parse_args()
    if a.model_dir:
        path = Path(a.model_dir)
    else:
        prod = registry.production()
        path = registry.download(prod["version"], settings.artifacts_dir / "cache")
    z = np.load(path / "model.npz", allow_pickle=True)
    ds = data.load()
    hv = pd.concat([ds.history, ds.val]).sort_values(["user", "ts"], kind="stable")
    r = feature_store.client()
    t0 = time.time()
    n = feature_store.bootstrap(r, hv, z["user_emb"], z["item_genres"])
    for key in ("eval:events", "eval:events_with_recs", "eval:hits"):
        r.delete(key)
    r.delete("pop:trend")
    print(f"[bootstrap] loaded features for {n:,} users from model {path.name} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
