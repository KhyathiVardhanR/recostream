"""Neural ranking model: re-scores MF candidates using user/item embeddings
plus behavioral features (interaction history, genre affinity, popularity,
recency). Trained listwise: softmax cross-entropy over each user's candidate
list, with the user's next interactions as positives.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .mf import device


class Ranker(nn.Module):
    def __init__(self, emb_dim: int, n_dense: int, hidden: int = 256):
        super().__init__()
        self.dense_norm = nn.LayerNorm(n_dense)
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim + n_dense, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, u_emb: torch.Tensor, i_emb: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """u_emb (n,d), i_emb (n,c,d), dense (n,c,f) -> scores (n,c)."""
        u = u_emb[:, None, :].expand_as(i_emb)
        x = torch.cat([u, i_emb, u * i_emb, self.dense_norm(dense)], dim=-1)
        return self.net(x).squeeze(-1)


def train_ranker(user_emb: np.ndarray, item_emb: np.ndarray, users: np.ndarray, cand: np.ndarray,
                 dense: np.ndarray, labels: np.ndarray, epochs: int = 6, batch: int = 256, lr: float = 1e-3,
                 seed: int = 0) -> Ranker:
    """users (n,), cand (n,c), dense (n,c,f), labels (n,c) in {0,1}."""
    torch.manual_seed(seed)
    dev = device()
    model = Ranker(user_emb.shape[1], dense.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    U = torch.from_numpy(user_emb).to(dev)
    V = torch.from_numpy(item_emb).to(dev)
    n = len(users)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, batch):
            b = order[s:s + batch]
            ub = torch.from_numpy(users[b].astype(np.int64)).to(dev)
            cb = torch.from_numpy(cand[b].astype(np.int64)).to(dev)
            db = torch.from_numpy(dense[b]).to(dev)
            yb = torch.from_numpy(labels[b].astype(np.float32)).to(dev)
            logits = model(U[ub], V[cb], db)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(yb * logp).sum(1).div(yb.sum(1).clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"[ranker] epoch {ep + 1}/{epochs} loss={total / n:.4f} ({time.time() - t0:.1f}s)", flush=True)
    return model.cpu().eval()


@torch.no_grad()
def rerank(model: Ranker, user_emb: np.ndarray, item_emb: np.ndarray, users_vec: np.ndarray, cand: np.ndarray,
           dense: np.ndarray, k: int, batch: int = 1024, dev: torch.device | None = None) -> np.ndarray:
    """users_vec (n,d) user embeddings (not ids) -> top-k items (n,k)."""
    dev = dev or torch.device("cpu")
    model = model.to(dev)
    V = torch.from_numpy(item_emb).to(dev)
    out = np.empty((len(cand), k), dtype=np.int32)
    for s in range(0, len(cand), batch):
        u = torch.from_numpy(users_vec[s:s + batch]).to(dev)
        c = torch.from_numpy(cand[s:s + batch].astype(np.int64)).to(dev)
        d = torch.from_numpy(dense[s:s + batch]).to(dev)
        scores = model(u, V[c], d)
        top = torch.topk(scores, k, dim=1).indices
        out[s:s + batch] = torch.gather(c, 1, top).cpu().numpy()
    return out
