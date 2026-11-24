"""Collaborative filtering: matrix factorization trained with BPR loss on
implicit feedback (every interaction is a positive). Used for candidate
generation: score all items for a user and keep the top N unseen ones.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(-1) + self.item_bias(i).squeeze(-1)


def train_bpr(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int, dim: int = 64, epochs: int = 12,
              batch: int = 65536, lr: float = 3e-3, reg: float = 1e-6, pop_neg_frac: float = 0.5,
              seed: int = 0) -> MF:
    """BPR with a mix of uniform and popularity-proportional negatives; the
    latter teaches the model not to simply rank by popularity."""
    torch.manual_seed(seed)
    dev = device()
    model = MF(n_users, n_items, dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u_all = torch.from_numpy(users.astype(np.int64)).to(dev)
    i_all = torch.from_numpy(items.astype(np.int64)).to(dev)
    pop = np.bincount(items, minlength=n_items).astype(np.float64) ** 0.75
    pop_cdf = torch.from_numpy(np.cumsum(pop / pop.sum()).astype(np.float32)).to(dev)
    n = len(users)
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            u, ip = u_all[idx], i_all[idx]
            m = len(idx)
            n_pop = int(m * pop_neg_frac)
            ineg = torch.empty(m, dtype=torch.int64, device=dev)
            ineg[:n_pop] = torch.searchsorted(pop_cdf, torch.rand(n_pop, device=dev)).clamp_(max=n_items - 1)
            ineg[n_pop:] = torch.randint(0, n_items, (m - n_pop,), device=dev)
            diff = model.score(u, ip) - model.score(u, ineg)
            loss = -torch.nn.functional.logsigmoid(diff).mean()
            loss = loss + reg * (model.user(u).pow(2).sum() + model.item(ip).pow(2).sum() +
                                 model.item(ineg).pow(2).sum()) / m
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * m
        print(f"[mf] epoch {ep + 1}/{epochs} loss={total / n:.4f} ({time.time() - t0:.1f}s)", flush=True)
    return model.cpu().eval()


def seen_matrix(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=(n_users, n_items))
    m.data[:] = 1.0
    return m


@torch.no_grad()
def top_candidates(user_emb: np.ndarray, item_emb: np.ndarray, item_bias: np.ndarray, users: np.ndarray,
                   seen: sp.csr_matrix, n: int, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-n unseen items per user by MF score. Returns (items, scores)."""
    dev = device()
    V = torch.from_numpy(item_emb).to(dev)
    b = torch.from_numpy(item_bias.reshape(-1)).to(dev)
    out_i = np.empty((len(users), n), dtype=np.int32)
    out_s = np.empty((len(users), n), dtype=np.float32)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        U = torch.from_numpy(user_emb[ub]).to(dev)
        scores = U @ V.T + b
        rows = seen[ub]
        r_idx = torch.from_numpy(np.repeat(np.arange(len(ub)), np.diff(rows.indptr))).to(dev)
        c_idx = torch.from_numpy(rows.indices.astype(np.int64)).to(dev)
        scores[r_idx, c_idx] = float("-inf")
        top = torch.topk(scores, n, dim=1)
        out_i[s:s + len(ub)] = top.indices.cpu().numpy()
        out_s[s:s + len(ub)] = top.values.cpu().numpy()
    return out_i, out_s
