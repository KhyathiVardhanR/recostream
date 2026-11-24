
### Offline evaluation on MovieLens-25M (model v20261124-154404)

Per-user temporal split; 20,000 random test users; each user's last 5 interactions are the test set; full ranking over 59,047 movies excluding seen ones.

| Model | NDCG@10 | Recall@10 | HitRate@10 | NDCG@10 vs all-time popularity | NDCG@10 vs recent popularity |
|---|---:|---:|---:|---:|---:|
| Popularity (all-time) | 0.0389 | 0.0459 | 0.1719 | +0% | -35% |
| Popularity (last 30 days before each user's cutoff) | 0.0599 | 0.0712 | 0.2492 | +54% | +0% |
| Matrix factorization (BPR) alone | 0.0388 | 0.0486 | 0.1848 | -0% | -35% |
| MF candidates + neural ranker | 0.0902 | 0.1049 | 0.3623 | +132% | +51% |

Training time (MF + ranker + evaluation): 278.0 s on an Apple M4 (PyTorch MPS).
