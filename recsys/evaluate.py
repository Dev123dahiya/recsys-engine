"""Offline ranking metrics. Every item the user has not seen is ranked; nothing is sampled."""
from __future__ import annotations

import math

import numpy as np


def evaluate(model, held_out: dict[int, int], k: int = 10, n_items: int | None = None) -> dict[str, float]:
    """Score a fitted model on one held-out item per user.

    hit_rate@k   share of users whose held-out item is in their top k
    ndcg@k       like hit rate, but a hit at rank 1 counts more than at rank k
    mrr          mean of 1 / rank of the held-out item within the top k
    coverage@k   share of the catalogue that appears in anyone's top k
    """
    hits = ndcg = mrr = 0.0
    recommended: set[int] = set()
    for user, target in held_out.items():
        top = model.recommend(user, k)
        recommended.update(top)
        if target in top:
            rank = top.index(target) + 1
            hits += 1
            ndcg += 1.0 / math.log2(rank + 1)      # one relevant item, so the ideal DCG is 1
            mrr += 1.0 / rank
    n = len(held_out)
    out = {f"hit_rate@{k}": hits / n, f"ndcg@{k}": ndcg / n, "mrr": mrr / n}
    if n_items:
        out[f"coverage@{k}"] = len(recommended) / n_items
    return out


def genre_precision(model, users: list[int], item_genre: np.ndarray, train_matrix: np.ndarray, k: int = 10) -> float:
    """For synthetic data only: share of recommendations in a genre the user already favours."""
    total = good = 0
    for u in users:
        history = item_genre[train_matrix[u] > 0]
        if len(history) == 0:
            continue
        counts = np.bincount(history, minlength=item_genre.max() + 1)
        favourites = set(np.flatnonzero(counts >= 0.25 * counts.sum()).tolist())
        for item in model.recommend(u, k):
            total += 1
            good += int(item_genre[item]) in favourites
    return good / max(total, 1)
