from __future__ import annotations

import numpy as np

from .data import Interactions


class Recommender:
    """Common interface: fit on interactions, score every item for a user, recommend the top k unseen."""

    def fit(self, data: Interactions) -> "Recommender":
        self._seen = data.matrix > 0
        self._popularity = self._seen.sum(axis=0).astype(np.float64)
        return self

    def scores(self, user: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        cold = not (0 <= user < self._seen.shape[0]) or not self._seen[user].any()
        # Cold start: with no history there is nothing to personalise on, so fall back to what is popular.
        s = self._popularity.copy() if cold else np.asarray(self.scores(user), dtype=np.float64).copy()
        if exclude_seen and not cold:
            s[self._seen[user]] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        return top[np.argsort(-s[top], kind="stable")].tolist()


class Popularity(Recommender):
    """Recommend the most-interacted-with items to everyone. The baseline every model has to beat."""

    def scores(self, user: int) -> np.ndarray:
        return self._popularity


class ItemKNN(Recommender):
    """Item-to-item collaborative filtering: "people who interacted with this also interacted with that".

    Item similarity is the cosine between item columns. ``shrink`` pulls
    similarities based on few co-occurrences towards zero, and only the top
    ``neighbours`` per item are kept, which removes noise from weak links.
    """

    def __init__(self, neighbours: int = 50, shrink: float = 10.0) -> None:
        self.neighbours, self.shrink = neighbours, shrink

    def fit(self, data: Interactions) -> "ItemKNN":
        super().fit(data)
        m = data.matrix
        co = m.T @ m                                     # co-occurrence counts between items
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < sim.shape[0]:
            cutoff = np.partition(sim, -self.neighbours, axis=1)[:, -self.neighbours][:, None]
            sim = np.where(sim >= cutoff, sim, 0.0)
        self.similarity = sim.astype(np.float32)
        self._matrix = m
        return self

    def scores(self, user: int) -> np.ndarray:
        return self._matrix[user] @ self.similarity

    def similar_items(self, item: int, k: int = 5) -> list[int]:
        return np.argsort(-self.similarity[item])[:k].tolist()


class ALS(Recommender):
    """Matrix factorisation for implicit feedback (Hu, Koren & Volinsky, 2008), by alternating least squares.

    Every user and item gets a vector of ``factors`` numbers; a user's score for
    an item is their dot product. Unobserved pairs are treated as weak negatives
    and observed ones as confident positives (weight 1 + alpha). With one side
    fixed the other has a closed-form solution, so training alternates between them.
    """

    def __init__(self, factors: int = 16, regularization: float = 0.1, alpha: float = 10.0, iterations: int = 15, seed: int = 0) -> None:
        self.factors, self.regularization, self.alpha, self.iterations, self.seed = factors, regularization, alpha, iterations, seed
        self.loss_history: list[float] = []

    def fit(self, data: Interactions) -> "ALS":
        super().fit(data)
        m = data.matrix
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (data.n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (data.n_items, self.factors))
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._solve(m, self.item_factors)
            self.item_factors = self._solve(m.T, self.user_factors)
            self.loss_history.append(self._loss(m))
        return self

    def _solve(self, m: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Solve for each row's vector with the other side held fixed.

        The trick that makes this affordable: Y^T C Y = Y^T Y + Y^T (C - I) Y, and
        C - I is non-zero only for the items a user touched, so the expensive
        part runs over a handful of rows instead of the whole catalogue.
        """
        gram = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        out = np.zeros((m.shape[0], self.factors))
        for row in range(m.shape[0]):
            idx = np.flatnonzero(m[row])
            if len(idx) == 0:
                continue
            y = fixed[idx]
            a = gram + self.alpha * (y.T @ y)
            b = (1.0 + self.alpha) * y.sum(axis=0)
            out[row] = np.linalg.solve(a, b)
        return out

    def _loss(self, m: np.ndarray) -> float:
        prediction = self.user_factors @ self.item_factors.T
        confidence = 1.0 + self.alpha * m
        reg = self.regularization * ((self.user_factors ** 2).sum() + (self.item_factors ** 2).sum())
        return float((confidence * (m - prediction) ** 2).sum() + reg)

    def scores(self, user: int) -> np.ndarray:
        return self.user_factors[user] @ self.item_factors.T
