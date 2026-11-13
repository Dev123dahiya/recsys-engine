from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interactions:
    """User-item events in time order. ``matrix`` is a dense 0/1 array (fine at this scale)."""

    users: np.ndarray
    items: np.ndarray
    n_users: int
    n_items: int

    @property
    def matrix(self) -> np.ndarray:
        m = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        m[self.users, self.items] = 1.0
        return m

    def __len__(self) -> int:
        return len(self.users)


def generate(n_users: int = 800, n_items: int = 400, n_genres: int = 8, mean_events: int = 25, seed: int = 0) -> tuple[Interactions, np.ndarray]:
    """Synthetic interactions with real structure to find.

    Each item has one genre; each user likes one or two genres. A user picks
    items mostly from their genres, with a long-tailed popularity bias and some
    random noise. Returns the interactions and each item's genre.
    """
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, n_genres, n_items)
    popularity = rng.zipf(1.6, n_items).astype(float)
    popularity = np.minimum(popularity, 50) / np.minimum(popularity, 50).sum()
    users, items = [], []
    for u in range(n_users):
        liked = rng.choice(n_genres, size=rng.integers(1, 3), replace=False)
        weights = popularity * np.where(np.isin(item_genre, liked), 12.0, 1.0)
        k = min(max(int(rng.poisson(mean_events)), 4), n_items // 2)
        chosen = rng.choice(n_items, size=k, replace=False, p=weights / weights.sum())
        users.extend([u] * k)
        items.extend(chosen.tolist())      # order within a user is their timeline
    return Interactions(np.array(users), np.array(items), n_users, n_items), item_genre


def leave_last_out(data: Interactions) -> tuple[Interactions, dict[int, int]]:
    """Hold out each user's most recent interaction as the test target.

    A random split would let the model train on events that happened after the
    one it is asked to predict. Holding out the last event mirrors real use:
    predict what comes next from what came before.
    """
    last_index: dict[int, int] = {}
    for i, u in enumerate(data.users):
        last_index[int(u)] = i
    keep = np.ones(len(data), dtype=bool)
    keep[list(last_index.values())] = False
    held_out = {u: int(data.items[i]) for u, i in last_index.items()}
    return Interactions(data.users[keep], data.items[keep], data.n_users, data.n_items), held_out
