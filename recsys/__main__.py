"""Demo: python -m recsys   (synthetic data; compares three recommenders on held-out interactions)"""
from __future__ import annotations

import time

import numpy as np

from . import ALS, ItemKNN, Popularity, evaluate, generate, leave_last_out
from .evaluate import genre_precision


def tune(candidates: dict, train, k: int = 10):
    """Pick settings on a validation split carved out of the training data, never on the test items."""
    inner_train, validation = leave_last_out(train)
    scores = {name: evaluate(make().fit(inner_train), validation, k)[f"ndcg@{k}"] for name, make in candidates.items()}
    best = max(scores, key=scores.get)
    return best, candidates[best], scores


def main() -> None:
    data, item_genre = generate(seed=0)
    train, held_out = leave_last_out(data)
    matrix = train.matrix
    density = len(data) / (data.n_users * data.n_items)
    print(f"{data.n_users} users, {data.n_items} items, {len(data):,} interactions ({density:.1%} of the matrix filled). Synthetic data.")
    print(f"Each user's last interaction is held out; the model ranks all {data.n_items} items they have not seen.")

    knn_name, knn_make, _ = tune({f"Item kNN, {n} neighbours": (lambda n=n: ItemKNN(neighbours=n)) for n in (20, 50, 200)}, train)
    als_grid = {f"ALS, {f} factors, alpha {a}": (lambda f=f, a=a: ALS(factors=f, alpha=a, iterations=10)) for f in (8, 16, 32) for a in (5, 20)}
    als_name, als_make, als_scores = tune(als_grid, train)
    print("Settings were chosen on a validation split of the training data. Validation NDCG@10 for ALS:")
    print("  " + ", ".join(f"{n.split(', ', 1)[1]}: {s:.3f}" for n, s in als_scores.items()) + "\n")

    print("| Model | Hit rate@10 | NDCG@10 | MRR | Catalogue coverage@10 | Recommendations in a favoured genre | Fit time |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    unseen = (matrix == 0).sum(axis=1).mean()

    class Random(Popularity):
        def scores(self, user: int) -> np.ndarray:
            return np.random.default_rng(user).random(self._seen.shape[1])

    for name, make in [("Random", Random), ("Popularity", Popularity), (knn_name, knn_make), (als_name, als_make)]:
        start = time.perf_counter()
        model = make().fit(train)
        elapsed = time.perf_counter() - start
        m = evaluate(model, held_out, k=10, n_items=data.n_items)
        gp = genre_precision(model, list(range(data.n_users)), item_genre, matrix)
        print(f"| {name} | {m['hit_rate@10']:.3f} | {m['ndcg@10']:.3f} | {m['mrr']:.3f} | {m['coverage@10']:.1%} | {gp:.2f} | {elapsed:.2f} s |")
    print(f"\n(Chance level for hit rate@10 is 10 / {unseen:.0f} unseen items = {10 / unseen:.3f}.)")

    als = als_make().fit(train)
    print(f"ALS loss per iteration: {' '.join(f'{x:,.0f}' for x in als.loss_history)}")
    knn = knn_make().fit(train)
    item = int(np.argmax(matrix.sum(axis=0)))
    neighbours = knn.similar_items(item, 5)
    print(f"Items most similar to item {item} (genre {item_genre[item]}): {neighbours} with genres {item_genre[neighbours].tolist()}")
    print(f"A brand-new user with no history gets the popular items: {als.recommend(data.n_users + 5, 5)}")


if __name__ == "__main__":
    main()
