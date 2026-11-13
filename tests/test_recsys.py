import numpy as np
import pytest

from recsys import ALS, Interactions, ItemKNN, Popularity, evaluate, generate, leave_last_out


@pytest.fixture(scope="module")
def split():
    data, genres = generate(n_users=300, n_items=150, seed=3)
    train, held_out = leave_last_out(data)
    return data, train, held_out, genres


def tiny():
    #                 items: 0  1  2  3  4
    rows = {0: [0, 1], 1: [0, 1, 2], 2: [0, 1, 2], 3: [3, 4], 4: [3, 4], 5: [3]}
    users = [u for u, items in rows.items() for _ in items]
    items = [i for items in rows.values() for i in items]
    return Interactions(np.array(users), np.array(items), 7, 5)      # user 6 has no history


def test_generator_is_reproducible_and_has_structure():
    a, genres = generate(seed=1)
    b, _ = generate(seed=1)
    assert np.array_equal(a.users, b.users) and np.array_equal(a.items, b.items)
    m = a.matrix
    assert m.shape == (800, 400) and set(np.unique(m)) == {0.0, 1.0} and (m.sum(axis=1) >= 4).all()
    top_share = np.mean([np.bincount(genres[m[u] > 0], minlength=8).max() / m[u].sum() for u in range(200)])
    assert top_share > 0.4          # users concentrate on few genres; uniform choice would give about 0.125


def test_leave_last_out_holds_out_exactly_the_last_event_per_user(split):
    data, train, held_out, _ = split
    assert len(train) == len(data) - data.n_users and len(held_out) == data.n_users
    for u in (0, 7, 123):
        timeline = data.items[data.users == u]
        assert held_out[u] == timeline[-1]
        assert list(train.items[train.users == u]) == list(timeline[:-1])
        assert train.matrix[u, held_out[u]] == 0       # the target is never visible in training


def test_recommendations_exclude_seen_items_and_are_ordered(split):
    _, train, _, _ = split
    for model in (Popularity().fit(train), ItemKNN().fit(train), ALS(factors=8, iterations=3).fit(train)):
        seen = set(np.flatnonzero(train.matrix[5]).tolist())
        top = model.recommend(5, 10)
        assert len(top) == len(set(top)) == 10 and not seen & set(top)
        scores = np.asarray(model.scores(5), dtype=float)[top]
        assert list(scores) == sorted(scores, reverse=True)
        assert set(model.recommend(5, 10, exclude_seen=False)) != set(top) or not seen


def test_popularity_ranks_by_interaction_count():
    model = Popularity().fit(tiny())
    assert model.recommend(6, 5) == [0, 1, 3, 2, 4]            # counts 3, 3, 3, 2, 2: ties keep item order
    assert model.recommend(5, 2) == [0, 1]                     # user 5 has seen item 3, so it is skipped


def test_item_knn_links_items_that_share_users():
    model = ItemKNN(neighbours=5, shrink=0.0).fit(tiny())
    sim = model.similarity
    assert sim[0, 1] == pytest.approx(1.0)                     # items 0 and 1 have identical audiences
    assert sim[0, 2] == pytest.approx(2 / np.sqrt(3 * 2), rel=1e-5)
    assert sim[0, 3] == 0 and sim[2, 4] == 0                   # no shared users across the two groups
    assert np.allclose(sim, sim.T) and np.all(np.diag(sim) == 0)
    assert model.recommend(0, 1) == [2]                        # liked 0 and 1 -> the group's remaining item
    assert model.recommend(5, 1) == [4]
    assert model.similar_items(3, 1) == [4]


def test_shrinkage_discounts_similarity_built_on_few_users():
    plain = ItemKNN(neighbours=5, shrink=0.0).fit(tiny()).similarity
    shrunk = ItemKNN(neighbours=5, shrink=10.0).fit(tiny()).similarity
    assert np.all(shrunk <= plain + 1e-9) and shrunk[0, 1] < 0.3


def test_als_loss_decreases_and_it_recovers_the_two_groups():
    model = ALS(factors=2, regularization=0.01, alpha=10, iterations=12, seed=0).fit(tiny())
    assert all(b <= a + 1e-6 for a, b in zip(model.loss_history, model.loss_history[1:]))
    assert model.loss_history[-1] < 0.5 * model.loss_history[0]
    assert model.recommend(0, 1) == [2] and model.recommend(5, 1) == [4]
    s = model.scores(0)
    assert s[2] > s[3] and s[2] > s[4]                         # prefers the unseen item from its own group


def test_als_is_deterministic_for_a_seed(split):
    _, train, _, _ = split
    a, b = ALS(factors=8, iterations=3, seed=1).fit(train), ALS(factors=8, iterations=3, seed=1).fit(train)
    assert np.array_equal(a.user_factors, b.user_factors)
    assert not np.array_equal(a.user_factors, ALS(factors=8, iterations=3, seed=2).fit(train).user_factors)


def test_cold_start_users_get_popular_items(split):
    _, train, _, _ = split
    popular = Popularity().fit(train).recommend(10_000, 5)
    for model in (ItemKNN().fit(train), ALS(factors=8, iterations=2).fit(train)):
        assert model.recommend(10_000, 5) == popular
        assert model.recommend(-1, 5) == popular
    assert ALS(factors=2, iterations=2).fit(tiny()).recommend(6, 2) == [0, 1]   # in range but no history


def test_metrics_on_a_hand_checked_case():
    class Fixed:
        def recommend(self, user, k):
            return {0: [5, 6, 7], 1: [9, 5, 6], 2: [1, 2, 3]}[user][:k]

    m = evaluate(Fixed(), {0: 5, 1: 5, 2: 8}, k=3, n_items=10)
    assert m["hit_rate@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert m["ndcg@3"] == pytest.approx((1 + 1 / np.log2(3) + 0) / 3)
    assert m["coverage@3"] == pytest.approx(7 / 10)             # items 1, 2, 3, 5, 6, 7, 9


def test_personalised_models_beat_popularity_which_beats_chance():
    data, _ = generate(seed=0)                      # the demo's dataset: 800 users, 400 items
    train, held_out = leave_last_out(data)
    k = 10
    chance = k / (data.n_items - len(train) / data.n_users)
    pop = evaluate(Popularity().fit(train), held_out, k, data.n_items)
    knn = evaluate(ItemKNN(neighbours=20).fit(train), held_out, k, data.n_items)
    als = evaluate(ALS(factors=8, alpha=5, iterations=10).fit(train), held_out, k, data.n_items)
    assert pop[f"hit_rate@{k}"] > 1.5 * chance
    assert knn[f"hit_rate@{k}"] > 1.5 * pop[f"hit_rate@{k}"] and als[f"hit_rate@{k}"] > 1.5 * pop[f"hit_rate@{k}"]
    assert knn[f"coverage@{k}"] > 2 * pop[f"coverage@{k}"] and als[f"coverage@{k}"] > 2 * pop[f"coverage@{k}"]
    assert 0 <= als["mrr"] <= als[f"ndcg@{k}"] <= als[f"hit_rate@{k}"] <= 1
