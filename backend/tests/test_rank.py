import numpy as np
import pytest

from app.recsys import rank as rank_module
from app.recsys.rank import rank_attractions
from app.tools.places_client import Attraction


def _attraction(name: str, rating: float | None, categories=None, rating_count: int = 10) -> Attraction:
    return Attraction(
        place_id=name,
        name=name,
        address="",
        lat=0.0,
        lng=0.0,
        rating=rating,
        rating_count=rating_count,
        price_level=None,
        categories=categories or ["museum"],
        description=None,
    )


@pytest.fixture
def fake_embed(monkeypatch):
    """Deterministic embeddings: 'Highly Relevant' matches the preference
    query exactly but has a low rating; 'Popular But Off Topic' is
    orthogonal to the query but has a high rating.
    """

    vectors = {
        "Highly Relevant. museum": np.array([1.0, 0.0]),
        "Popular But Off Topic. museum": np.array([0.0, 1.0]),
        "art and history": np.array([1.0, 0.0]),
    }

    def _fake_embed_texts(texts: list[str]) -> np.ndarray:
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr(rank_module, "embed_texts", _fake_embed_texts)
    return vectors


def test_rank_attractions_pure_similarity_favors_relevant_candidate(fake_embed):
    candidates = [
        _attraction("Highly Relevant", rating=2.0),
        _attraction("Popular But Off Topic", rating=5.0),
    ]

    ranked = rank_attractions(candidates, "art and history", alpha=1.0)

    assert ranked[0].attraction.name == "Highly Relevant"
    assert ranked[0].similarity == pytest.approx(1.0)


def test_rank_attractions_pure_rating_favors_popular_candidate(fake_embed):
    candidates = [
        _attraction("Highly Relevant", rating=2.0),
        _attraction("Popular But Off Topic", rating=5.0),
    ]

    ranked = rank_attractions(candidates, "art and history", alpha=0.0)

    assert ranked[0].attraction.name == "Popular But Off Topic"
    # Bayesian-shrunk toward the pool mean (both have rating_count=10, well
    # under MIN_RATING_COUNT), so it's pulled well below the raw 1.0 --
    # still clearly ahead of the other candidate, just not taken at face
    # value. See test_rank.py::test_low_review_count_is_shrunk_toward_pool_mean
    # for the mechanism itself.
    assert ranked[0].normalized_rating > ranked[1].normalized_rating
    assert ranked[0].normalized_rating < 1.0


def test_low_review_count_is_shrunk_toward_pool_mean():
    lucky = _attraction("Lucky Few Reviews", rating=5.0, rating_count=3)
    established = _attraction("Well Established", rating=4.2, rating_count=2000)

    lucky_score = rank_module._normalized_rating(lucky, prior_mean=0.7)
    established_score = rank_module._normalized_rating(established, prior_mean=0.7)

    # A near-perfect rating on 3 reviews shouldn't be trusted at face value --
    # it's pulled well below its own raw 1.0 toward the pool mean.
    assert lucky_score < 1.0
    # A well-reviewed place's own rating dominates almost entirely instead.
    assert established_score == pytest.approx(4.2 / 5, abs=0.02)


def test_zero_reviews_falls_back_fully_to_prior_mean():
    unreviewed = _attraction("New Place", rating=5.0, rating_count=0)

    assert rank_module._normalized_rating(unreviewed, prior_mean=0.6) == pytest.approx(0.6)


def test_no_rating_at_all_uses_default_prior_not_pool_mean():
    no_rating = _attraction("Mystery Place", rating=None)

    # A place with literally no rating data falls back to the fixed
    # DEFAULT_RATING_PRIOR, not the pool mean -- "unrated" and "rated low
    # with few reviews" are different situations.
    assert rank_module._normalized_rating(no_rating, prior_mean=0.9) == rank_module.DEFAULT_RATING_PRIOR


def test_rank_attractions_handles_missing_rating_with_prior(fake_embed):
    candidates = [_attraction("Highly Relevant", rating=None)]

    ranked = rank_attractions(candidates, "art and history", alpha=0.5)

    assert ranked[0].normalized_rating == rank_module.DEFAULT_RATING_PRIOR


def test_rank_attractions_empty_candidates_returns_empty():
    assert rank_attractions([], "anything") == []


def test_rank_attractions_respects_top_n(fake_embed):
    candidates = [
        _attraction("Highly Relevant", rating=2.0),
        _attraction("Popular But Off Topic", rating=5.0),
    ]

    ranked = rank_attractions(candidates, "art and history", top_n=1)

    assert len(ranked) == 1


def test_popularity_rank_favors_heavily_reviewed_landmark_over_similarly_rated_obscure_spot():
    # Same star rating -- only review count should separate them.
    landmark = _attraction("Big Ben", rating=4.6, rating_count=99_521)
    obscure = _attraction("Historical Plaque", rating=4.6, rating_count=58)

    ranked = rank_module.rank_attractions_by_popularity([landmark, obscure])

    assert ranked[0].attraction.name == "Big Ben"
    # The gap should be wide, not a narrow embedding-noise-sized margin --
    # this is the whole point of using review count as its own signal.
    assert ranked[0].score - ranked[1].score > 0.2


def test_popularity_rank_still_lets_a_much_higher_rating_win():
    slightly_less_reviewed_but_better_rated = _attraction("Tower Bridge", rating=4.8, rating_count=50_000)
    heavily_reviewed_but_mediocre = _attraction("Tourist Trap", rating=3.5, rating_count=50_000)

    ranked = rank_module.rank_attractions_by_popularity(
        [slightly_less_reviewed_but_better_rated, heavily_reviewed_but_mediocre]
    )

    assert ranked[0].attraction.name == "Tower Bridge"


def test_popularity_rank_caps_at_very_high_review_counts():
    assert rank_module._popularity(_attraction("A", rating=4.5, rating_count=100_000)) == pytest.approx(1.0)
    assert rank_module._popularity(_attraction("B", rating=4.5, rating_count=10_000_000)) == pytest.approx(1.0)


def test_popularity_rank_empty_candidates_returns_empty():
    assert rank_module.rank_attractions_by_popularity([]) == []


def test_popularity_rank_respects_top_n():
    candidates = [_attraction(f"Place {i}", rating=4.5, rating_count=i * 100) for i in range(5)]

    ranked = rank_module.rank_attractions_by_popularity(candidates, top_n=2)

    assert len(ranked) == 2
