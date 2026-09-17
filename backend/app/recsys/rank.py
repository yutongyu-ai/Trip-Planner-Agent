"""Online content-based reranking of candidate attractions.

Not a trained model: this is embedding similarity between the user's stated
preferences and each candidate's text, blended with a normalized rating
prior so that semantically-close-but-poorly-rated places don't dominate.
The blend weight (alpha) is tuned offline against a held-out eval set —
see eval/offline_recsys/run_eval.py — rather than guessed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.recsys.embed import embed_texts
from app.tools.places_client import Attraction

DEFAULT_ALPHA = 0.7  # weight on semantic similarity
DEFAULT_RATING_PRIOR = 0.5  # normalized-rating fallback when a place has no rating
# Bayesian-average confidence threshold (the "m" in the IMDB weighted-rating
# formula): roughly how many ratings a place needs before its own average is
# trusted over the candidate pool's mean. Below this, a place's score gets
# pulled toward the pool average instead of taken at face value -- otherwise
# a 5-star place with 3 reviews outranks a 4.6-star place with thousands,
# which is exactly the "obscure but lucky" result this is meant to avoid.
MIN_RATING_COUNT = 30


@dataclass(frozen=True)
class RankedAttraction:
    attraction: Attraction
    score: float
    similarity: float
    normalized_rating: float


def _candidate_text(attraction: Attraction) -> str:
    parts = [attraction.name, " ".join(attraction.categories)]
    if attraction.description:
        parts.append(attraction.description)
    return ". ".join(p for p in parts if p)


def _raw_normalized_rating(attraction: Attraction) -> float | None:
    if attraction.rating is None:
        return None
    return max(0.0, min(attraction.rating, 5.0)) / 5.0


def _normalized_rating(attraction: Attraction, prior_mean: float) -> float:
    """A place's own rating, Bayesian-shrunk toward `prior_mean` (the
    candidate pool's average) by how few reviews it has -- see
    MIN_RATING_COUNT. A place with no rating at all still falls back to
    DEFAULT_RATING_PRIOR rather than the pool mean, since "no data" and
    "a handful of reviews trending toward average" aren't the same thing.
    """
    raw = _raw_normalized_rating(attraction)
    if raw is None:
        return DEFAULT_RATING_PRIOR
    count = attraction.rating_count or 0
    confidence = count / (count + MIN_RATING_COUNT)
    return confidence * raw + (1 - confidence) * prior_mean


def rank_attractions(
    candidates: list[Attraction],
    preference_text: str,
    alpha: float = DEFAULT_ALPHA,
    top_n: int | None = None,
) -> list[RankedAttraction]:
    """Rank `candidates` by blended semantic-similarity + rating score,
    descending. `alpha` in [0, 1] trades off similarity vs. rating.
    """
    if not candidates:
        return []

    candidate_texts = [_candidate_text(c) for c in candidates]
    embeddings = embed_texts(candidate_texts + [preference_text])
    candidate_embeddings, query_embedding = embeddings[:-1], embeddings[-1]

    # Embeddings are L2-normalized, so dot product == cosine similarity.
    similarities = candidate_embeddings @ query_embedding

    raw_ratings = [r for c in candidates if (r := _raw_normalized_rating(c)) is not None]
    prior_mean = sum(raw_ratings) / len(raw_ratings) if raw_ratings else DEFAULT_RATING_PRIOR

    ranked = []
    for attraction, similarity in zip(candidates, similarities):
        normalized_rating = _normalized_rating(attraction, prior_mean)
        score = alpha * float(similarity) + (1 - alpha) * normalized_rating
        ranked.append(
            RankedAttraction(
                attraction=attraction,
                score=score,
                similarity=float(similarity),
                normalized_rating=normalized_rating,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_n] if top_n is not None else ranked


# Review counts at/above this cap the popularity term at 1.0 in
# rank_attractions_by_popularity -- log10(100_000 + 1) ~= 5. Past this many
# reviews a place is already an unambiguous highlight; letting the term keep
# growing would let a handful of mega-attractions swamp the rating half of
# the score entirely.
_POPULARITY_REVIEW_CAP = 100_000


def _popularity(attraction: Attraction) -> float:
    """How many people have actually reviewed this place, log-scaled to
    [0, 1] and capped -- a real "is this a well-known highlight" signal,
    distinct from _normalized_rating's use of the same count (there, review
    count only gates how much to trust the *rating*; it never rewards a
    place for being popular per se). log-scaled because review counts span
    orders of magnitude: a landmark with 100,000 reviews and one with 10,000
    are both clearly well-known, but 100 reviews vs. 10 is a much smaller
    real-world gap than the raw numbers suggest.
    """
    count = attraction.rating_count or 0
    return min(math.log10(count + 1) / math.log10(_POPULARITY_REVIEW_CAP + 1), 1.0)


def rank_attractions_by_popularity(
    candidates: list[Attraction], top_n: int | None = None
) -> list[RankedAttraction]:
    """Rank by rating and popularity alone, no semantic similarity at all.

    For a traveller who stated no interests, rank_attractions's preference
    text degrades to a generic "general sightseeing" phrase that embeds
    almost identically against every candidate tagged tourist_attraction --
    the similarity term becomes noise rather than signal, and (confirmed
    against a real London search) that noise is enough to bump globally
    famous, heavily-reviewed landmarks like Big Ben below an obscure
    58-review historical plaque that happened to embed a little closer.

    Rating alone doesn't fix this either: London's top candidates are almost
    all rated within 4.5-4.8, a band far too narrow to reliably separate
    "must-see landmark" from "minor spot" -- Big Ben and a 58-review plaque
    can carry the *same* star rating. Review count is the signal that
    actually captures "this is a well-known highlight", so it's used here as
    its own scoring term (see _popularity) rather than only as a confidence
    gate on the rating (see _normalized_rating). This is a separate ranking
    path, not a low-alpha call into rank_attractions, precisely so the
    interest-driven case's offline-tuned DEFAULT_ALPHA (see
    eval/offline_recsys) is untouched by this.
    """
    if not candidates:
        return []

    raw_ratings = [r for c in candidates if (r := _raw_normalized_rating(c)) is not None]
    prior_mean = sum(raw_ratings) / len(raw_ratings) if raw_ratings else DEFAULT_RATING_PRIOR

    ranked = []
    for attraction in candidates:
        normalized_rating = _normalized_rating(attraction, prior_mean)
        score = 0.5 * normalized_rating + 0.5 * _popularity(attraction)
        ranked.append(
            RankedAttraction(
                attraction=attraction, score=score, similarity=0.0, normalized_rating=normalized_rating
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_n] if top_n is not None else ranked
