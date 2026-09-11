"""Offline evaluation of `rank_attractions` against a rating-only baseline.

Uses the Yelp Open Dataset (business.json + review.json, see data/) as a
proxy for the online use case: a user's own high-rated review text on one
attraction stands in for the trip preferences a traveller would type into
the app, and that same user's other high-rated attractions in the same
city are the relevance labels for a held-out ranking test.

Run: python eval/offline_recsys/run_eval.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# This is a small model (all-MiniLM-L6-v2); left at its default thread count
# on a many-core shared machine, torch oversubscribes threads and an
# embedding call that should take a fraction of a second instead takes
# minutes. Capping it fixes that without needing a GPU.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402

from app.recsys.rank import DEFAULT_ALPHA, rank_attractions  # noqa: E402
from app.tools.places_client import Attraction  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
BUSINESS_PATH = DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_PATH = DATA_DIR / "yelp_academic_dataset_review.json"

CITY = "Philadelphia"
MIN_HIGH_STARS = 4.0
K_VALUES = (5, 10)
MAX_EVAL_USERS = 150
RANDOM_SEED = 42
BASELINE_ALPHA = 0.0  # pure rating sort, no semantic similarity

# Specific (leaf) Yelp category tags that denote an actual tourist
# attraction. Deliberately narrower than the broad "Arts & Entertainment" /
# "Active Life" parent tags, which also cover gyms, nightlife venues, etc.
ATTRACTION_CATEGORIES = {
    "Museums", "Art Museums", "Children's Museums",
    "Landmarks & Historical Buildings", "Historical Tours",
    "Parks", "Dog Parks", "Botanical Gardens", "Beaches", "Hiking",
    "Zoos", "Aquariums", "Petting Zoos",
    "Art Galleries", "Public Art", "Performing Arts",
    "Tours", "Walking Tours", "Bus Tours", "Food Tours", "Boat Tours",
    "Bike Tours", "Aerial Tours", "Beer Tours",
    "Amusement Parks", "Trampoline Parks",
    "Stadiums & Arenas",
}


@dataclass(frozen=True)
class EvalSample:
    preference_text: str
    relevant_ids: set[str]
    held_out_id: str  # source of preference_text; excluded from the candidate pool


def load_attractions(city: str) -> dict[str, Attraction]:
    """Parse business.json for `city`'s attraction-category businesses."""
    attractions: dict[str, Attraction] = {}
    with BUSINESS_PATH.open() as f:
        for line in f:
            record = json.loads(line)
            if record["city"] != city:
                continue
            categories = [c.strip() for c in (record.get("categories") or "").split(",")]
            if not (set(categories) & ATTRACTION_CATEGORIES):
                continue
            attractions[record["business_id"]] = Attraction(
                place_id=record["business_id"],
                name=record["name"],
                address=record.get("address", ""),
                lat=record["latitude"],
                lng=record["longitude"],
                rating=record.get("stars"),
                rating_count=record.get("review_count"),
                price_level=None,
                categories=categories,
                description=None,
            )
    return attractions


def build_eval_samples(attraction_ids: set[str], rng: random.Random) -> list[EvalSample]:
    """Stream review.json once, keep high-rated reviews on `attraction_ids`,
    group by user, and turn each user with >=2 such reviews into one
    leave-one-out sample (one review held out as the preference query, the
    rest as relevance labels).
    """
    user_reviews: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with REVIEW_PATH.open() as f:
        for line in f:
            review = json.loads(line)
            if review["business_id"] not in attraction_ids:
                continue
            if review["stars"] < MIN_HIGH_STARS:
                continue
            text = (review.get("text") or "").strip()
            if text:
                user_reviews[review["user_id"]].append((review["business_id"], text))

    samples = []
    for reviews in user_reviews.values():
        if len(reviews) < 2:
            continue
        reviews = list(reviews)
        rng.shuffle(reviews)
        held_out_id, preference_text = reviews[0]
        relevant_ids = {business_id for business_id, _ in reviews[1:]}
        samples.append(EvalSample(preference_text, relevant_ids, held_out_id))
    return samples


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, business_id in enumerate(ranked_ids[:k])
        if business_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for business_id in ranked_ids[:k] if business_id in relevant_ids)
    return hits / k


def evaluate(
    alpha: float, samples: list[EvalSample], candidates: dict[str, Attraction]
) -> dict[str, float]:
    metrics: dict[str, list[float]] = {f"{name}@{k}": [] for name in ("ndcg", "precision") for k in K_VALUES}
    for sample in samples:
        pool = [a for place_id, a in candidates.items() if place_id != sample.held_out_id]
        ranked = rank_attractions(pool, sample.preference_text, alpha=alpha)
        ranked_ids = [r.attraction.place_id for r in ranked]
        for k in K_VALUES:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, sample.relevant_ids, k))
            metrics[f"precision@{k}"].append(precision_at_k(ranked_ids, sample.relevant_ids, k))
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def main() -> None:
    print(f"Loading {CITY} attraction businesses from {BUSINESS_PATH.name}...")
    attractions = load_attractions(CITY)
    print(f"  {len(attractions)} candidate attractions")

    print(f"Scanning {REVIEW_PATH.name} for high-rated reviews (reads the full file once)...")
    rng = random.Random(RANDOM_SEED)
    samples = build_eval_samples(set(attractions), rng)
    print(f"  {len(samples)} users with >= 2 high-rated reviews")

    if len(samples) > MAX_EVAL_USERS:
        samples = rng.sample(samples, MAX_EVAL_USERS)
    print(f"  evaluating on {len(samples)} sampled users (seed={RANDOM_SEED})")

    print("\nRanking candidates for each sample...")
    result_current = evaluate(DEFAULT_ALPHA, samples, attractions)
    result_baseline = evaluate(BASELINE_ALPHA, samples, attractions)

    current_label = f"rank_attractions (alpha={DEFAULT_ALPHA})"
    baseline_label = f"rating-only baseline (alpha={BASELINE_ALPHA})"
    print(f"\n{'metric':<14} {current_label:<32} {baseline_label:<32}")
    for name in result_current:
        print(f"{name:<14} {result_current[name]:<32.4f} {result_baseline[name]:<32.4f}")


if __name__ == "__main__":
    main()
