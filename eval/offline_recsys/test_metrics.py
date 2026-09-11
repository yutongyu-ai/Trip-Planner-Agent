"""Unit tests for the NDCG@k / Precision@k formulas in run_eval.py.

These only exercise the metric math against hand-built rankings; they do
not touch the Yelp data files or the embedding model (see run_eval.py's
docstring for that end-to-end evaluation).
"""

import pytest

from run_eval import ndcg_at_k, precision_at_k


def test_precision_counts_hits_in_top_k():
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C"}
    assert precision_at_k(ranked, relevant, k=5) == pytest.approx(2 / 5)


def test_precision_ignores_hits_below_k():
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"E"}
    assert precision_at_k(ranked, relevant, k=2) == 0.0


def test_precision_with_no_relevant_items_is_zero():
    ranked = ["A", "B", "C"]
    assert precision_at_k(ranked, set(), k=3) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    ranked = ["A", "C", "B", "D", "E"]
    relevant = {"A", "C"}
    assert ndcg_at_k(ranked, relevant, k=5) == pytest.approx(1.0)


def test_ndcg_penalizes_relevant_items_ranked_lower():
    relevant = {"A", "C"}
    best = ndcg_at_k(["A", "C", "B", "D", "E"], relevant, k=5)
    worst = ndcg_at_k(["B", "D", "A", "E", "C"], relevant, k=5)
    assert worst < best

    same_hits_different_order = precision_at_k(["B", "D", "A", "E", "C"], relevant, k=5)
    assert same_hits_different_order == precision_at_k(["A", "C", "B", "D", "E"], relevant, k=5)


def test_ndcg_with_no_relevant_items_is_zero():
    ranked = ["A", "B", "C"]
    assert ndcg_at_k(ranked, set(), k=3) == 0.0


def test_ndcg_when_fewer_relevant_items_than_k():
    # Only one relevant item exists at all, so the ideal top-k already
    # contains it: a ranking that surfaces it first should still score 1.0.
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"A"}
    assert ndcg_at_k(ranked, relevant, k=5) == pytest.approx(1.0)
