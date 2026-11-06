import math

import pytest

from eval.metrics import Evidence, covers, locate, mrr, ndcg_at_k, recall_at_k, relevance_matrix, retrieval_scores


def test_locate_is_whitespace_and_case_insensitive():
    doc = "Intro.\n\nThe   Quick\nbrown fox jumps."
    ev = locate(Evidence("u", "the quick brown fox"), doc)
    assert ev.located and doc[ev.start:ev.end] == "The   Quick\nbrown fox"


def test_covers_needs_same_url_and_half_overlap(mk):
    ev = Evidence("u", "q", 100, 200)
    assert covers(mk("1", "u", 150, 400), ev)  # 50% covered
    assert not covers(mk("2", "u", 160, 400), ev)  # 40%
    assert not covers(mk("3", "other", 0, 1000), ev)


def test_ranking_metrics(mk):
    evidence = [Evidence("u", "q", 0, 10), Evidence("u", "q", 100, 110)]
    ranked = [mk("x", "u", 500, 600), mk("a", "u", 0, 50), mk("b", "u", 90, 120)]
    rel = relevance_matrix(ranked, evidence)
    assert rel == [set(), {0}, {1}]
    assert mrr(rel) == pytest.approx(0.5)
    assert recall_at_k(rel, 2, 2) == pytest.approx(0.5)
    assert recall_at_k(rel, 2, 3) == pytest.approx(1.0)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(rel, 2, 10) == pytest.approx(dcg / idcg)


def test_duplicate_chunks_do_not_inflate_ndcg(mk):
    evidence = [Evidence("u", "q", 0, 10)]
    rel = relevance_matrix([mk("a", "u", 0, 20), mk("b", "u", 0, 30)], evidence)
    assert ndcg_at_k(rel, 1, 10) == pytest.approx(1.0)


def test_retrieval_scores_context_precision(mk):
    evidence = [Evidence("u", "q", 0, 10)]
    ranked = [mk("a", "u", 0, 20), mk("b", "u", 300, 400)]
    s = retrieval_scores(ranked, ranked, evidence, ks=[1, 5])
    assert s["recall@1"] == 1.0 and s["context_precision"] == 0.5 and s["k_selected"] == 2
