"""Unit tests for the ranking study. Run from ranking_study/:

    python -m unittest discover -s tests -v

They use small synthetic data, so no download is needed.
"""

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from granite_query.evaluation import (LENGTH_EDGES, build_matched_pool, build_pool,  # noqa: E402
                                      metrics_for, perturb)
from granite_query.ranking import (BM25Ranker, LengthRanker, LSARanker, RandomRanker,  # noqa: E402
                                   RocchioRanker, TfidfRanker, minhash_dedup, tokenize)

CORPUS = [
    "Tom has 3 apples and buys 5 more apples. How many apples does he have?",
    "Write a short poem about the ocean.",
    "A train travels 60 miles each hour for 3 hours. How many miles does it travel?",
    "Translate the sentence into French: I like apples.",
    "Sara saves 4 dollars each week. How many dollars does she save in 6 weeks?",
]


class MetricTests(unittest.TestCase):
    def test_perfect_ranking(self):
        labels = np.array([1, 1, 0, 0])
        p10, mrr, ndcg, rec = metrics_for(np.array([4.0, 3.0, 2.0, 1.0]), labels, k=2, recall_k=2)
        self.assertEqual((p10, mrr, rec), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(ndcg, 1.0)

    def test_hand_computed_values(self):
        # Ranking order: item 2 (non relevant), item 0 (relevant), item 1, item 3 (relevant).
        labels = np.array([1, 0, 0, 1])
        scores = np.array([3.0, 2.0, 4.0, 1.0])
        p, mrr, ndcg, rec = metrics_for(scores, labels, k=2, recall_k=3)
        self.assertAlmostEqual(p, 0.5)                 # 1 relevant in the top 2
        self.assertAlmostEqual(mrr, 0.5)               # first relevant at rank 2
        ideal = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg, (1 / math.log2(3)) / ideal)
        self.assertAlmostEqual(rec, 0.5)               # 1 of 2 relevant in the top 3

    def test_no_relevant_items(self):
        self.assertTrue(np.all(metrics_for(np.array([1.0, 2.0]), np.array([0, 0])) == 0))


class BM25Tests(unittest.TestCase):
    def test_matches_hand_formula(self):
        k1, b = 1.5, 0.75
        docs = [tokenize(t) for t in CORPUS]
        n, avgdl = len(docs), sum(map(len, docs)) / len(docs)
        vocab = {t for d in docs for t in d}
        idf = {t: math.log(n - sum(t in d for d in docs) + 0.5) - math.log(sum(t in d for d in docs) + 0.5)
               for t in vocab}
        mean_idf = sum(idf.values()) / len(idf)
        idf = {t: (v if v >= 0 else 0.25 * mean_idf) for t, v in idf.items()}
        query = "How many apples does Tom have"
        expected = []
        for d in docs:
            s = 0.0
            for t in tokenize(query):
                f = d.count(t)
                if f:
                    s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
            expected.append(s)
        got = BM25Ranker(CORPUS, k1=k1, b=b).score(query)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_matches_rank_bm25_when_installed(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.skipTest("rank_bm25 not installed")
        q = "How many dollars does she save each week"
        ref = BM25Okapi([tokenize(t) for t in CORPUS]).get_scores(tokenize(q))
        np.testing.assert_allclose(BM25Ranker(CORPUS).score(q), ref, rtol=1e-10)

    def test_math_problem_ranks_math_first(self):
        top = np.argmax(BM25Ranker(CORPUS).score("Ann has 2 apples and buys 4 more apples. How many apples?"))
        self.assertEqual(top, 0)

    def test_stop_word_variant_drops_stop_words(self):
        r = BM25Ranker(CORPUS, remove_stop=True)
        self.assertEqual(r.name, "BM25 no stop words")
        self.assertTrue(np.all(r.score("how many each much") == 0))  # all on the stop list


class OtherRankerTests(unittest.TestCase):
    def test_every_ranker_returns_one_finite_score_per_item(self):
        for r in [RandomRanker(CORPUS), LengthRanker(CORPUS), TfidfRanker(CORPUS),
                  RocchioRanker(CORPUS, k=2), LSARanker(CORPUS, dims=3)]:
            s = r.score("How many apples")
            self.assertEqual(s.shape, (len(CORPUS),), r.name)
            self.assertTrue(np.all(np.isfinite(s)), r.name)

    def test_length_ranker_ignores_query(self):
        r = LengthRanker(CORPUS)
        np.testing.assert_array_equal(r.score("a"), r.score("something else entirely"))
        self.assertEqual(np.argmax(r.score("x")), int(np.argmax([len(t.split()) for t in CORPUS])))

    def test_tfidf_exact_match_scores_highest(self):
        self.assertEqual(np.argmax(TfidfRanker(CORPUS).score(CORPUS[3])), 3)


class DedupTests(unittest.TestCase):
    def test_removes_exact_and_near_duplicates_keeps_distinct(self):
        base = ("Janet has 16 ducks that lay eggs every day. She eats three for breakfast every morning "
                "and bakes muffins for her friends every day with four. She sells the remaining eggs at "
                "the farmers market for 2 dollars per fresh duck egg. How much money in dollars does she "
                "make every day at the farmers market after breakfast and baking are done?")
        texts = [base, "Write a haiku about autumn leaves falling slowly.", base, perturb(base)]
        self.assertEqual(minhash_dedup(texts), [0, 1])

    def test_short_near_duplicates_can_slip_through(self):
        # Documents the known limit: in a short text one changed word removes a large
        # share of the 3 gram shingles, so Jaccard can fall below the 0.8 threshold.
        short = "Buy 3 pens and 7 pencils for the class today please"
        self.assertEqual(minhash_dedup([short, perturb(short)]), [0, 1])

    def test_perturb_changes_only_first_number(self):
        self.assertEqual(perturb("Buy 3 pens and 7 pencils"), "Buy 4 pens and 7 pencils")


class PoolTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(0)
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        self.alpaca = [" ".join(rng.choice(words) for _ in range(rng.choice([5, 10, 30, 40, 50, 60, 90])))
                       for _ in range(3000)]
        self.gsm = [f"Problem {i}: " + " ".join(["word"] * rng.randint(20, 80)) for i in range(400)]

    def test_build_pool_sizes_and_labels(self):
        texts, labels = build_pool(self.alpaca, self.gsm, 100, 20, 5, random.Random(1))
        self.assertEqual(len(texts), 125)
        self.assertEqual(int(labels.sum()), 25)

    def test_matched_pool_follows_gsm_length_distribution(self):
        texts, labels = build_matched_pool(self.alpaca, self.gsm, 200, 20, 4, random.Random(1))
        alp_len = [len(t.split()) for t, l in zip(texts, labels) if l == 0]
        got = np.histogram(alp_len, LENGTH_EDGES)[0] / len(alp_len)
        want = np.histogram([len(t.split()) for t in self.gsm], LENGTH_EDGES)[0] / len(self.gsm)
        np.testing.assert_allclose(got, want, atol=0.02)

    def test_matched_pool_refuses_impossible_request(self):
        with self.assertRaises(ValueError):
            build_matched_pool(self.alpaca[:50], self.gsm, 2000, 20, 4, random.Random(1))


if __name__ == "__main__":
    unittest.main()
