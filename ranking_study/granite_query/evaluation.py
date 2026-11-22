"""Data loading, pool construction and retrieval metrics."""

import json
import re
from pathlib import Path

import numpy as np


def load_data(data_dir):
    data_dir = Path(data_dir)
    alpaca = json.loads((data_dir / "alpaca.json").read_text(encoding="utf8"))
    alpaca_texts = [(a["instruction"] + " " + a.get("input", "")).strip() for a in alpaca]

    def read(name):
        lines = (data_dir / name).read_text(encoding="utf8").splitlines()
        return [json.loads(l)["question"] for l in lines if l.strip()]

    return alpaca_texts, read("gsm8k_train.jsonl"), read("gsm8k_test.jsonl")


def perturb(text):
    """Make a near duplicate by adding 1 to the first number in the problem."""
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 1), text, count=1)


def build_pool(alpaca, gsm_train, n_alpaca, n_gsm, n_dupes, rng):
    """Mix Alpaca (label 0) with GSM8K problems and near duplicates (label 1)."""
    items = [(t, 0) for t in rng.sample(alpaca, n_alpaca)]
    gsm = rng.sample(gsm_train, n_gsm)
    items += [(t, 1) for t in gsm]
    items += [(perturb(t), 1) for t in rng.sample(gsm, n_dupes)]
    rng.shuffle(items)
    return [t for t, _ in items], np.array([l for _, l in items])


LENGTH_EDGES = [0, 25, 35, 45, 55, 70, 10**6]


def build_matched_pool(alpaca, gsm_train, n_alpaca, n_gsm, n_dupes, rng):
    """Like build_pool, but Alpaca items are sampled so their word count
    distribution matches GSM8K. This removes document length as a shortcut."""
    gsm = rng.sample(gsm_train, n_gsm)
    g_len = np.array([len(t.split()) for t in gsm_train])
    share = np.histogram(g_len, LENGTH_EDGES)[0] / len(g_len)
    buckets = [[] for _ in share]
    for t in alpaca:
        buckets[np.searchsorted(LENGTH_EDGES, len(t.split()), side="right") - 1].append(t)
    picked = []
    for frac, bucket in zip(share, buckets):
        want = int(round(frac * n_alpaca))
        if want > len(bucket):
            raise ValueError(f"only {len(bucket)} Alpaca items in a length bucket, need {want}; lower n_alpaca")
        picked += rng.sample(bucket, want)
    items = [(t, 0) for t in picked] + [(t, 1) for t in gsm]
    items += [(perturb(t), 1) for t in rng.sample(gsm, n_dupes)]
    rng.shuffle(items)
    return [t for t, _ in items], np.array([l for _, l in items])


METRICS =["P@10", "MRR", "nDCG@10", "Recall@100"]


def metrics_for(scores, labels, k=10, recall_k=100):
    order = np.argsort(-scores, kind="stable")
    rel = labels[order]
    total = labels.sum()
    hits = np.flatnonzero(rel)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = discounts[: min(k, total)].sum()
    return np.array([
        rel[:k].mean(),
        1.0 / (hits[0] + 1) if len(hits) else 0.0,
        (rel[:k] * discounts).sum() / idcg if idcg else 0.0,
        rel[:recall_k].sum() / total if total else 0.0,
    ])


def evaluate_ranker(ranker, queries, labels):
    return np.mean([metrics_for(ranker.score(q), labels) for q in queries], axis=0)
