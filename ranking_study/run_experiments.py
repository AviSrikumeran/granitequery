"""Run every Granite Query experiment and write results/results.json.

Experiments
1. Main comparison: seven rankers on three pools (standard, MinHash
   deduplicated, length matched), 3 seeds each.
2. BM25 parameter sweep over k1 and b, on the standard and length matched pools.
3. Pool size scaling: how precision changes as more Alpaca noise is added.
4. Error analysis: the highest scoring non relevant BM25 results.
5. Demo pool export for the website's in browser search.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from granite_query.evaluation import METRICS, build_matched_pool, build_pool, evaluate_ranker, load_data
from granite_query.ranking import (BM25Ranker, LengthRanker, LSARanker, RandomRanker, RocchioRanker,
                                   TfidfRanker, minhash_dedup)


def make_rankers(texts, seed):
    return [
        RandomRanker(texts, seed),
        LengthRanker(texts),
        TfidfRanker(texts),
        RocchioRanker(texts),
        LSARanker(texts, seed=seed),
        BM25Ranker(texts),
        BM25Ranker(texts, remove_stop=True),
    ]


def summarize(by_method):
    out = {}
    for name, runs in by_method.items():
        arr = np.array(runs)
        out[name] = {m: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
                     for i, m in enumerate(METRICS)}
    return out


def main_comparison(alpaca, gsm_train, gsm_test, args):
    runs = {"full": {}, "dedup": {}, "length_matched": {}}
    dedup_stats = []
    for seed in args.seeds:
        rng = random.Random(seed)
        texts, labels = build_pool(alpaca, gsm_train, args.n_alpaca, args.n_gsm, args.n_dupes, rng)
        queries = rng.sample(gsm_test, args.n_queries)
        keep = minhash_dedup(texts)
        dedup_stats.append({
            "seed": seed, "pool_before": len(texts), "pool_after": len(keep),
            "removed_relevant": int(labels.sum() - labels[keep].sum()),
            "removed_nonrelevant": int((labels == 0).sum() - (labels[keep] == 0).sum()),
        })
        m_texts, m_labels = build_matched_pool(alpaca, gsm_train, args.n_matched_alpaca,
                                               args.n_matched_alpaca // 10, args.n_matched_alpaca // 50,
                                               random.Random(seed))
        pools = {"full": (texts, labels), "dedup": ([texts[i] for i in keep], labels[keep]),
                 "length_matched": (m_texts, m_labels)}
        for setting, (t, l) in pools.items():
            for r in make_rankers(t, seed):
                runs[setting].setdefault(r.name, []).append(evaluate_ranker(r, queries, l))
        print(f"  seed {seed}: dedup kept {len(keep)} of {len(texts)}")
    return {k: summarize(v) for k, v in runs.items()}, dedup_stats


def bm25_sweep(alpaca, gsm_train, gsm_test, args):
    out = {}
    for setting in ["full", "length_matched"]:
        rng = random.Random(args.seeds[0])
        if setting == "full":
            texts, labels = build_pool(alpaca, gsm_train, args.n_alpaca, args.n_gsm, args.n_dupes, rng)
        else:
            texts, labels = build_matched_pool(alpaca, gsm_train, args.n_matched_alpaca,
                                               args.n_matched_alpaca // 10, args.n_matched_alpaca // 50, rng)
        queries = random.Random(args.seeds[0]).sample(gsm_test, args.n_queries)
        grid = []
        for k1 in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
                m = evaluate_ranker(BM25Ranker(texts, k1=k1, b=b), queries, labels)
                grid.append({"k1": k1, "b": b, **dict(zip(METRICS, map(float, m)))})
        out[setting] = grid
    return out


def pool_scaling(alpaca, gsm_train, gsm_test, args):
    rows = []
    for n_alpaca in [1000, 2500, 5000, 10000, 20000, 40000]:
        rng = random.Random(args.seeds[0])
        texts, labels = build_pool(alpaca, gsm_train, n_alpaca, args.n_gsm, args.n_dupes, rng)
        queries = rng.sample(gsm_test, args.n_queries)
        row = {"n_alpaca": n_alpaca, "pool_size": len(texts),
               "relevant_share": float(labels.mean())}
        for r in [RandomRanker(texts), TfidfRanker(texts), BM25Ranker(texts)]:
            row[r.name] = float(evaluate_ranker(r, queries, labels)[0])
        rows.append(row)
        print(f"  pool {len(texts)}: BM25 P@10 {row['BM25']:.3f}")
    return rows


def error_analysis(alpaca, gsm_train, gsm_test, args, n=8):
    rng = random.Random(args.seeds[0])
    texts, labels = build_pool(alpaca, gsm_train, args.n_alpaca, args.n_gsm, args.n_dupes, rng)
    queries = rng.sample(gsm_test, args.n_queries)
    bm = BM25Ranker(texts)
    fps, per_query = [], []
    for q in queries:
        s = bm.score(q)
        top = np.argsort(-s)[:10]
        per_query.append(float(labels[top].mean()))
        fps += [(float(s[i]), q, texts[i]) for i in top if labels[i] == 0]
    fps.sort(reverse=True)
    hist = np.histogram(per_query, bins=np.linspace(0, 1, 11))[0].tolist()
    return {"false_positives": [{"score": s, "query": q, "hit": t} for s, q, t in fps[:n]],
            "p10_histogram": hist, "share_of_queries_with_p10_1": float(np.mean(np.array(per_query) == 1.0)),
            "share_of_queries_with_p10_below_half": float(np.mean(np.array(per_query) < 0.5))}


def export_demo(alpaca, gsm_train, gsm_test, out_path, seed=7):
    rng = random.Random(seed)
    texts, labels = build_pool(alpaca, gsm_train, 3000, 300, 60, rng)
    examples = rng.sample(gsm_test, 12)
    Path(out_path).write_text(json.dumps({"docs": texts, "labels": labels.tolist(),
                                          "example_queries": examples}), encoding="utf8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n_alpaca", type=int, default=5000)
    ap.add_argument("--n_gsm", type=int, default=500)
    ap.add_argument("--n_dupes", type=int, default=100)
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--n_matched_alpaca", type=int, default=1800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--demo_out", default="results/demo_pool.json")
    args = ap.parse_args()

    t0 = time.time()
    alpaca, gsm_train, gsm_test = load_data(args.data)
    print("1/5 main comparison")
    main_res, dedup_stats = main_comparison(alpaca, gsm_train, gsm_test, args)
    print("2/5 BM25 parameter sweep")
    sweep = bm25_sweep(alpaca, gsm_train, gsm_test, args)
    print("3/5 pool size scaling")
    scaling = pool_scaling(alpaca, gsm_train, gsm_test, args)
    print("4/5 error analysis")
    errors = error_analysis(alpaca, gsm_train, gsm_test, args)
    print("5/5 demo pool export")
    Path(args.demo_out).parent.mkdir(parents=True, exist_ok=True)
    export_demo(alpaca, gsm_train, gsm_test, args.demo_out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "main": main_res, "dedup": dedup_stats, "bm25_sweep": sweep,
               "pool_scaling": scaling, "errors": errors}, open(args.out, "w"), indent=2)

    for setting in ["full", "dedup", "length_matched"]:
        print(f"\nPool: {setting}")
        print(f"{'Method':<28}" + "".join(f"{m:>16}" for m in METRICS))
        for name, ms in main_res[setting].items():
            print(f"{name:<28}" + "".join(f"{ms[m]['mean']:>9.3f} ± {ms[m]['std']:.3f}" for m in METRICS))
    for setting, grid in sweep.items():
        best = max(grid, key=lambda r: r["P@10"])
        print(f"Best BM25 on {setting}: k1={best['k1']}, b={best['b']}, P@10={best['P@10']:.3f}")
    print(f"Saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
