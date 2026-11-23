# Granite Query: Ranking Training Data as a Retrieval Problem

CS 410 Text Information Systems, Fall 2026 final project
Avinash Sri-kumeran (aus4), working alone

Project website with interactive charts and a live search demo: https://avisrikumeran.github.io/granitequery/ (also https://granitequery.live)

## 1. Overview

Before you fine tune a language model such as IBM Granite for a task, you have to decide which training examples to use. General instruction pools like Alpaca are large and mostly unrelated to any one task. This project treats data selection as a retrieval problem:

> Given a math word problem as the query, how well can a ranker pull math word problems to the top of a mixed training pool?

The software builds mixed candidate pools from Alpaca and GSM8K, ranks every pool item for every query with seven ranking methods, scores the rankings with standard IR metrics, and publishes the results on a static website. The site also has a BM25 search engine that runs in the browser.

The project started as a full Granite fine tuning study on Modal. Accuracy got worse after each round of ranking and retraining, and there was no GPU cluster access, so the scope narrowed to the ranking step. The ranking step can be measured exactly, and it runs on a laptop CPU.

### Main findings

1. On the length matched pool, which is the fair comparison, **BM25 is the best ranker** on three of the four metrics: P@10 0.706 vs 0.104 for random. LSA has a slightly higher MRR (0.874 vs 0.857). BM25 is also the most stable real ranker across seeds (P@10 std 0.006).
2. **Document length is a hidden shortcut.** GSM8K problems have a median of about 42 words and Alpaca items about 11. On the standard pool, ranking by length alone gives P@10 0.467. That inflates LSA (P@10 0.888) and BM25 with no length normalization (b = 0 and k1 = 1.0 reach P@10 0.974 in the single seed sweep). Both advantages go away once Alpaca is length matched.
3. **Removing stop words hurts** in every pool. Words like "how", "many", and "each" are the strongest signal that a text is a word problem.
4. **Rocchio pseudo relevance feedback adds almost nothing**, and on the length matched pool it slightly hurts.
5. **MinHash deduplication** caught 68 to 84 of the 100 planted near duplicates per seed, with zero false removals.
6. Precision falls as the pool grows (BM25 P@10 from 0.92 at 1,600 items to 0.55 at 40,600, single seed), but BM25 stays far above random at every size.
7. **No query leakage.** None of the 1,319 GSM8K test questions has an exact or near duplicate (MinHash Jaccard 0.7 or higher) in the GSM8K training set, so the queries never appear in the pool.

## 2. Experimental design

| Part | Setting |
|---|---|
| Standard pool | 5,000 random Alpaca items, 500 GSM8K training problems, 100 near duplicates (a GSM8K problem with its first number increased by 1) |
| Deduplicated pool | Standard pool after MinHash LSH (word 3 gram shingles, 128 permutations, Jaccard threshold 0.8) |
| Length matched pool | 1,800 Alpaca items sampled so their word counts follow the GSM8K length distribution, plus 180 GSM8K problems and 36 near duplicates |
| Queries | 300 GSM8K test questions per seed (never in the pool) |
| Relevance | A pool item is relevant if it is a GSM8K problem or a near duplicate of one |
| Metrics | Precision@10, MRR, nDCG@10, Recall@100 |
| Repeats | 3 seeds (0, 1, 2); each seed draws a new pool and new queries |

Rankers:

| Ranker | Description |
|---|---|
| Random | Uniform random scores |
| Length only | Longer items first; ignores the query (sanity check) |
| TF IDF | scikit learn TF IDF with cosine similarity |
| TF IDF + Rocchio | Pseudo relevance feedback: top 10 treated as relevant, q' = 1.0·q + 0.75·centroid |
| LSA | TF IDF reduced to 200 dimensions with truncated SVD, cosine similarity |
| BM25 | Okapi BM25, k1 = 1.5, b = 0.75 |
| BM25 no stop words | Same, with scikit learn's English stop word list removed from queries and documents |

## 3. Results (mean ± std over 3 seeds)

### Length matched pool (fair comparison)

| Ranker | P@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| Random | 0.104 ± 0.005 | 0.258 ± 0.014 | 0.104 ± 0.007 | 0.050 ± 0.000 |
| Length only | 0.033 ± 0.047 | 0.079 ± 0.029 | 0.022 ± 0.031 | 0.049 ± 0.010 |
| TF IDF | 0.515 ± 0.026 | 0.752 ± 0.016 | 0.540 ± 0.021 | 0.163 ± 0.009 |
| TF IDF + Rocchio | 0.510 ± 0.031 | 0.736 ± 0.018 | 0.540 ± 0.029 | 0.155 ± 0.014 |
| LSA | 0.664 ± 0.033 | 0.874 ± 0.020 | 0.697 ± 0.032 | 0.195 ± 0.014 |
| **BM25** | **0.706 ± 0.006** | 0.857 ± 0.006 | **0.720 ± 0.006** | **0.212 ± 0.005** |
| BM25 no stop words | 0.554 ± 0.012 | 0.768 ± 0.020 | 0.576 ± 0.010 | 0.181 ± 0.004 |

### Standard pool

| Ranker | P@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| Random | 0.111 ± 0.003 | 0.272 ± 0.009 | 0.110 ± 0.004 | 0.018 ± 0.000 |
| Length only | 0.467 ± 0.094 | 0.389 ± 0.079 | 0.417 ± 0.061 | 0.119 ± 0.004 |
| TF IDF | 0.644 ± 0.017 | 0.811 ± 0.029 | 0.658 ± 0.022 | 0.096 ± 0.002 |
| TF IDF + Rocchio | 0.652 ± 0.018 | 0.839 ± 0.026 | 0.681 ± 0.022 | 0.104 ± 0.002 |
| LSA | 0.888 ± 0.010 | 0.950 ± 0.010 | 0.896 ± 0.010 | 0.131 ± 0.002 |
| BM25 | 0.786 ± 0.010 | 0.898 ± 0.016 | 0.797 ± 0.013 | 0.115 ± 0.002 |
| BM25 no stop words | 0.578 ± 0.009 | 0.787 ± 0.013 | 0.600 ± 0.011 | 0.079 ± 0.002 |

### Deduplicated pool

| Ranker | P@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| Random | 0.096 ± 0.004 | 0.234 ± 0.005 | 0.094 ± 0.003 | 0.018 ± 0.000 |
| Length only | 0.467 ± 0.094 | 0.389 ± 0.079 | 0.414 ± 0.057 | 0.132 ± 0.006 |
| TF IDF | 0.620 ± 0.022 | 0.816 ± 0.027 | 0.638 ± 0.024 | 0.104 ± 0.002 |
| TF IDF + Rocchio | 0.624 ± 0.023 | 0.808 ± 0.029 | 0.644 ± 0.026 | 0.112 ± 0.002 |
| LSA | 0.885 ± 0.007 | 0.955 ± 0.010 | 0.894 ± 0.009 | 0.149 ± 0.001 |
| BM25 | 0.768 ± 0.013 | 0.898 ± 0.017 | 0.782 ± 0.015 | 0.126 ± 0.002 |
| BM25 no stop words | 0.548 ± 0.012 | 0.787 ± 0.013 | 0.574 ± 0.013 | 0.085 ± 0.002 |

Recall@100 has a ceiling because each pool has more than 100 relevant items (about 0.167 on the standard pool and 0.463 on the length matched pool).

The website also has the BM25 k1 × b sweep (25 settings on two pools), the pool size scaling curve, and the error analysis.

## 4. Implementation

```
granitequery/
├── ranking_study/               CS 410 submission code (everything below)
│   ├── granite_query/
│   │   ├── ranking.py           all rankers + MinHash deduplication
│   │   └── evaluation.py        data loading, pool construction, metrics
│   ├── run_experiments.py       runs all five experiments, writes results/results.json
│   ├── build_site.py            builds ../docs/index.html from the results
│   ├── download_data.py         fetches GSM8K and Alpaca from GitHub
│   ├── site/template.html       website template (charts and demo in plain JavaScript)
│   ├── tests/                   unit tests (python -m unittest discover -s tests)
│   └── results/                 results.json and the website demo pool
├── docs/index.html              the project website (GitHub Pages and Vercel serve docs/)
└── README_original.md           notes on the earlier fine tuning version of the project
```

Key implementation details:

1. **BM25** (`BM25Ranker`) is written from scratch on scipy sparse matrices. The document term weight matrix is computed once, and each query is a single sparse matrix product. The IDF formula matches the `rank_bm25` package, log((N − n + 0.5)/(n + 0.5)) with negative values floored at 0.25 × mean IDF. Scores were checked against `rank_bm25` and agree to within 1e−12. For 100 queries on the 5,600 item pool it runs about 30 times faster than the package, which keeps the 25 setting sweep cheap.
2. **Rocchio feedback** (`RocchioRanker`) runs a first TF IDF retrieval, takes the centroid of the top 10 document vectors, forms q' = αq + β·centroid, normalizes it, and retrieves again.
3. **LSA** (`LSARanker`) fits `TruncatedSVD` with 200 components on the pool's TF IDF matrix and ranks by cosine similarity in the reduced space.
4. **MinHash dedup** (`minhash_dedup`) uses `datasketch` with word 3 gram shingles. It keeps the first item of each near duplicate group.
5. **Length matched sampling** (`build_matched_pool`) puts GSM8K and Alpaca items into word count buckets (0 to 25, 25 to 35, 35 to 45, 45 to 55, 55 to 70, and over 70) and samples Alpaca to match GSM8K's share in each bucket.
6. **Metrics** (`metrics_for`) use binary relevance. nDCG uses log2 discounts with the ideal DCG computed from the true number of relevant items.
7. **Website**: `build_site.py` puts the results and a 3,360 item demo pool into `docs/index.html`. The page reuses the site's shared component styles (`docs/site/site.css`, `docs/site/home.css`) and its animated hero art (`docs/site/hero.js`). Charts are hand drawn SVG with hover tooltips and table views, and the demo uses the same BM25 formula in JavaScript.

## 5. Installation and usage

Requirements: Python 3.10 or newer. No GPU or API key is needed.

```bash
git clone https://github.com/AviSrikumeran/granitequery
cd granitequery/ranking_study
pip install -r requirements.txt
python download_data.py        # about 27 MB into data/
python run_experiments.py      # about 1 minute on a laptop CPU
python build_site.py           # writes ../docs/index.html
```

To run the unit tests (16 tests on synthetic data, no download needed):

```bash
python -m unittest discover -s tests -v
```

One test also checks BM25 against the `rank_bm25` reference package, which is installed by requirements.txt.

Open `docs/index.html` (at the repository root) in a browser to see the site locally. GitHub Pages and Vercel both publish the `docs/` folder from the `main` branch.

Useful options for `run_experiments.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--n_alpaca` | 5000 | Alpaca items in the standard pool |
| `--n_gsm` | 500 | GSM8K problems in the standard pool |
| `--n_dupes` | 100 | Planted near duplicates |
| `--n_queries` | 300 | GSM8K test queries per seed |
| `--n_matched_alpaca` | 1800 | Alpaca items in the length matched pool (at most about 1,900 fit the long length buckets) |
| `--seeds` | 0 1 2 | Random seeds |

You can also use the rankers directly:

```python
from granite_query.ranking import BM25Ranker
pool = ["Tom has 3 apples and buys 5 more. How many apples does he have?",
        "Write a haiku about autumn.",
        "Translate this sentence into French.",
        "Name three primary colors.",
        "Summarize the plot of a famous novel."]
scores = BM25Ranker(pool).score("Sara has 4 pens and buys 2 more. How many pens?")
print(scores.round(2))  # [4.55 0. 0. 0. 0.]: the math problem ranks first
```

## 6. Limitations and future work

1. Relevance comes from which dataset an item belongs to, so a real math question inside Alpaca counts as not relevant.
2. The evaluation measures retrieval quality, not downstream model quality. The next step is to fine tune Granite on BM25 ranked subsets and random subsets of equal size and compare GSM8K accuracy.
3. All rankers are lexical or linear. A neural embedding ranker, compared on the length matched pool, is the natural next baseline.

## 7. Team contribution

This was a solo project. Avinash Sri-kumeran designed the experiments, implemented the rankers, evaluation, and website, ran all experiments, and wrote the documentation.

## Data and licenses

GSM8K (MIT license) comes from https://github.com/openai/grade-school-math. Stanford Alpaca (CC BY NC 4.0) comes from https://github.com/tatsu-lab/stanford_alpaca. The data is downloaded at run time and is not stored in this repository, except for the small demo pool used by the website.

## Earlier version of this project

The rest of this repository (`granite_query/` at the root, `api/`, `server/`, and the other pages in `docs/`) belongs to the original fine tuning version of Granite Query, which continues as a separate project. See `README_original.md`. The earlier homepage is saved as `docs/original.html`, and the full earlier site is on the `original-site-backup` branch.
