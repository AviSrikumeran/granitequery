"""Ranking methods for Granite Query.

Every ranker takes a list of pool texts, is fit once, and then returns a
score for every pool item given a query string. Higher score means more
relevant to the target task.
"""

import re

import numpy as np
from datasketch import MinHash, MinHashLSH
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN_PATTERN = r"[a-z0-9]+"
TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text, remove_stop=False):
    toks = TOKEN_RE.findall(text.lower())
    if remove_stop:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    return toks


class RandomRanker:
    name = "Random"

    def __init__(self, texts, seed=0):
        self.n = len(texts)
        self.rng = np.random.default_rng(seed)

    def score(self, query):
        return self.rng.random(self.n)


class LengthRanker:
    """Sanity baseline that ignores the query and ranks longer items first."""
    name = "Length only"

    def __init__(self, texts):
        self.lengths = np.array([len(t.split()) for t in texts], dtype=float)

    def score(self, query):
        return self.lengths


class BM25Ranker:
    """Okapi BM25 on sparse matrices. The IDF matches the rank_bm25 package:
    log((N - n + 0.5) / (n + 0.5)), with negative values replaced by
    0.25 times the mean IDF."""

    def __init__(self, texts, k1=1.5, b=0.75, remove_stop=False, epsilon=0.25):
        self.name = "BM25 no stop words" if remove_stop else "BM25"
        self.vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True,
                                   stop_words=list(ENGLISH_STOP_WORDS) if remove_stop else None)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * idf.mean()
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / avgdl)
        w = tf.copy()
        w.data = idf[tf.indices] * tf.data * (k1 + 1) / denom
        self.w_t = w.T.tocsr()

    def score(self, query):
        q = self.vec.transform([query])
        return np.asarray((q @ self.w_t).todense()).ravel()


class TfidfRanker:
    name = "TF IDF"

    def __init__(self, texts):
        self.vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)
        self.docs = self.vec.fit_transform(texts)

    def score(self, query):
        return (self.docs @ self.vec.transform([query]).T).toarray().ravel()


class RocchioRanker(TfidfRanker):
    """TF IDF with Rocchio pseudo relevance feedback: assume the top k
    results are relevant and move the query vector toward their centroid."""
    name = "TF IDF + Rocchio feedback"

    def __init__(self, texts, k=10, alpha=1.0, beta=0.75):
        super().__init__(texts)
        self.k, self.alpha, self.beta = k, alpha, beta

    def score(self, query):
        q = self.vec.transform([query])
        first = (self.docs @ q.T).toarray().ravel()
        top = np.argsort(-first)[: self.k]
        centroid = sparse.csr_matrix(self.docs[top].mean(axis=0))
        q2 = normalize(self.alpha * q + self.beta * centroid)
        return (self.docs @ q2.T).toarray().ravel()


class LSARanker:
    """Latent semantic analysis: TF IDF projected to 200 SVD dimensions,
    then cosine similarity in that space."""
    name = "LSA (200 dims)"

    def __init__(self, texts, dims=200, seed=0):
        self.vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)
        x = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.docs = normalize(self.svd.fit_transform(x))

    def score(self, query):
        q = normalize(self.svd.transform(self.vec.transform([query])))
        return self.docs @ q.ravel()


def minhash_dedup(texts, threshold=0.8, num_perm=128):
    """Keep the first item of each near duplicate cluster, using word 3 gram shingles."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, t in enumerate(texts):
        toks = tokenize(t)
        shingles = {" ".join(toks[j:j + 3]) for j in range(max(1, len(toks) - 2))}
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf8"))
        if not lsh.query(m):
            lsh.insert(str(i), m)
            keep.append(i)
    return keep
