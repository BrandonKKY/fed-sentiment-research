"""Embedding-based hawkish/dovish scoring (approach 2), fully local.

Why not a pretrained transformer: this machine runs Python 3.14 and, at the
time of writing, none of torch / onnxruntime / model2vec ship cp314 wheels,
so a HuggingFace scorer (e.g. FOMC-RoBERTa, Shah et al. 2023) is not
installable locally, and API-based scoring is ruled out by the no-API-cost
constraint. Instead we build classical count-based word embeddings from
scratch with numpy/scipy:

  * PPMI-weighted word-context co-occurrence matrix + truncated SVD.
    Levy & Goldberg (2014, NeurIPS) show word2vec's SGNS implicitly
    factorises this matrix; Levy, Goldberg & Dagan (2015, TACL) recommend
    the context-distribution smoothing (alpha = 0.75) and the symmetric
    sqrt-singular-value weighting used here.
  * A hawk-dove semantic axis from seed words (SemAxis: An, Kwak & Ahn
    2018, ACL): axis = mean(hawkish seed vectors) - mean(dovish seed
    vectors).
  * A sentence scores as the cosine between its mean content-word vector
    and the axis; a document scores as the mean over its sentences.

Look-ahead control: the embedding TRAINING corpus is restricted to FOMC
documents whose meeting date is on or before 2014-12-31 (pre-sample). The
vectors are then frozen; scoring a 2015+ document uses only that document's
own words. No text from the evaluation period influences the geometry.

Honesty rule for seeds: the seed lists below were fixed a priori on
economic grounds (words describing the policy stance itself) and MUST NOT
be edited in response to validation results. Any change invalidates the
out-of-sample claim and requires rerunning everything as a new experiment.

Hyperparameters (fixed a priori, not tuned): window=5, min_count=10,
dim=100, alpha=0.75.
"""

import json
import re

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .textclean import split_sentences

WINDOW = 5
MIN_COUNT = 10
DIM = 100
ALPHA = 0.75

HAWK_SEEDS = (
    "tighten tightening tighter restrictive restraint firming hike raise "
    "raising overheating inflationary".split()
)
DOVE_SEEDS = (
    "ease easing easier accommodative accommodation stimulus cut cuts "
    "lowering weakness downturn recession sluggish".split()
)

# Minimal stopword list; only used when averaging word vectors per sentence.
STOPWORDS = frozenset(
    "the a an and or of to in for on with at by from as is are was were be "
    "been being has have had that this these those it its their his her "
    "they he she we you i not no nor but if then than so such which who "
    "whom whose what when where how all any both each few more most other "
    "some own same s t can will just don should now would could may might "
    "must shall do does did doing during before after above below up down "
    "out off over under again further once here there about against "
    "between into through".split()
)

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(sentence):
    return _TOKEN_RE.findall(sentence.lower())


def sentences_to_token_lists(texts):
    """Flatten documents into per-sentence token lists (co-occurrence never
    crosses a sentence boundary)."""
    out = []
    for text in texts:
        for sent in split_sentences(text):
            toks = tokenize(sent)
            if len(toks) >= 3:
                out.append(toks)
    return out


def build_vocab(token_lists, min_count=MIN_COUNT):
    counts = {}
    for toks in token_lists:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    vocab = sorted(t for t, c in counts.items() if c >= min_count)
    return {t: i for i, t in enumerate(vocab)}


def cooccurrence_counts(token_lists, vocab, window=WINDOW):
    """Symmetric within-sentence co-occurrence counts."""
    pair_counts = {}
    for toks in token_lists:
        idx = [vocab.get(t, -1) for t in toks]
        n = len(idx)
        for i in range(n):
            wi = idx[i]
            if wi < 0:
                continue
            hi = min(n, i + window + 1)
            for j in range(i + 1, hi):
                wj = idx[j]
                if wj < 0:
                    continue
                key = (wi, wj) if wi <= wj else (wj, wi)
                pair_counts[key] = pair_counts.get(key, 0) + 1
    return pair_counts


def ppmi_matrix(pair_counts, vsize, alpha=ALPHA):
    """Positive pointwise mutual information with context smoothing."""
    rows, cols, vals = [], [], []
    for (i, j), c in pair_counts.items():
        rows.append(i)
        cols.append(j)
        vals.append(c)
        if i != j:
            rows.append(j)
            cols.append(i)
            vals.append(c)
    counts = sparse.csr_matrix(
        (np.array(vals, dtype=np.float64), (rows, cols)), shape=(vsize, vsize)
    )
    total = counts.sum()
    word_freq = np.asarray(counts.sum(axis=1)).ravel()
    ctx_freq = np.asarray(counts.sum(axis=0)).ravel() ** alpha
    ctx_freq /= ctx_freq.sum()
    word_prob = word_freq / total

    coo = counts.tocoo()
    with np.errstate(divide="ignore"):
        pmi = np.log(
            (coo.data / total)
            / (word_prob[coo.row] * ctx_freq[coo.col])
        )
    keep = pmi > 0
    return sparse.csr_matrix(
        (pmi[keep], (coo.row[keep], coo.col[keep])), shape=(vsize, vsize)
    )


def svd_embeddings(ppmi, dim=DIM, seed=0):
    """Truncated SVD; word vector = U * sqrt(S), L2-normalised rows."""
    k = min(dim, min(ppmi.shape) - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(ppmi.shape))
    u, s, _ = svds(ppmi, k=k, v0=v0)
    order = np.argsort(-s)
    vecs = u[:, order] * np.sqrt(s[order])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class FedEmbedder:
    def __init__(self, vocab, vectors, axis, meta=None):
        self.vocab = vocab          # {token: row}
        self.vectors = vectors      # (V, d), rows L2-normalised
        self.axis = axis            # (d,), L2-normalised hawk-minus-dove axis
        self.meta = meta or {}

    # -- construction --------------------------------------------------

    @classmethod
    def train(cls, texts):
        token_lists = sentences_to_token_lists(texts)
        vocab = build_vocab(token_lists)
        pair_counts = cooccurrence_counts(token_lists, vocab)
        ppmi = ppmi_matrix(pair_counts, len(vocab))
        vectors = svd_embeddings(ppmi)

        hawk_in = [w for w in HAWK_SEEDS if w in vocab]
        dove_in = [w for w in DOVE_SEEDS if w in vocab]
        if len(hawk_in) < 4 or len(dove_in) < 4:
            raise RuntimeError(
                f"too few seeds in vocab (hawk={hawk_in}, dove={dove_in}); "
                "corpus too small for a stable axis"
            )
        axis = vectors[[vocab[w] for w in hawk_in]].mean(axis=0) - vectors[
            [vocab[w] for w in dove_in]
        ].mean(axis=0)
        axis = axis / np.linalg.norm(axis)
        meta = {
            "n_sentences": len(token_lists),
            "n_tokens": int(sum(len(t) for t in token_lists)),
            "vocab_size": len(vocab),
            "hawk_seeds_in_vocab": hawk_in,
            "dove_seeds_in_vocab": dove_in,
            "window": WINDOW,
            "min_count": MIN_COUNT,
            "dim": int(vectors.shape[1]),
            "alpha": ALPHA,
        }
        return cls(vocab, vectors, axis, meta)

    # -- persistence ----------------------------------------------------

    def save(self, path):
        tokens = sorted(self.vocab, key=self.vocab.get)
        np.savez_compressed(
            path,
            tokens=np.array(tokens),
            vectors=self.vectors,
            axis=self.axis,
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        tokens = [str(t) for t in data["tokens"]]
        vocab = {t: i for i, t in enumerate(tokens)}
        meta = json.loads(str(data["meta"]))
        return cls(vocab, data["vectors"], data["axis"], meta)

    # -- scoring ----------------------------------------------------------

    def sentence_score(self, sentence):
        idxs = [
            self.vocab[t]
            for t in tokenize(sentence)
            if t in self.vocab and t not in STOPWORDS
        ]
        if not idxs:
            return None
        mean_vec = self.vectors[idxs].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm == 0:
            return None
        return float(mean_vec @ self.axis / norm)

    def score_text(self, text):
        """Mean sentence projection on the hawk-dove axis, plus coverage."""
        scores = []
        n_total = 0
        for sent in split_sentences(text):
            n_total += 1
            s = self.sentence_score(sent)
            if s is not None:
                scores.append(s)
        if not scores:
            return {"emb_score": float("nan"), "emb_coverage": 0.0}
        return {
            "emb_score": float(np.mean(scores)),
            "emb_coverage": len(scores) / max(n_total, 1),
        }

    # -- diagnostics ------------------------------------------------------

    def nearest(self, word, k=8):
        """k nearest vocabulary neighbours by cosine (sanity checking)."""
        if word not in self.vocab:
            return []
        v = self.vectors[self.vocab[word]]
        sims = self.vectors @ v
        order = np.argsort(-sims)
        tokens = sorted(self.vocab, key=self.vocab.get)
        out = []
        for i in order:
            if tokens[i] != word:
                out.append((tokens[i], float(sims[i])))
            if len(out) >= k:
                break
        return out
