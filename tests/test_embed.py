"""Tests for the PPMI-SVD embedding scorer, on synthetic corpora only."""

import numpy as np
import pytest

from fedsent.embed import (
    FedEmbedder,
    build_vocab,
    cooccurrence_counts,
    ppmi_matrix,
    sentences_to_token_lists,
    svd_embeddings,
)

# A tiny synthetic corpus with two clearly separated "topics": hawkish seed
# words co-occur with 'pressures'/'overheating' contexts, dovish seeds with
# 'support'/'slack'. Repetition satisfies MIN_COUNT=10.
HAWK_SENT = "The committee will tighten policy and raise rates as inflationary pressures build."
DOVE_SENT = "The committee will ease policy and cut rates to support demand amid slack."
CORPUS = [" ".join([HAWK_SENT] * 12 + [DOVE_SENT] * 12)]
# extra seed forms so FedEmbedder.train finds >= 4 per side
CORPUS[0] += " " + " ".join(
    ["Tightening continued and raising rates was discussed by the committee."] * 12
    + ["Easing continued and cuts were discussed by the committee."] * 12
)


def test_build_vocab_min_count():
    toks = sentences_to_token_lists(CORPUS)
    vocab = build_vocab(toks, min_count=10)
    assert "tighten" in vocab and "ease" in vocab
    # every sentence starts with "the committee", so these are frequent
    assert "committee" in vocab


def test_cooccurrence_is_symmetric_and_windowed():
    vocab = {"a": 0, "b": 1, "c": 2}
    counts = cooccurrence_counts([["a", "b", "c"]], vocab, window=1)
    assert counts[(0, 1)] == 1 and counts[(1, 2)] == 1
    assert (0, 2) not in counts  # distance 2 > window 1


def test_ppmi_nonnegative_and_shape():
    toks = sentences_to_token_lists(CORPUS)
    vocab = build_vocab(toks, min_count=10)
    ppmi = ppmi_matrix(cooccurrence_counts(toks, vocab), len(vocab))
    assert ppmi.shape == (len(vocab), len(vocab))
    assert ppmi.data.min() > 0  # positive PMI only


def test_svd_rows_are_unit_norm():
    toks = sentences_to_token_lists(CORPUS)
    vocab = build_vocab(toks, min_count=10)
    ppmi = ppmi_matrix(cooccurrence_counts(toks, vocab), len(vocab))
    vecs = svd_embeddings(ppmi, dim=8)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_axis_orders_hawkish_above_dovish():
    emb = FedEmbedder.train(CORPUS)
    hawk_score = emb.sentence_score("They may tighten policy and raise rates.")
    dove_score = emb.sentence_score("They may ease policy and cut rates.")
    assert hawk_score is not None and dove_score is not None
    assert hawk_score > dove_score


def test_save_load_roundtrip(tmp_path):
    emb = FedEmbedder.train(CORPUS)
    path = tmp_path / "emb.npz"
    emb.save(path)
    loaded = FedEmbedder.load(path)
    np.testing.assert_allclose(loaded.vectors, emb.vectors)
    np.testing.assert_allclose(loaded.axis, emb.axis)
    assert loaded.meta == emb.meta
    s = "Inflationary pressures may build and the committee could tighten policy."
    assert loaded.score_text(s) == pytest.approx(emb.score_text(s))


def test_score_text_handles_full_oov():
    emb = FedEmbedder.train(CORPUS)
    out = emb.score_text("Zebra xylophone quixotic.")
    assert out["emb_coverage"] == 0.0
    assert np.isnan(out["emb_score"])
