"""Task 2: score every FOMC document with both sentiment methods.

1. Trains the PPMI-SVD embedding space on the PRE-SAMPLE corpus only
   (documents with meeting_date <= 2014-12-31) and saves it, plus
   nearest-neighbour sanity diagnostics.
2. Scores every statement and minutes document with:
     - lexicon variant "abg"       (Apel & Blix Grimaldi 2012, as published)
     - lexicon variant "abg_fomc"  (our documented FOMC-English extension)
     - "emb_score"                 (SemAxis projection in the frozen space)
3. Writes data/processed/scores.csv with one row per document, including
   avail_date: the first date the text was PUBLIC (statements: meeting day,
   14:00 ET; minutes: their release date, ~3 weeks later).
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedsent.embed import FedEmbedder  # noqa: E402
from fedsent.lexicon import VARIANTS, score_text  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
PRETRAIN_CUTOFF = dt.date(2014, 12, 31)
EMB_PATH = PROCESSED / "fed_embeddings.npz"

SANITY_WORDS = ("inflation", "unemployment", "accommodation", "tightening")


def load_documents():
    stmt = pd.read_csv(PROCESSED / "statements.csv", parse_dates=["meeting_date"])
    mins = pd.read_csv(
        PROCESSED / "minutes.csv", parse_dates=["meeting_date", "release_date"]
    )
    stmt["doc_type"] = "statement"
    stmt["avail_date"] = stmt["meeting_date"]
    stmt["release_estimated"] = False
    mins["doc_type"] = "minutes"
    mins["avail_date"] = mins["release_date"]
    cols = [
        "doc_type",
        "meeting_date",
        "avail_date",
        "release_estimated",
        "n_chars",
        "text",
    ]
    return pd.concat([stmt[cols], mins[cols]], ignore_index=True).sort_values(
        ["avail_date", "doc_type"]
    )


def main():
    docs = load_documents()
    print(f"loaded {len(docs)} documents "
          f"({(docs.doc_type == 'statement').sum()} statements, "
          f"{(docs.doc_type == 'minutes').sum()} minutes)")

    pretrain_mask = docs["meeting_date"].dt.date <= PRETRAIN_CUTOFF
    pretrain_texts = docs.loc[pretrain_mask, "text"].tolist()
    print(
        f"embedding pretrain corpus: {len(pretrain_texts)} documents "
        f"(meeting_date <= {PRETRAIN_CUTOFF})"
    )
    if EMB_PATH.exists():
        embedder = FedEmbedder.load(EMB_PATH)
        print(f"loaded cached embeddings: {embedder.meta}")
    else:
        embedder = FedEmbedder.train(pretrain_texts)
        embedder.save(EMB_PATH)
        print(f"trained embeddings: {embedder.meta}")

    print("\nnearest-neighbour sanity check (frozen pre-2015 space):")
    for word in SANITY_WORDS:
        nn = ", ".join(f"{w} {s:.2f}" for w, s in embedder.nearest(word, k=6))
        print(f"  {word}: {nn}")

    rows = []
    for _, doc in docs.iterrows():
        row = {
            "doc_type": doc.doc_type,
            "meeting_date": doc.meeting_date.date(),
            "avail_date": doc.avail_date.date(),
            "release_estimated": bool(doc.release_estimated),
            "n_chars": doc.n_chars,
        }
        for variant in VARIANTS:
            row.update(score_text(doc.text, variant))
        row.update(embedder.score_text(doc.text))
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(PROCESSED / "scores.csv", index=False)

    print(f"\nwrote {len(scores)} rows to scores.csv")
    for col in ("lex_abg_net", "lex_abg_fomc_net", "emb_score"):
        by_type = scores.groupby("doc_type")[col]
        print(f"\n{col}:")
        print(by_type.describe()[["count", "mean", "std", "min", "max"]])
    zero_hit = (
        (scores["lex_abg_hawk"] + scores["lex_abg_dove"]) == 0
    ).groupby(scores["doc_type"]).sum()
    print(f"\ndocuments with ZERO abg lexicon hits (score pinned at 0):")
    print(zero_hit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
