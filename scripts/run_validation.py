"""Task 3: honest validation of the sentiment scores against markets.

Reads data/processed/scores.csv and data/market/*.csv, runs the pre-
registered event study (see fedsent/validate.py docstring), and writes:

  output/validation_grid.csv   every (split, doc_type, signal, asset,
                               outcome) cell with N, Pearson, Spearman
  output/events_statements.csv per-event table used for figures
  output/summary.txt           headline results: measure-validation check,
                               primary hypothesis train/test, Bonferroni
                               threshold, and the train-top-5 replication
                               table

Nothing in here tunes on test data. The primary hypothesis was fixed a
priori in fedsent/validate.py; everything else is labelled exploratory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedsent.validate import (  # noqa: E402
    HORIZONS,
    SAMPLE_START,
    TRAIN_END,
    AssetSeries,
    build_signals,
    correlate,
)

PROCESSED = ROOT / "data" / "processed"
MARKET = ROOT / "data" / "market"
OUTPUT = ROOT / "output"

SCORE_COLS = ("lex_abg_net", "lex_abg_fomc_net", "emb_score")


def load_assets():
    return [
        AssetSeries.from_csv(MARKET / "spy.csv", "SPY", "price", "close"),
        AssetSeries.from_csv(MARKET / "dgs2.csv", "DGS2", "yield", "value"),
        AssetSeries.from_csv(MARKET / "dgs10.csv", "DGS10", "yield", "value"),
    ]


def main():
    OUTPUT.mkdir(exist_ok=True)
    scores = pd.read_csv(
        PROCESSED / "scores.csv", parse_dates=["meeting_date", "avail_date"]
    )
    scores = build_signals(scores, SCORE_COLS)
    sample = scores[scores["avail_date"] >= SAMPLE_START].copy()
    n_est = int(sample.loc[sample.doc_type == "minutes", "release_estimated"].sum())
    print(
        f"validation sample: {len(sample)} events "
        f"({(sample.doc_type == 'statement').sum()} statements, "
        f"{(sample.doc_type == 'minutes').sum()} minutes; "
        f"{n_est} minutes with estimated release dates)"
    )

    assets = load_assets()
    signal_cols = [c for col in SCORE_COLS for c in (col, f"{col}_chg")]

    grid_rows = []
    event_tables = {}
    for doc_type, events in sample.groupby("doc_type"):
        events = events.sort_values("avail_date").reset_index(drop=True)
        avail = events["avail_date"].values.astype("datetime64[D]")
        merged = events.copy()
        for asset in assets:
            moves = asset.event_moves(avail)
            for out_col in ["ann"] + [f"fwd_{k}" for k in HORIZONS]:
                merged[f"{asset.name}_{out_col}"] = moves[out_col].values
            for split_name, split_mask in (
                ("train", events["avail_date"] <= TRAIN_END),
                ("test", events["avail_date"] > TRAIN_END),
            ):
                for sig in signal_cols:
                    for out_col in ["ann"] + [f"fwd_{k}" for k in HORIZONS]:
                        res = correlate(
                            events.loc[split_mask, sig],
                            moves.loc[split_mask.values, out_col],
                        )
                        grid_rows.append(
                            {
                                "split": split_name,
                                "doc_type": doc_type,
                                "signal": sig,
                                "asset": asset.name,
                                "outcome": out_col,
                                **res,
                            }
                        )
        event_tables[doc_type] = merged

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUTPUT / "validation_grid.csv", index=False)
    event_tables["statement"].drop(columns=[], errors="ignore").to_csv(
        OUTPUT / "events_statements.csv", index=False
    )
    event_tables["minutes"].to_csv(OUTPUT / "events_minutes.csv", index=False)

    # ------------------------------------------------------------------ #
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    fwd = grid[grid["outcome"] != "ann"]
    m_fwd = len(fwd[fwd["split"] == "train"])
    bonf = 0.05 / m_fwd
    emit("=" * 72)
    emit("HONEST VALIDATION SUMMARY (see REPORT.md for interpretation)")
    emit("=" * 72)
    emit(f"forward-return grid: {m_fwd} hypotheses per split; "
         f"Bonferroni alpha = 0.05/{m_fwd} = {bonf:.2e}")

    emit("\n--- Measure validation (announcement-day, NOT tradable) ---")
    emit("a priori check: hawkish statement surprise (lex_abg_net_chg) should")
    emit("move 2y yields UP on the day.")
    for split in ("train", "test"):
        row = grid[
            (grid.split == split)
            & (grid.doc_type == "statement")
            & (grid.signal == "lex_abg_net_chg")
            & (grid.asset == "DGS2")
            & (grid.outcome == "ann")
        ].iloc[0]
        emit(
            f"  {split:5s}: n={row.n:3.0f}  pearson r={row.pearson_r:+.3f} "
            f"(p={row.pearson_p:.4f})  spearman rho={row.spearman_rho:+.3f} "
            f"(p={row.spearman_p:.4f})"
        )

    emit("\n--- PRIMARY pre-registered predictive hypothesis ---")
    emit("statement lex_abg_net_chg -> 5-day forward DGS2 change (positive)")
    for split in ("train", "test"):
        row = grid[
            (grid.split == split)
            & (grid.doc_type == "statement")
            & (grid.signal == "lex_abg_net_chg")
            & (grid.asset == "DGS2")
            & (grid.outcome == "fwd_5")
        ].iloc[0]
        emit(
            f"  {split:5s}: n={row.n:3.0f}  pearson r={row.pearson_r:+.3f} "
            f"(p={row.pearson_p:.4f})  spearman rho={row.spearman_rho:+.3f} "
            f"(p={row.spearman_p:.4f})"
        )

    emit("\n--- Exploratory grid: top-5 train correlations vs their test twins ---")
    train_fwd = fwd[(fwd.split == "train") & fwd.pearson_r.notna()].copy()
    train_fwd["abs_r"] = train_fwd.pearson_r.abs()
    top = train_fwd.nlargest(5, "abs_r")
    emit(f"{'doc_type':10s} {'signal':22s} {'asset':6s} {'outcome':8s} "
         f"{'train r':>9s} {'train p':>9s} {'test r':>9s} {'test p':>9s}")
    for _, r in top.iterrows():
        twin = fwd[
            (fwd.split == "test")
            & (fwd.doc_type == r.doc_type)
            & (fwd.signal == r.signal)
            & (fwd.asset == r.asset)
            & (fwd.outcome == r.outcome)
        ].iloc[0]
        emit(
            f"{r.doc_type:10s} {r.signal:22s} {r.asset:6s} {r.outcome:8s} "
            f"{r.pearson_r:+9.3f} {r.pearson_p:9.4f} "
            f"{twin.pearson_r:+9.3f} {twin.pearson_p:9.4f}"
        )

    emit("\n--- Bonferroni survivors in the forward grid (both splits) ---")
    surv = fwd[fwd.pearson_p < bonf]
    if len(surv) == 0:
        emit("  NONE. No forward-looking correlation survives multiple-testing")
        emit("  correction in either split.")
    else:
        for _, r in surv.iterrows():
            emit(
                f"  {r.split} {r.doc_type} {r.signal} {r.asset} {r.outcome}: "
                f"r={r.pearson_r:+.3f} p={r.pearson_p:.2e} n={r.n:.0f}"
            )

    ann = grid[grid["outcome"] == "ann"]
    m_ann = len(ann[ann["split"] == "train"])
    bonf_ann = 0.05 / m_ann
    emit(f"\n--- Announcement-day family ({m_ann} tests/split, "
         f"Bonferroni alpha = {bonf_ann:.2e}) ---")
    emit("contemporaneous, NOT tradable; survivors:")
    surv_ann = ann[ann.pearson_p < bonf_ann]
    if len(surv_ann) == 0:
        emit("  NONE.")
    else:
        for _, r in surv_ann.iterrows():
            emit(
                f"  {r.split} {r.doc_type} {r.signal} {r.asset}: "
                f"r={r.pearson_r:+.3f} p={r.pearson_p:.2e} n={r.n:.0f}"
            )

    emit("\n--- Method agreement across all 2015+ events ---")
    for dt_ in ("statement", "minutes"):
        sub = sample[sample.doc_type == dt_]
        r12 = sub["lex_abg_net"].corr(sub["lex_abg_fomc_net"])
        r13 = sub["lex_abg_net"].corr(sub["emb_score"])
        r23 = sub["lex_abg_fomc_net"].corr(sub["emb_score"])
        emit(
            f"  {dt_:10s} corr(abg, abg_fomc)={r12:+.2f}  "
            f"corr(abg, emb)={r13:+.2f}  corr(abg_fomc, emb)={r23:+.2f}"
        )

    (OUTPUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUTPUT / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
