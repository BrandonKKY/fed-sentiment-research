"""Event-study validation of FOMC sentiment scores (Task 3).

Design fixed BEFORE looking at any correlation output:

* Event timing. A statement is public at 14:00 ET on its meeting date;
  minutes at 14:00 ET on their release date. US equity/bond closes are at
  16:00 ET, so the close of the event day already reflects the text. The
  base observation for forward moves is therefore t0 = the first trading
  day >= avail_date; forward moves t0 -> t0+k use only post-publication
  information.
* Announcement-day reaction (t0-1 -> t0) is computed as a MEASURE
  VALIDATION check: hawkish statement surprises should move short-maturity
  yields up on the day (Kuttner 2001; Gurkaynak, Sack & Swanson 2005). It
  is NOT tradable -- signal and move are contemporaneous -- and is reported
  separately from the predictive tests.
* Signals: score level, and score change vs the previous document of the
  same type (a crude surprise proxy).
* Consecutive same-type events are ~6 weeks apart while the longest
  forward horizon is 21 trading days (~29 calendar days), so forward
  windows within an event set do not overlap; plain correlation tests are
  therefore not inflated by overlapping observations.
* Train/test split by avail_date: train <= 2020-12-31 < test. The grid is
  exploratory and Bonferroni-flagged. The PRIMARY hypothesis, stated a
  priori: the CHANGE in statement lex_abg_net correlates positively with
  (a) the announcement-day change and (b) the 5-day forward change of the
  2-year Treasury yield (DGS2).
"""

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = (1, 5, 21)
TRAIN_END = pd.Timestamp("2020-12-31")
SAMPLE_START = pd.Timestamp("2015-01-01")


class AssetSeries:
    """A daily close series with its own trading calendar.

    kind = "price": moves are simple returns close[a]/close[b] - 1.
    kind = "yield": moves are level differences in percentage points.
    """

    def __init__(self, name, dates, values, kind):
        if kind not in ("price", "yield"):
            raise ValueError(f"bad kind {kind!r}")
        order = np.argsort(dates)
        self.name = name
        self.dates = np.asarray(dates, dtype="datetime64[D]")[order]
        self.values = np.asarray(values, dtype=np.float64)[order]
        self.kind = kind

    @classmethod
    def from_csv(cls, path, name, kind, value_col):
        df = pd.read_csv(path, parse_dates=["date"])
        return cls(
            name, df["date"].values.astype("datetime64[D]"), df[value_col], kind
        )

    def _move(self, i_from, i_to):
        if self.kind == "price":
            return self.values[i_to] / self.values[i_from] - 1.0
        return self.values[i_to] - self.values[i_from]

    def event_moves(self, avail_dates, horizons=HORIZONS):
        """Announcement-day and forward moves for each event date.

        Returns a DataFrame indexed like avail_dates with columns
        t0_date, ann, fwd_<k>. NaN where the calendar runs out.

        Look-ahead guard: t0 is the first trading day >= avail_date, so the
        base close is never earlier than the text's publication.
        """
        avail = np.asarray(avail_dates, dtype="datetime64[D]")
        t0 = np.searchsorted(self.dates, avail, side="left")
        out = {
            "t0_date": np.full(len(avail), np.datetime64("NaT"), dtype="datetime64[D]"),
            "ann": np.full(len(avail), np.nan),
        }
        for k in horizons:
            out[f"fwd_{k}"] = np.full(len(avail), np.nan)
        for i, idx in enumerate(t0):
            if idx >= len(self.dates):
                continue
            assert self.dates[idx] >= avail[i], "look-ahead: base close precedes publication"
            out["t0_date"][i] = self.dates[idx]
            if idx > 0:
                out["ann"][i] = self._move(idx - 1, idx)
            for k in horizons:
                if idx + k < len(self.dates):
                    out[f"fwd_{k}"][i] = self._move(idx, idx + k)
        return pd.DataFrame(out)


def correlate(signal, outcome, min_n=8):
    """Pearson and Spearman correlation on pairwise-complete observations."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_n or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(x[mask], y[mask])
    sr = stats.spearmanr(x[mask], y[mask])
    return {
        "n": n,
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def build_signals(scores, score_cols):
    """Add <col>_chg columns: change vs the previous same-type document.

    Requires scores sorted by avail_date within each doc_type; the diff is
    strictly backward-looking (current minus previous document's score).
    """
    scores = scores.sort_values(["doc_type", "avail_date"]).copy()
    for col in score_cols:
        scores[f"{col}_chg"] = scores.groupby("doc_type")[col].diff()
    return scores
