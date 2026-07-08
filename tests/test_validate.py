"""Event-study machinery tests on synthetic data (no market downloads)."""

import numpy as np
import pandas as pd
import pytest

from fedsent.validate import AssetSeries, build_signals, correlate

# Ten consecutive weekdays, Mon 2024-01-01 .. Fri 2024-01-12.
DATES = np.array(
    [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ],
    dtype="datetime64[D]",
)
PRICES = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])


def test_event_on_trading_day():
    a = AssetSeries("X", DATES, PRICES, "price")
    moves = a.event_moves(np.array(["2024-01-03"], dtype="datetime64[D]"), horizons=(1, 2))
    assert moves["t0_date"][0] == np.datetime64("2024-01-03")
    assert moves["ann"][0] == pytest.approx(102.0 / 101.0 - 1)
    assert moves["fwd_1"][0] == pytest.approx(103.0 / 102.0 - 1)
    assert moves["fwd_2"][0] == pytest.approx(104.0 / 102.0 - 1)


def test_event_on_weekend_maps_to_next_trading_day():
    # Sat 2024-01-06 -> base close must be Mon 2024-01-08, never Friday
    # (using Friday's close would let a post-Friday text "predict" the
    # weekend gap = look-ahead).
    a = AssetSeries("X", DATES, PRICES, "price")
    moves = a.event_moves(np.array(["2024-01-06"], dtype="datetime64[D]"), horizons=(1,))
    assert moves["t0_date"][0] == np.datetime64("2024-01-08")
    assert moves["fwd_1"][0] == pytest.approx(106.0 / 105.0 - 1)


def test_yield_kind_uses_differences():
    a = AssetSeries("Y", DATES, PRICES / 25.0, "yield")
    moves = a.event_moves(np.array(["2024-01-02"], dtype="datetime64[D]"), horizons=(1,))
    assert moves["fwd_1"][0] == pytest.approx((102.0 - 101.0) / 25.0)


def test_calendar_runout_gives_nan():
    a = AssetSeries("X", DATES, PRICES, "price")
    moves = a.event_moves(np.array(["2024-01-12", "2024-02-01"], dtype="datetime64[D]"), horizons=(1,))
    assert np.isnan(moves["fwd_1"][0])          # last trading day: no forward close
    assert pd.isna(moves["t0_date"][1])         # event after the series ends
    assert np.isnan(moves["ann"][1])


def test_correlate_perfect_and_degenerate():
    x = np.arange(10.0)
    res = correlate(x, 2 * x + 1)
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["pearson_p"] < 1e-8
    # too few observations -> NaN, never a fabricated number
    res = correlate(x[:3], x[:3])
    assert np.isnan(res["pearson_r"]) and res["n"] == 3
    # zero-variance signal -> NaN
    res = correlate(np.ones(10), x)
    assert np.isnan(res["pearson_r"])


def test_correlate_ignores_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0])
    res = correlate(x, y, min_n=5)
    assert res["n"] == 7
    assert res["pearson_r"] == pytest.approx(1.0)


def test_build_signals_diffs_within_doc_type():
    scores = pd.DataFrame(
        {
            "doc_type": ["statement", "minutes", "statement", "minutes"],
            "avail_date": pd.to_datetime(
                ["2024-01-31", "2024-02-21", "2024-03-20", "2024-04-10"]
            ),
            "s": [0.1, 1.0, 0.4, 1.5],
        }
    )
    out = build_signals(scores, ["s"])
    stmt = out[out.doc_type == "statement"].sort_values("avail_date")
    mins = out[out.doc_type == "minutes"].sort_values("avail_date")
    assert np.isnan(stmt["s_chg"].iloc[0])                       # no prior statement
    assert stmt["s_chg"].iloc[1] == pytest.approx(0.3)           # 0.4 - 0.1
    assert mins["s_chg"].iloc[1] == pytest.approx(0.5)           # 1.5 - 1.0, never mixes types
