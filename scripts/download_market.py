"""Task 3 prerequisite: download market data (free, keyless sources).

  data/market/spy.csv    SPY daily closes (Stooq)
  data/market/dgs2.csv   2y Treasury constant-maturity yield (FRED DGS2)
  data/market/dgs10.csv  10y Treasury constant-maturity yield (FRED DGS10)

Starts in 2014 to give a buffer of prior closes before the first 2015 event.
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedsent.market import fetch_fred_series, fetch_yahoo_daily  # noqa: E402

MARKET = ROOT / "data" / "market"


def describe(name, df):
    print(
        f"  {name}: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}"
    )


def main():
    MARKET.mkdir(parents=True, exist_ok=True)
    start = dt.date(2014, 1, 1)
    end = dt.date.today()

    spy = fetch_yahoo_daily("SPY", start, end)
    spy.to_csv(MARKET / "spy.csv", index=False)
    describe("SPY (Yahoo)", spy)

    for series in ("DGS2", "DGS10"):
        df = fetch_fred_series(series)
        df = df[df["date"] >= pd.Timestamp(start)]
        df.to_csv(MARKET / f"{series.lower()}.csv", index=False)
        describe(f"{series} (FRED)", df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
