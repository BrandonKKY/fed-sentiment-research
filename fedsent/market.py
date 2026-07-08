"""Free, keyless market data loaders.

* SPY daily closes from Yahoo Finance's public v8 chart endpoint (the same
  one yfinance wraps). Dividend-adjusted closes are used so multi-day
  returns are total-return-like. (Stooq was tried first but now sits behind
  a JavaScript proof-of-work wall.)
* Treasury constant-maturity yields (DGS2, DGS10) from FRED's public
  fredgraph.csv endpoint -- free, no API key.

Both are end-of-day series. FRED yield series use "." for market holidays;
those rows are dropped so each series keeps its own trading calendar.
"""

import datetime as dt
from io import StringIO

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fed-sentiment-research/0.1"
}

YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _to_epoch(d):
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def fetch_yahoo_daily(symbol, start, end):
    """Daily adjusted closes from Yahoo Finance.

    Returns DataFrame(date, close) sorted by date, where close is the
    dividend/split-adjusted close.
    """
    url = YAHOO_URL.format(symbol=symbol, p1=_to_epoch(start), p2=_to_epoch(end))
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"unexpected Yahoo response for {symbol!r}: {payload!r}"[:300])
    result = result[0]
    timestamps = result.get("timestamp", [])
    adj = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    if adj is None:  # fall back to unadjusted close
        adj = result["indicators"]["quote"][0]["close"]
    dates = [
        pd.Timestamp(
            dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
        )
        for ts in timestamps
    ]
    out = pd.DataFrame({"date": dates, "close": pd.to_numeric(adj, errors="coerce")})
    out = out.dropna().drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def fetch_fred_series(series_id):
    """Full history of a FRED series. Returns DataFrame(date, value).

    Handles both the legacy 'DATE' and current 'observation_date' header.
    Missing observations ('.') are dropped.
    """
    url = FRED_URL.format(series_id=series_id)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] != 2:
        raise RuntimeError(
            f"unexpected FRED response for {series_id!r}: {resp.text[:200]!r}"
        )
    date_col, value_col = df.columns[0], df.columns[1]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out.dropna().drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def today():
    return dt.date.today()
