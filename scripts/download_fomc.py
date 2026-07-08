"""Task 1: download ~10 years of FOMC statements and minutes.

Writes:
  data/raw/statements/YYYYMMDD.html   raw pages (idempotent: skipped if present)
  data/raw/minutes/YYYYMMDD.html
  data/processed/statements.csv       meeting_date, url, title, n_chars, text
  data/processed/minutes.csv          meeting_date, release_date,
                                      release_estimated, url, title, n_chars, text

Every extracted statement must have "FOMC statement" in its page title and
every minutes page "Minutes"; documents failing the check are excluded and
reported, never silently kept.
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedsent.scrape import (  # noqa: E402
    MINUTES_LAG_DAYS,
    FetchError,
    discover_documents,
    fetch,
    parse_last_update,
)
from fedsent.textclean import html_to_text  # noqa: E402

# Sample period for validation is 2015+; years 2006-2014 are downloaded as a
# pre-sample corpus used ONLY to train the PPMI-SVD word embeddings, so the
# embedding space is frozen before the evaluation sample begins (no look-ahead).
FIRST_YEAR, LAST_YEAR = 2006, 2026

RAW_STMT = ROOT / "data" / "raw" / "statements"
RAW_MIN = ROOT / "data" / "raw" / "minutes"
PROCESSED = ROOT / "data" / "processed"


def download_raw(pairs, dest_dir, session, label):
    dest_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for date, url in sorted(pairs):
        path = dest_dir / f"{date}.html"
        if not path.exists():
            try:
                path.write_text(fetch(url, session), encoding="utf-8")
            except FetchError as exc:
                print(f"  WARN could not fetch {label} {date}: {exc}")
                continue
        done += 1
        if done % 10 == 0:
            print(f"  {label}: {done}/{len(pairs)} on disk")
    print(f"  {label}: {done}/{len(pairs)} on disk (done)")


def main():
    print(f"Discovering FOMC documents {FIRST_YEAR}-{LAST_YEAR} ...")
    session = requests.Session()
    statements, minutes = discover_documents(FIRST_YEAR, LAST_YEAR, session)
    print(f"discovered {len(statements)} statements, {len(minutes)} minutes pages")

    download_raw(statements.items(), RAW_STMT, session, "statements")
    download_raw(
        [(d, url) for d, (url, _) in minutes.items()], RAW_MIN, session, "minutes"
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)

    stmt_rows, excluded = [], []
    for date, url in sorted(statements.items()):
        path = RAW_STMT / f"{date}.html"
        if not path.exists():
            continue
        title, text = html_to_text(path.read_text(encoding="utf-8"))
        if "FOMC statement" not in title:
            excluded.append((date, title))
            continue
        stmt_rows.append(
            {
                "meeting_date": dt.datetime.strptime(date, "%Y%m%d").date(),
                "url": url,
                "title": title,
                "n_chars": len(text),
                "text": text,
            }
        )
    stmt_df = pd.DataFrame(stmt_rows).sort_values("meeting_date")
    stmt_df.to_csv(PROCESSED / "statements.csv", index=False)

    min_rows, min_excluded = [], []
    release_source_counts = {"calendar": 0, "last_update": 0, "estimated": 0}
    for date, (url, release) in sorted(minutes.items()):
        path = RAW_MIN / f"{date}.html"
        if not path.exists():
            continue
        raw_html = path.read_text(encoding="utf-8")
        title, text = html_to_text(raw_html)
        # Modern minutes pages carry a generic site title ("The Fed - Monetary
        # Policy:"), so acceptance is content-based, not title-based.
        is_minutes = (
            "minutes" in title.lower()
            or "federal open market committee" in text.lower()
        )
        if not is_minutes or len(text) < 5000:
            min_excluded.append((date, title))
            continue
        meeting_date = dt.datetime.strptime(date, "%Y%m%d").date()
        # Release date priority: calendar "(Released ...)" text, then the
        # page's own "Last Update:" footer, then meeting+21d as a flagged
        # estimate. Disagreements between the first two are reported.
        last_update = parse_last_update(raw_html)
        if release is not None and last_update is not None and release != last_update:
            print(
                f"  WARN release-date mismatch for {date}: "
                f"calendar={release} last_update={last_update} (using calendar)"
            )
        if release is not None:
            release_source_counts["calendar"] += 1
        elif last_update is not None:
            release = last_update
            release_source_counts["last_update"] += 1
        else:
            release_source_counts["estimated"] += 1
        estimated = release is None
        release_date = release or meeting_date + dt.timedelta(days=MINUTES_LAG_DAYS)
        if not estimated and not (
            dt.timedelta(days=5)
            <= (release_date - meeting_date)
            <= dt.timedelta(days=60)
        ):
            print(
                f"  WARN odd minutes release lag for {date}: "
                f"meeting {meeting_date}, released {release_date}"
            )
        min_rows.append(
            {
                "meeting_date": meeting_date,
                "release_date": release_date,
                "release_estimated": estimated,
                "url": url,
                "title": title,
                "n_chars": len(text),
                "text": text,
            }
        )
    min_df = pd.DataFrame(min_rows).sort_values("meeting_date")
    min_df.to_csv(PROCESSED / "minutes.csv", index=False)

    print("\n=== Summary ===")
    print(f"statements kept: {len(stmt_df)}   excluded (title check): {len(excluded)}")
    for date, title in excluded:
        print(f"  excluded statement {date}: {title!r}")
    print(f"minutes kept:    {len(min_df)}   excluded (title check): {len(min_excluded)}")
    for date, title in min_excluded:
        print(f"  excluded minutes {date}: {title!r}")
    est = int(min_df["release_estimated"].sum()) if len(min_df) else 0
    print(f"minutes with estimated (meeting+{MINUTES_LAG_DAYS}d) release date: {est}")
    print(f"minutes release-date sources: {release_source_counts}")

    print("\nstatements / minutes per year:")
    stmt_years = stmt_df["meeting_date"].map(lambda d: d.year).value_counts()
    min_years = min_df["meeting_date"].map(lambda d: d.year).value_counts()
    for year in sorted(set(stmt_years.index) | set(min_years.index)):
        print(f"  {year}: {stmt_years.get(year, 0)} statements, "
              f"{min_years.get(year, 0)} minutes")

    sample = min_df[min_df["meeting_date"] >= dt.date(2015, 1, 1)]
    print(
        f"\n2015+ minutes (validation sample): {len(sample)}, "
        f"of which estimated release date: {int(sample['release_estimated'].sum())}"
    )

    print(
        "\nstatement length (chars): "
        f"min={stmt_df['n_chars'].min()} median={int(stmt_df['n_chars'].median())} "
        f"max={stmt_df['n_chars'].max()}"
    )
    print(
        "minutes length (chars):   "
        f"min={min_df['n_chars'].min()} median={int(min_df['n_chars'].median())} "
        f"max={min_df['n_chars'].max()}"
    )
    for _, row in stmt_df[stmt_df["n_chars"] < 1500].iterrows():
        print(f"  SHORT statement {row.meeting_date}: {row.n_chars} chars {row.title!r}")
    for _, row in min_df[min_df["n_chars"] > 150000].iterrows():
        print(f"  HUGE minutes {row.meeting_date}: {row.n_chars} chars {row.title!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
