"""Download FOMC statements and meeting minutes from federalreserve.gov.

Document discovery walks the Fed's FOMC calendar pages:

  * https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
      -- current year plus roughly the five preceding years
  * https://www.federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm
      -- one archive page per older year

Statement pages look like ``/newsevents/pressreleases/monetaryYYYYMMDDa.htm``
(the ``a1.htm`` implementation notes are deliberately excluded) and minutes
pages like ``/monetarypolicy/fomcminutesYYYYMMDD.htm``.

The ``(Released Month D, YYYY)`` text that follows each minutes link on the
calendar pages is the *public release date* of the minutes -- roughly three
weeks AFTER the meeting. Scoring and validation code must timestamp minutes
by release date, not meeting date, otherwise it is looking ahead.

All documents are public. Requests are rate limited and sent with an
identifying User-Agent.
"""

import datetime as dt
import re
import time

import requests

BASE = "https://www.federalreserve.gov"
CALENDAR_URL = BASE + "/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = BASE + "/monetarypolicy/fomchistorical{year}.htm"

HEADERS = {
    "User-Agent": "fed-sentiment-research/0.1 (personal research project; python-requests)"
}
REQUEST_DELAY_S = 0.5

STATEMENT_HREF = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
# Pre-2011 statements use an older URL scheme with varying letter suffixes;
# non-statement releases that also match are filtered later by page title.
STATEMENT_HREF_OLD = re.compile(r"/newsevents/press/monetary/(\d{8})[a-z]\.htm")
MINUTES_HREF = re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm")
RELEASED_TEXT = re.compile(r"\(Released\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\)")

# Minutes are normally published three weeks after the meeting. Used only as
# a flagged fallback when the calendar page does not state the release date.
MINUTES_LAG_DAYS = 21


class FetchError(RuntimeError):
    pass


def fetch(url, session=None, retries=3):
    """GET a URL politely; return response text or raise FetchError."""
    sess = session if session is not None else requests
    last_err = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
            continue
        if resp.status_code == 200:
            time.sleep(REQUEST_DELAY_S)
            return resp.text
        if resp.status_code == 404:
            raise FetchError(f"404: {url}")
        last_err = f"HTTP {resp.status_code}"
        time.sleep(2.0 * (attempt + 1))
    raise FetchError(f"failed after {retries} attempts: {url} ({last_err})")


def parse_meeting_links(html):
    """Extract statement and minutes links from a calendar/historical page.

    Works on the raw HTML string rather than the DOM so that the same code
    handles both the current calendar layout and the older historical-page
    layout. Returns:

        statements: {"YYYYMMDD": absolute_url}
        minutes:    {"YYYYMMDD": (absolute_url, release_date_or_None)}

    The minutes release date is taken from the "(Released Month D, YYYY)"
    text that the Fed prints immediately after each minutes link; we search
    a 500-character window after the link, which is well inside the current
    meeting's block and well short of the next one.
    """
    statements = {}
    minutes = {}
    for m in STATEMENT_HREF.finditer(html):
        statements.setdefault(m.group(1), BASE + m.group(0))
    for m in STATEMENT_HREF_OLD.finditer(html):
        statements.setdefault(m.group(1), BASE + m.group(0))
    for m in MINUTES_HREF.finditer(html):
        date = m.group(1)
        # The current calendar page prints "(Released ...)" just AFTER the
        # minutes link; the historical year pages print it just BEFORE.
        release = _parse_released(html[m.end(): m.end() + 500]) or _parse_released(
            html[max(0, m.start() - 300): m.start()]
        )
        if date not in minutes or (minutes[date][1] is None and release is not None):
            minutes[date] = (BASE + m.group(0), release)
    return statements, minutes


def _parse_released(window):
    rel = RELEASED_TEXT.search(window)
    if not rel:
        return None
    cleaned = re.sub(r"\s+", " ", rel.group(1)).strip()
    try:
        return dt.datetime.strptime(cleaned, "%B %d, %Y").date()
    except ValueError:
        return None


LAST_UPDATE_TEXT = re.compile(
    r"Last [Uu]pdate:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})"
)


def parse_last_update(html):
    """The 'Last Update:' footer of a Fed page.

    For minutes pages this equals the public release date (verified against
    the calendar's '(Released ...)' text where both exist). It would drift
    if the Fed later edited the page, so calendar release dates take
    priority and any disagreement is reported by the download script.
    """
    m = LAST_UPDATE_TEXT.search(html)
    if not m:
        return None
    cleaned = re.sub(r"\s+", " ", m.group(1)).strip()
    try:
        return dt.datetime.strptime(cleaned, "%B %d, %Y").date()
    except ValueError:
        return None


def discover_documents(first_year, last_year, session=None):
    """Find all statement/minutes URLs for meetings in [first_year, last_year].

    Starts from the current calendar page, then fills in any year it does not
    cover from that year's historical archive page.
    """
    html = fetch(CALENDAR_URL, session)
    statements, minutes = parse_meeting_links(html)
    covered_years = {d[:4] for d in statements}
    for year in range(first_year, last_year + 1):
        if str(year) in covered_years:
            continue
        try:
            hist = fetch(HISTORICAL_URL.format(year=year), session)
        except FetchError:
            continue  # page not published (yet) for that year
        s2, m2 = parse_meeting_links(hist)
        for k, v in s2.items():
            statements.setdefault(k, v)
        for k, v in m2.items():
            if k not in minutes or (minutes[k][1] is None and v[1] is not None):
                minutes[k] = v

    def in_range(datestr):
        return first_year <= int(datestr[:4]) <= last_year

    statements = {k: v for k, v in statements.items() if in_range(k)}
    minutes = {k: v for k, v in minutes.items() if in_range(k)}
    return statements, minutes
