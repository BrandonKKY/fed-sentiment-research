"""Scraper parsing tests. Synthetic HTML snippets + a saved real fixture;
tests never touch the network."""

import datetime as dt

from conftest import FIXTURES

from fedsent.scrape import parse_last_update, parse_meeting_links

CALENDAR_STYLE = """
<strong>Statement:</strong><br>
<a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF</a> |
<a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a><br>
<strong>Minutes:</strong><br>
<a href="/monetarypolicy/files/fomcminutes20260128.pdf">PDF</a> |
<a href="/monetarypolicy/fomcminutes20260128.htm">HTML</a>
<br> (Released February 18, 2026)
"""

HISTORICAL_STYLE = """
<p><a href="/newsevents/press/monetary/20080122b.htm">Statement</a></p>
<p>Minutes: (Released February 18, 2015):
<a href="/monetarypolicy/fomcminutes20150128.htm">HTML</a> |
<a href="/monetarypolicy/files/fomcminutes20150128.pdf">PDF</a></p>
"""


def test_calendar_style_links_and_release_date():
    statements, minutes = parse_meeting_links(CALENDAR_STYLE)
    assert statements == {
        "20260128": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260128a.htm"
    }
    url, released = minutes["20260128"]
    assert url.endswith("/monetarypolicy/fomcminutes20260128.htm")
    assert released == dt.date(2026, 2, 18)


def test_implementation_note_a1_is_not_a_statement():
    statements, _ = parse_meeting_links(
        '<a href="/newsevents/pressreleases/monetary20260128a1.htm">Implementation Note</a>'
    )
    assert statements == {}


def test_historical_style_old_statement_and_release_before_link():
    statements, minutes = parse_meeting_links(HISTORICAL_STYLE)
    assert "20080122" in statements  # old /newsevents/press/monetary/ pattern
    _, released = minutes["20150128"]
    assert released == dt.date(2015, 2, 18)  # "(Released ...)" precedes the link


def test_missing_release_date_is_none():
    _, minutes = parse_meeting_links(
        '<a href="/monetarypolicy/fomcminutes20190130.htm">HTML</a>'
    )
    assert minutes["20190130"][1] is None


def test_parse_last_update_real_minutes_fixture():
    html = (FIXTURES / "minutes_20240612.html").read_text(encoding="utf-8")
    # June 2024 minutes were publicly released on 3 July 2024
    assert parse_last_update(html) == dt.date(2024, 7, 3)


def test_parse_last_update_absent():
    assert parse_last_update("<html><body>nothing here</body></html>") is None
