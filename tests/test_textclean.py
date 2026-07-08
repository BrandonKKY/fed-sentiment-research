"""HTML extraction and sentence splitting, tested on a real page fixture
(no network access in tests)."""

from conftest import FIXTURES

from fedsent.textclean import html_to_text, split_sentences


def test_statement_fixture_extraction():
    html = (FIXTURES / "statement_20220615.html").read_text(encoding="utf-8")
    title, text = html_to_text(html)
    assert "FOMC statement" in title
    assert "inflation" in text.lower()
    assert "target range" in text.lower()
    # boilerplate must be gone
    assert "For immediate release" not in text
    assert "Last Update" not in text
    # a June 2022 statement is a normal-length modern statement
    assert 2000 < len(text) < 8000


def test_minutes_fixture_extraction():
    html = (FIXTURES / "minutes_20240612.html").read_text(encoding="utf-8")
    title, text = html_to_text(html)
    assert "federal open market committee" in text.lower()
    assert len(text) > 20000


def test_split_sentences_protects_abbreviations():
    sents = split_sentences("The U.S. economy grew solidly. Inflation rose somewhat.")
    assert len(sents) == 2
    assert "U.S. economy" in sents[0]


def test_split_sentences_drops_fragments():
    assert split_sentences("Overview. The Committee met today in Washington.") == [
        "The Committee met today in Washington."
    ]
