"""HTML-to-text extraction and sentence splitting for Fed documents.

Deliberately dependency-light: BeautifulSoup for HTML, plain regex for
sentence splitting (no nltk download). Fed prose is formal and regular, so
a rule-based splitter with an abbreviation guard is adequate; residual
boundary errors only blur sentence windows slightly and affect both scoring
methods equally.
"""

import re

from bs4 import BeautifulSoup

# Paragraphs matching these prefixes are page furniture, not statement text.
# The voting paragraph ("Voting for the monetary policy action were ...") is
# intentionally KEPT: dissents are genuine policy signal.
BOILERPLATE = re.compile(
    r"^(For immediate release|For release at|Media Contact|For media inquiries|"
    r"Last [Uu]pdate:|Implementation Note issued|Accessible [Vv]ersion|"
    r"Return to top|Watch [Ll]ive|Share\b)"
)

_ABBREVIATIONS = [
    "U.S.", "U.K.", "U.N.", "Mr.", "Ms.", "Mrs.", "Dr.", "Jr.", "St.",
    "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "No.", "Gov.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.",
]


def html_to_text(html):
    """Extract (title, body_text) from a Fed statement or minutes page.

    Content lives in ``div#article`` on both current and migrated legacy
    pages; paragraphs are joined with blank lines.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    node = (
        soup.find(id="article")
        or soup.find("div", attrs={"class": "col-xs-12 col-sm-8 col-md-8"})
        or soup.body
    )
    if node is None:
        return title, ""
    parts = []
    for p in node.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if not text or BOILERPLATE.match(text):
            continue
        parts.append(text)
    return title, "\n\n".join(parts)


def split_sentences(text):
    """Split text into sentences; returns sentences with >= 3 words.

    Periods inside known abbreviations are masked before splitting so that
    e.g. "U.S. economy" does not produce a boundary.
    """
    sentences = []
    for paragraph in text.split("\n\n"):
        protected = paragraph
        for ab in _ABBREVIATIONS:
            protected = protected.replace(ab, ab.replace(".", "\x00"))
        for piece in re.split(r"(?<=[.!?])\s+", protected):
            piece = piece.replace("\x00", ".").strip()
            if len(piece.split()) >= 3:
                sentences.append(piece)
    return sentences
