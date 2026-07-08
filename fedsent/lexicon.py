"""Lexicon-based hawkish/dovish scoring of Fed communications (approach 1).

Primary source (cited): Apel, M. and M. Blix Grimaldi (2012), "The
Information Content of Central Bank Minutes", Sveriges Riksbank Working
Paper Series No. 261. Open-access copy: EconStor,
https://www.econstor.eu/handle/10419/81866 (word lists transcribed for this
project from Section 6 and footnotes 11 and 15 of that paper).

Their design: hawkishness/dovishness is counted from two-word combinations
of a direction adjective and a policy-relevant noun ("higher inflation" =
hawkish, "lower growth" = dovish), which disambiguates single words.

  Nouns (base list; * = wildcard, compounds allowed):
      inflation*, price*, wage*, oil price*, cyclical position*, growth*,
      development*
  Extended nouns (the paper's own "Net Index Extended" robustness list):
      employment*, unemployment*, recovery*, cost*
  Hawkish direction words: high(er), strong(er), fast(er), increasing,
      increased
  Dovish direction words:  low(er), weak(er), slow(er), decreasing,
      decreased

The document score is their Net Index (normalisation following Birz & Lott
2011, as adopted by A&BG):

    net = (n_hawk - n_dove) / (n_hawk + n_dove + 1)        in (-1, 1)

We use the paper's extended noun list throughout: the Fed's dual mandate
makes employment vocabulary essential for FOMC text.

Documented adaptations for English FOMC prose (A&BG scored Swedish-language
Riksbank minutes; a strict adjacent-bigram match transplants poorly):

  1. A direction word pairs with AT MOST ONE noun: the nearest noun within
     WINDOW = 3 tokens in the same sentence, either order (ties prefer the
     noun that follows, matching English adjective-noun order). So "higher
     core inflation" and "growth has slowed" both count, while in "higher
     inflation and stronger growth" each adjective attaches only to its own
     noun. WINDOW is fixed a priori and never tuned against market
     outcomes.
  2. "unemployment" has an INVERTED sign: rising unemployment signals
     economic weakness => dovish, falling unemployment => hawkish. (A&BG do
     not state a sign convention for this noun; the economically sensible
     one is used and disclosed.)
  3. "cyclical position" is the paper's English rendering of Swedish
     "konjunktur" and essentially never occurs in FOMC text; it is kept for
     fidelity and contributes ~nothing.
  4. No negation handling, exactly like the original lexicon. Known
     limitation, reported in the write-up.

A second variant ("abg_fomc") extends the direction lists with common FOMC
verbs/adjectives (rose/risen, declined, elevated, subdued, ...). This is
OUR extension, clearly labelled, evaluated separately, and never silently
mixed with the cited lexicon.
"""

import re

from .textclean import split_sentences

# --- word lists ------------------------------------------------------------

ABG_UP = frozenset(
    "high higher strong stronger fast faster increasing increased".split()
)
ABG_DOWN = frozenset(
    "low lower weak weaker slow slower decreasing decreased".split()
)

# Our documented FOMC-English extension (variant "abg_fomc" only).
FOMC_EXTRA_UP = frozenset(
    "rise rises rose risen rising elevated accelerate accelerated "
    "accelerating robust solid strengthen strengthened strengthening "
    "pickup".split()
)
FOMC_EXTRA_DOWN = frozenset(
    "decline declines declined declining fall falls fell fallen falling "
    "moderate moderated moderating ease eased easing soften softened "
    "softening subdued sluggish muted slowed slowing weaken weakened "
    "weakening diminish diminished diminishing".split()
)

# Noun stems; multi-word entries match consecutive tokens. Wildcard
# semantics: a token matches a stem if token.startswith(stem), mirroring
# the paper's "inflation*" compound matching.
ABG_NOUNS = (
    "inflation",
    "price",
    "wage",
    "oil price",
    "cyclical position",
    "growth",
    "development",
    # extended list (Net Index Extended):
    "employment",
    "unemployment",
    "recovery",
    "cost",
)

# Nouns whose direction flips the hawk/dove sign.
INVERTED_NOUNS = frozenset({"unemployment"})

WINDOW = 3  # max token distance between direction word and noun

_TOKEN_RE = re.compile(r"[a-z]+")

VARIANTS = ("abg", "abg_fomc")


def direction_sets(variant):
    if variant == "abg":
        return ABG_UP, ABG_DOWN
    if variant == "abg_fomc":
        return ABG_UP | FOMC_EXTRA_UP, ABG_DOWN | FOMC_EXTRA_DOWN
    raise ValueError(f"unknown lexicon variant {variant!r}")


def _tokens(sentence):
    return _TOKEN_RE.findall(sentence.lower())


def _noun_positions(toks):
    """Positions of lexicon nouns in a token list.

    Returns a list of (start, end, stem) spans. Multi-word nouns consume
    their tokens so e.g. the "price" inside a matched "oil price" is not
    double counted.
    """
    multi = [n.split() for n in ABG_NOUNS if " " in n]
    single = [n for n in ABG_NOUNS if " " not in n]
    consumed = set()
    spans = []
    for i, tok in enumerate(toks):
        for parts in multi:
            if (
                tok == parts[0]
                and i + 1 < len(toks)
                and toks[i + 1].startswith(parts[1])
            ):
                spans.append((i, i + 1, " ".join(parts)))
                consumed.update((i, i + 1))
    for i, tok in enumerate(toks):
        if i in consumed:
            continue
        for stem in single:
            if tok.startswith(stem):
                spans.append((i, i, stem))
                break
    return spans


def count_hawk_dove(text, variant="abg"):
    """Count hawkish and dovish direction-word/noun pairs in a document."""
    up_words, down_words = direction_sets(variant)
    hawk = dove = 0
    for sentence in split_sentences(text):
        toks = _tokens(sentence)
        spans = _noun_positions(toks)
        if not spans:
            continue
        directions = [
            (i, +1 if tok in up_words else -1)
            for i, tok in enumerate(toks)
            if tok in up_words or tok in down_words
        ]
        # Each direction word attaches to at most one noun: the nearest
        # span within WINDOW, ties broken toward the noun that follows.
        for pos, sign in directions:
            best_key, best_stem = None, None
            for start, end, stem in spans:
                dist = min(abs(pos - start), abs(pos - end))
                if dist == 0 or dist > WINDOW:
                    continue
                key = (dist, 0 if start > pos else 1)
                if best_key is None or key < best_key:
                    best_key, best_stem = key, stem
            if best_stem is not None:
                effective = -sign if best_stem in INVERTED_NOUNS else sign
                if effective > 0:
                    hawk += 1
                else:
                    dove += 1
    return hawk, dove


def net_index(hawk, dove):
    """A&BG Net Index: (hawk - dove) / (hawk + dove + 1), bounded in (-1, 1)."""
    return (hawk - dove) / (hawk + dove + 1)


def score_text(text, variant="abg"):
    hawk, dove = count_hawk_dove(text, variant)
    return {
        f"lex_{variant}_hawk": hawk,
        f"lex_{variant}_dove": dove,
        f"lex_{variant}_net": net_index(hawk, dove),
    }
