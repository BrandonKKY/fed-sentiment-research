"""Unit tests for the Apel & Blix Grimaldi lexicon scorer."""

from fedsent.lexicon import count_hawk_dove, net_index, score_text


def test_hawkish_pairs():
    hawk, dove = count_hawk_dove("Participants noted higher inflation and stronger growth.")
    assert (hawk, dove) == (2, 0)


def test_dovish_pairs():
    hawk, dove = count_hawk_dove("Lower inflation and weaker growth were expected this year.")
    assert (hawk, dove) == (0, 2)


def test_unemployment_sign_is_inverted():
    hawk, dove = count_hawk_dove("The unemployment rate moved higher over the year.")
    assert (hawk, dove) == (0, 1)  # rising unemployment = dovish
    hawk, dove = count_hawk_dove("Lower unemployment was reported by the committee.")
    assert (hawk, dove) == (1, 0)


def test_window_limits_matching():
    # direction word 3 tokens from the noun: counts
    hawk, dove = count_hawk_dove("Inflation has been higher recently, they said.")
    assert hawk == 1
    # direction word far beyond the window: does not count
    hawk, dove = count_hawk_dove(
        "Inflation, according to many federal reserve participants surveyed, was somewhat higher."
    )
    assert (hawk, dove) == (0, 0)


def test_oil_price_bigram_not_double_counted():
    hawk, dove = count_hawk_dove("Members discussed higher oil prices at the meeting.")
    assert (hawk, dove) == (1, 0)  # one pair via "oil price", not a second via "price"


def test_wildcard_compounds():
    hawk, dove = count_hawk_dove("Stronger inflationary pressure was evident to everyone.")
    assert hawk == 1  # inflation* matches "inflationary"


def test_neutral_sentence_scores_zero():
    hawk, dove = count_hawk_dove("The Committee seeks to foster maximum employment and price stability.")
    assert (hawk, dove) == (0, 0)


def test_net_index_formula():
    assert net_index(3, 1) == (3 - 1) / (3 + 1 + 1)
    assert net_index(0, 0) == 0.0
    assert -1 < net_index(0, 50) < 0 < net_index(50, 0) < 1


def test_fomc_extension_covers_fed_verbs():
    text = "Inflation has risen further this year."
    strict = score_text(text, "abg")
    extended = score_text(text, "abg_fomc")
    assert strict["lex_abg_hawk"] == 0            # "risen" not in the 2012 lists
    assert extended["lex_abg_fomc_hawk"] == 1     # our documented extension


def test_no_negation_handling_documented_behaviour():
    # Known limitation inherited from the original lexicon: negations are
    # not parsed, so this still counts as hawkish. The test pins the
    # behaviour so any future change is deliberate.
    hawk, dove = count_hawk_dove("Inflation is not higher than before.")
    assert hawk == 1
