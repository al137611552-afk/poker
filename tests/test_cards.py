import pytest

from holdem.cards import (
    FULL_DECK,
    card_from_str,
    card_rank,
    card_suit,
    card_to_str,
    cards_from_str,
    cards_to_str,
    make_card,
)


def test_roundtrip_all_cards():
    for card in FULL_DECK:
        assert card_from_str(card_to_str(card)) == card


def test_known_encodings():
    assert card_to_str(0) == "2c"
    assert card_to_str(51) == "As"
    assert card_from_str("As") == 51
    assert card_rank(card_from_str("Td")) == 8
    assert card_suit(card_from_str("Td")) == 1


def test_make_card_bounds():
    assert make_card(12, 3) == 51
    with pytest.raises(ValueError):
        make_card(13, 0)
    with pytest.raises(ValueError):
        make_card(0, 4)


def test_card_list_parsing():
    cards = cards_from_str("AsKd7c")
    assert cards_to_str(cards) == "AsKd7c"
    assert cards_from_str("As Kd 7c") == cards
    with pytest.raises(ValueError):
        cards_from_str("AsK")
    with pytest.raises(ValueError):
        card_from_str("as")
