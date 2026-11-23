# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Parsing of a generated answer into a bit string of its token pair."""

from types import SimpleNamespace

import pytest

from audit_llm.Bits_Generation.parsing_bits_tools import (
    answer_to_bit_string,
    bits_token_pair_to_scrapper,
    token_pair_string_to_list,
    token_pair_to_string,
    validate_seq,
)


@pytest.mark.parametrize(
    "answer, items, expected",
    [
        # longest match first: "Orange" before "O"; the index is the position of the item in the pair
        ("OrangeOrangeOOrangeOrangeOOOOrange", ["O", "Orange"], "110110001"),
        ("OrangeOrangeOOrangeOrangeOOOOrange", ["Orange", "O"], "001001110"),
        # case-sensitive
        ("Hat hat HAT", ["Hat", "hat"], "01"),
        ("CASE case", ["CASE"], "0"),
        # overlaps: left to right, longest item at each position
        ("thatt", ["that", "t"], "01"),
        ("banana", ["ana", "ban"], "10"),
        ("ABBAAABBABAABBABABABAABABAB", ["ABA", "BAB"], "11001"),
        ("aaaaa", ["aa"], "00"),
        # symbols outside the pair are skipped
        ("I picked 01", ["0", "1"], "01"),
        ("23145020", ["0", "1"], "100"),
        ("I took a car", ["hat", "boat"], ""),
        # empty inputs
        ("", ["A", "B"], ""),
        ("ABC", [], ""),
    ],
)
def test_answer_to_bit_string(answer, items, expected):
    assert answer_to_bit_string(answer, items) == expected


def test_scrapper_applies_the_minimum_length():
    scrapper = bits_token_pair_to_scrapper(min_seq_length=4)
    assert scrapper("0101 and 1", "0-1") == "01011"
    assert scrapper("010", "0-1") == ""
    assert scrapper("ali yg ali ali yg", "ali-yg") == "01001"
    assert scrapper("ali yg ali", "ali-yg") == ""


def test_validate_seq_counts_the_symbols_of_the_pair():
    output = SimpleNamespace(outputs=[SimpleNamespace(text="0110" * 5)])
    generation = SimpleNamespace(token_pair="0-1")
    assert validate_seq(generation, output, min_seq_length=20)
    assert not validate_seq(generation, output, min_seq_length=21)
    other = SimpleNamespace(token_pair="a-b")
    assert not validate_seq(other, output, min_seq_length=1)


def test_token_pair_string_round_trip():
    assert token_pair_to_string(["ali", "yg"]) == "ali-yg"
    assert token_pair_string_to_list("ali-yg") == ["ali", "yg"]
