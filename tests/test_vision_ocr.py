"""Tests for vision_ocr.py — BB amount parsing and text fixing."""

import pytest
from src.vision_ocr import parse_bb_amount, _fix_bb_text, _extract_amount


class TestFixBBText:
    def test_br_to_bb(self):
        assert _fix_bb_text("330 5 BR") == "330 5 BB"

    def test_rr_to_bb(self):
        assert _fix_bb_text("97 RR") == "97 BB"

    def test_r8_to_bb(self):
        assert _fix_bb_text("100 R8") == "100 BB"

    def test_already_bb(self):
        assert _fix_bb_text("50 BB") == "50 BB"

    def test_no_suffix(self):
        assert _fix_bb_text("123") == "123"


class TestParseBBAmount:
    def test_clean_integer(self):
        assert parse_bb_amount("97 BB") == 97.0

    def test_clean_float(self):
        assert parse_bb_amount("4.5 BB") == 4.5

    def test_space_as_decimal(self):
        assert parse_bb_amount("330 5 BB") == 330.5

    def test_just_number(self):
        assert parse_bb_amount("100") == 100.0

    def test_br_suffix(self):
        assert parse_bb_amount("87.5 BR") == 87.5

    def test_rr_suffix(self):
        assert parse_bb_amount("13 RR") == 13.0

    def test_empty(self):
        assert parse_bb_amount("") is None

    def test_garbage(self):
        assert parse_bb_amount("hello world") is None

    def test_large_number(self):
        assert parse_bb_amount("1500 BB") == 1500.0

    def test_small_decimal(self):
        assert parse_bb_amount("0.5 BB") == 0.5


class TestExtractAmount:
    def test_float_amount(self):
        assert _extract_amount("call 4.5") == 4.5

    def test_space_decimal(self):
        assert _extract_amount("raise to 9 5") == 9.5

    def test_integer_amount(self):
        assert _extract_amount("call 12") == 12.0

    def test_no_amount(self):
        assert _extract_amount("fold") is None

    def test_multiple_numbers_takes_last(self):
        # "raise to 9" -> last number is 9
        assert _extract_amount("raise to 9") == 9.0
