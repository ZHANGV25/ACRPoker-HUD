"""Tests for solver/range_lookup.py"""

import pytest
from solver.range_lookup import RangeLookup, ip_oop_positions


@pytest.fixture
def rl():
    return RangeLookup()


class TestRFI:
    def test_utg_rfi_exists(self, rl):
        r = rl.rfi("UTG")
        assert r is not None
        assert "AA" in r
        assert "72o" not in r

    def test_btn_rfi_wider_than_utg(self, rl):
        utg = rl.rfi("UTG").split(",")
        btn = rl.rfi("BTN").split(",")
        assert len(btn) > len(utg)

    def test_all_positions_have_rfi(self, rl):
        for pos in ["UTG", "MP", "CO", "BTN", "SB"]:
            assert rl.rfi(pos) is not None, f"{pos} missing RFI range"

    def test_rfi_contains_premiums(self, rl):
        for pos in ["UTG", "MP", "CO", "BTN", "SB"]:
            r = rl.rfi(pos)
            assert "AA" in r
            assert "KK" in r
            assert "AKs" in r


class TestVsRFI:
    def test_bb_vs_btn_call(self, rl):
        r = rl.vs_rfi("BB", "BTN", "call")
        assert r is not None
        assert len(r) > 0

    def test_bb_vs_btn_3bet(self, rl):
        r = rl.vs_rfi("BB", "BTN", "3bet")
        assert r is not None
        assert "AA" in r

    def test_bb_vs_utg_tighter_than_bb_vs_btn(self, rl):
        vs_utg = rl.vs_rfi("BB", "UTG", "call")
        vs_btn = rl.vs_rfi("BB", "BTN", "call")
        assert len(vs_btn.split(",")) > len(vs_utg.split(","))

    def test_sb_vs_utg_mostly_3bet(self, rl):
        call_range = rl.vs_rfi("SB", "UTG", "call")
        three_bet = rl.vs_rfi("SB", "UTG", "3bet")
        # SB vs UTG should have no/tiny call range, just 3bet
        assert call_range is None or len(call_range) == 0
        assert three_bet is not None

    def test_sb_vs_btn_has_call_and_3bet(self, rl):
        assert rl.vs_rfi("SB", "BTN", "call") is not None
        assert rl.vs_rfi("SB", "BTN", "3bet") is not None


class TestVs3Bet:
    def test_utg_vs_3bet_call(self, rl):
        r = rl.vs_3bet("UTG", "call")
        assert r is not None
        assert "AQs" in r

    def test_utg_vs_3bet_4bet(self, rl):
        r = rl.vs_3bet("UTG", "4bet")
        assert r is not None
        assert "AA" in r
        assert "AKs" in r

    def test_all_positions_have_vs_3bet(self, rl):
        for pos in ["UTG", "MP", "CO", "BTN", "SB"]:
            assert rl.vs_3bet(pos, "call") is not None, f"{pos} missing vs_3bet call"
            assert rl.vs_3bet(pos, "4bet") is not None, f"{pos} missing vs_3bet 4bet"


class TestGenericGet:
    def test_rfi_key(self, rl):
        assert rl.get("UTG_rfi") == rl.rfi("UTG")

    def test_vs_rfi_key(self, rl):
        assert rl.get("BB_vs_BTN_call") == rl.vs_rfi("BB", "BTN", "call")

    def test_vs_rfi_3bet_key(self, rl):
        assert rl.get("BB_vs_BTN_3bet") == rl.vs_rfi("BB", "BTN", "3bet")

    def test_unknown_key(self, rl):
        assert rl.get("UNKNOWN_KEY") is None


class TestIPOOP:
    def test_bb_vs_btn(self):
        oop, ip = ip_oop_positions("BB", "BTN")
        assert oop == "BB"
        assert ip == "BTN"

    def test_sb_vs_bb(self):
        oop, ip = ip_oop_positions("SB", "BB")
        assert oop == "SB"
        assert ip == "BB"

    def test_utg_vs_co(self):
        oop, ip = ip_oop_positions("UTG", "CO")
        assert oop == "UTG"
        assert ip == "CO"

    def test_order_doesnt_matter(self):
        oop1, ip1 = ip_oop_positions("BTN", "BB")
        oop2, ip2 = ip_oop_positions("BB", "BTN")
        assert oop1 == oop2
        assert ip1 == ip2
