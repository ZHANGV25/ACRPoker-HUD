"""Tests for player stats tracking and archetype classification."""

import os
import tempfile
import pytest
from solver.hh_parser import parse_hand
from solver.player_stats import StatsDB, PlayerHUDStats, classify_archetype


SAMPLE_HAND_1 = """Hand #100001 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 05:00:00 UTC
Test 6-max Seat #1 is the button
Seat 1: PlayerA ($2.00)
Seat 2: PlayerB ($2.00)
Seat 3: PlayerC ($2.00)
Seat 4: PlayerD ($2.00)
Seat 5: PlayerE ($2.00)
Seat 6: Hero ($2.00)
PlayerB posts the small blind $0.01
PlayerC posts the big blind $0.02
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
PlayerD raises $0.06 to $0.06
PlayerE folds
Hero raises $0.18 to $0.18
PlayerA folds
PlayerB folds
PlayerC folds
PlayerD calls $0.12
*** FLOP *** [Qs Jh 4c]
Main pot $0.39 | Rake $0.00
PlayerD checks
Hero bets $0.20
PlayerD calls $0.20
*** TURN *** [Qs Jh 4c] [Td]
Main pot $0.79 | Rake $0.00
PlayerD checks
Hero bets $0.40
PlayerD folds
Uncalled bet ($0.40) returned to Hero
Hero does not show
*** SUMMARY ***
Total pot $0.79
Board [Qs Jh 4c Td]
Seat 4: PlayerD folded on the Turn"""


SAMPLE_HAND_2 = """Hand #100002 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 05:05:00 UTC
Test 6-max Seat #2 is the button
Seat 1: PlayerA ($2.00)
Seat 2: PlayerB ($2.00)
Seat 3: PlayerC ($2.00)
Seat 4: PlayerD ($2.00)
Seat 5: PlayerE ($2.00)
Seat 6: Hero ($2.00)
PlayerC posts the small blind $0.01
PlayerD posts the big blind $0.02
*** HOLE CARDS ***
Dealt to Hero [7s 2d]
PlayerE folds
Hero folds
PlayerA folds
PlayerB raises $0.06 to $0.06
PlayerC calls $0.05
PlayerD calls $0.04
*** FLOP *** [Kh 8d 3s]
Main pot $0.18 | Rake $0.00
PlayerC checks
PlayerD checks
PlayerB bets $0.10
PlayerC folds
PlayerD calls $0.10
*** TURN *** [Kh 8d 3s] [5c]
Main pot $0.38 | Rake $0.00
PlayerD checks
PlayerB checks
*** RIVER *** [Kh 8d 3s 5c] [Ac]
Main pot $0.38 | Rake $0.00
PlayerD bets $0.20
PlayerB calls $0.20
*** SHOW DOWN ***
Main pot $0.78 | Rake $0.00
PlayerD shows [Ad 8c] (two pair, Aces and Eights [Ad Ac 8d 8c Kh])
PlayerB shows [Kd Qs] (a pair of Kings [Kd Kh Qs Ac 8d])
PlayerD collected $0.78 from main pot
*** SUMMARY ***
Total pot $0.78
Board [Kh 8d 3s 5c Ac]"""


SAMPLE_HAND_3 = """Hand #100003 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 05:10:00 UTC
Test 6-max Seat #3 is the button
Seat 1: PlayerA ($2.00)
Seat 2: PlayerB ($2.00)
Seat 3: PlayerC ($2.00)
Seat 4: PlayerD ($2.00)
Seat 5: PlayerE ($2.00)
Seat 6: Hero ($2.00)
PlayerD posts the small blind $0.01
PlayerE posts the big blind $0.02
*** HOLE CARDS ***
Dealt to Hero [Ts 9s]
Hero calls $0.02
PlayerA folds
PlayerB folds
PlayerC raises $0.08 to $0.08
PlayerD folds
PlayerE folds
Hero calls $0.06
*** FLOP *** [Jh 8d 2c]
Main pot $0.19 | Rake $0.00
Hero checks
PlayerC bets $0.10
Hero folds
Uncalled bet ($0.10) returned to PlayerC
PlayerC does not show
*** SUMMARY ***
Total pot $0.19
Board [Jh 8d 2c]"""


@pytest.fixture
def db():
    """Create a temporary stats database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = StatsDB(path)
    yield db
    db.close()
    os.unlink(path)


class TestStatsDB:
    def test_record_and_query(self, db):
        h = parse_hand(SAMPLE_HAND_1.strip().split("\n"))
        db.record_hand(h)
        assert db.has_hand("100001")

        # PlayerD: raised preflop (VPIP=1, PFR=1), saw flop, no showdown
        stats = db.get_stats("PlayerD")
        assert stats.hands == 1
        assert stats.vpip == 100.0
        assert stats.pfr == 100.0

        # Hero: 3bet (VPIP=1, PFR=1, 3bet=1), saw flop
        stats = db.get_stats("Hero")
        assert stats.vpip == 100.0
        assert stats.pfr == 100.0
        assert stats.three_bet == 100.0

    def test_no_duplicate(self, db):
        h = parse_hand(SAMPLE_HAND_1.strip().split("\n"))
        db.record_hand(h)
        db.record_hand(h)
        stats = db.get_stats("PlayerD")
        assert stats.hands == 1

    def test_multiple_hands(self, db):
        h1 = parse_hand(SAMPLE_HAND_1.strip().split("\n"))
        h2 = parse_hand(SAMPLE_HAND_2.strip().split("\n"))
        h3 = parse_hand(SAMPLE_HAND_3.strip().split("\n"))
        db.record_hand(h1)
        db.record_hand(h2)
        db.record_hand(h3)

        # PlayerD: hand1 raised, hand2 called, hand3 no vpip action
        stats = db.get_stats("PlayerD")
        assert stats.hands == 3
        # VPIP: hand1 (raise) + hand2 (call) = 2/3 = 66.7%
        assert 60 < stats.vpip < 70

        # Hero: hand1 raised(3bet), hand2 fold, hand3 called
        stats = db.get_stats("Hero")
        assert stats.hands == 3
        # VPIP: hand1 (raise) + hand3 (call) = 2/3 = 66.7%
        assert 60 < stats.vpip < 70

    def test_showdown_stats(self, db):
        h = parse_hand(SAMPLE_HAND_2.strip().split("\n"))
        db.record_hand(h)
        stats = db.get_stats("PlayerD")
        assert stats.hands == 1

    def test_fold_to_cbet(self, db):
        h = parse_hand(SAMPLE_HAND_3.strip().split("\n"))
        db.record_hand(h)
        # Hero called preflop, faced cbet on flop, folded
        stats = db.get_stats("Hero")
        assert stats.fold_to_cbet == 100.0

    def test_get_all_stats(self, db):
        h1 = parse_hand(SAMPLE_HAND_1.strip().split("\n"))
        h2 = parse_hand(SAMPLE_HAND_2.strip().split("\n"))
        db.record_hand(h1)
        db.record_hand(h2)
        all_stats = db.get_all_stats(min_hands=1)
        assert "PlayerD" in all_stats
        assert "Hero" in all_stats

    def test_bulk_import(self, db):
        h1 = parse_hand(SAMPLE_HAND_1.strip().split("\n"))
        h2 = parse_hand(SAMPLE_HAND_2.strip().split("\n"))
        count = db.bulk_import([h1, h2])
        assert count == 2
        count = db.bulk_import([h1, h2])
        assert count == 0


class TestClassifyArchetype:
    def test_fish(self):
        assert classify_archetype(vpip=50, pfr=8, three_bet=2, n_hands=50) == "fish"

    def test_calling_station(self):
        assert classify_archetype(vpip=40, pfr=8, three_bet=2, n_hands=50) == "calling_station"

    def test_nit(self):
        assert classify_archetype(vpip=12, pfr=10, three_bet=3, n_hands=50) == "nit"

    def test_tag(self):
        assert classify_archetype(vpip=22, pfr=18, three_bet=7, n_hands=50) == "TAG"

    def test_lag(self):
        assert classify_archetype(vpip=32, pfr=26, three_bet=10, n_hands=50) == "LAG"

    def test_maniac(self):
        assert classify_archetype(vpip=60, pfr=40, three_bet=15, n_hands=50) == "maniac"

    def test_whale(self):
        assert classify_archetype(vpip=65, pfr=8, three_bet=2, n_hands=50) == "whale"

    def test_unknown_few_hands(self):
        assert classify_archetype(vpip=50, pfr=30, three_bet=10, n_hands=10) == "unknown"


class TestPlayerHUDStats:
    def test_short_label(self):
        s = PlayerHUDStats("Test", hands=50, vpip=35.5, pfr=12.3, three_bet=5.0,
                           archetype="fish")
        label = s.short_label()
        assert "36/12/5" in label
        assert "fish" in label
        assert "(50)" in label

    def test_short_label_few_hands(self):
        s = PlayerHUDStats("Test", hands=3)
        assert s.short_label() == ""
