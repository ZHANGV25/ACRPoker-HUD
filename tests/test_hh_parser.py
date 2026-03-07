"""Tests for hand history parser."""

import pytest
from solver.hh_parser import parse_hand, parse_file, ParsedHand


SAMPLE_HAND_SIMPLE = """Hand #2686662183 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 04:59:57 UTC
Ideal 6-max Seat #4 is the button
Seat 1: swarna89 ($2.00)
Seat 2: Pavelik ($2.96)
Seat 3: WoWiWoN ($0.98)
Seat 4: dahui36 ($2.23)
Seat 5: victoremm ($2.01)
Seat 6: vortexted ($1.87)
victoremm posts the small blind $0.01
vortexted posts the big blind $0.02
*** HOLE CARDS ***
Dealt to vortexted [Ad Qh]
swarna89 folds
Pavelik folds
WoWiWoN folds
dahui36 raises $0.04 to $0.04
victoremm folds
vortexted raises $0.06 to $0.08
dahui36 folds
Uncalled bet ($0.04) returned to vortexted
vortexted does not show
*** SUMMARY ***
Total pot $0.09
Seat 1: swarna89 folded on the Pre-Flop and did not bet
Seat 2: Pavelik folded on the Pre-Flop and did not bet
Seat 3: WoWiWoN folded on the Pre-Flop and did not bet
Seat 4: dahui36 (button) folded on the Pre-Flop
Seat 5: victoremm (small blind) folded on the Pre-Flop
Seat 6: vortexted did not show and won $0.09"""


SAMPLE_HAND_SHOWDOWN = """Hand #2686678568 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 05:27:00 UTC
Ideal 6-max Seat #1 is the button
Seat 1: swarna89 ($2.26)
Seat 2: Pavelik ($2.54)
Seat 3: Heebert ($2.16)
Seat 4: vla01092019 ($0.74)
Seat 5: RYJITSU ($2.44)
Seat 6: vortexted ($1.92)
Pavelik posts the small blind $0.01
Heebert posts the big blind $0.02
*** HOLE CARDS ***
Dealt to vortexted [2s 7h]
vla01092019 raises $0.04 to $0.04
RYJITSU folds
vortexted folds
swarna89 folds
Pavelik folds
Heebert calls $0.02
*** FLOP *** [4s 3c 7c]
Main pot $0.12 | Rake $0.00
Heebert checks
vla01092019 bets $0.02
Heebert calls $0.02
*** TURN *** [4s 3c 7c] [Kd]
Main pot $0.16 | Rake $0.00
Heebert checks
vla01092019 checks
*** RIVER *** [4s 3c 7c Kd] [7d]
Main pot $0.16 | Rake $0.00
Heebert checks
vla01092019 checks
*** SHOW DOWN ***
Main pot $0.16 | Rake $0.00
Heebert shows [2h 2c] (two pair, Sevens and Deuces [7d 7c 2h 2c Kd])
vla01092019 shows [Ad 8s] (a pair of Sevens [7d 7c Ad Kd 8s])
Heebert collected $0.16 from main pot
*** SUMMARY ***
Total pot $0.16
Board [4s 3c 7c Kd 7d]
Seat 1: swarna89 (button) folded on the Pre-Flop
Seat 2: Pavelik (small blind) folded on the Pre-Flop
Seat 3: Heebert (big blind) showed [2h 2c] and won $0.16 with two pair, Sevens and Deuces [7d 7c 2h 2c Kd]
Seat 4: vla01092019 showed [Ad 8s] and lost with a pair of Sevens [7d 7c Ad Kd 8s]
Seat 5: RYJITSU folded on the Pre-Flop and did not bet
Seat 6: vortexted folded on the Pre-Flop and did not bet"""


SAMPLE_HAND_ALLIN = """Hand #2686670000 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 05:10:00 UTC
Ideal 6-max Seat #3 is the button
Seat 1: swarna89 ($2.00)
Seat 2: Pavelik ($2.90)
Seat 3: WoWiWoN ($1.36)
Seat 4: dahui36 ($2.19)
Seat 5: RYJITSU ($2.00)
Seat 6: vortexted ($2.09)
dahui36 posts the small blind $0.01
RYJITSU posts the big blind $0.02
*** HOLE CARDS ***
Dealt to vortexted [As Kd]
vortexted raises $0.06 to $0.06
swarna89 folds
Pavelik folds
WoWiWoN raises $0.20 to $0.20
dahui36 folds
RYJITSU folds
vortexted raises $2.03 to $2.09 and is all-in
WoWiWoN calls $1.16 and is all-in
Uncalled bet ($0.73) returned to vortexted
*** FLOP *** [Qh Jh 6d]
Main pot $2.75 | Rake $0.00
*** TURN *** [Qh Jh 6d] [Tc]
Main pot $2.75 | Rake $0.00
*** RIVER *** [Qh Jh 6d Tc] [2s]
Main pot $2.75 | Rake $0.00
*** SHOW DOWN ***
Main pot $2.75 | Rake $0.00
vortexted shows [As Kd] (a straight, Ace high [As Kd Qh Jh Tc])
WoWiWoN shows [Ah Th] (a pair of Tens [Ah Th Tc Qh Jh])
vortexted collected $2.75 from main pot
*** SUMMARY ***
Total pot $2.75
Board [Qh Jh 6d Tc 2s]
Seat 1: swarna89 folded on the Pre-Flop and did not bet
Seat 2: Pavelik folded on the Pre-Flop and did not bet
Seat 3: WoWiWoN (button) showed [Ah Th] and lost with a pair of Tens [Ah Th Tc Qh Jh]
Seat 4: dahui36 (small blind) folded on the Pre-Flop
Seat 5: RYJITSU (big blind) folded on the Pre-Flop
Seat 6: vortexted showed [As Kd] and won $2.75 with a straight, Ace high [As Kd Qh Jh Tc]"""


class TestParseHand:
    def test_simple_hand(self):
        lines = SAMPLE_HAND_SIMPLE.strip().split("\n")
        h = parse_hand(lines)
        assert h is not None
        assert h.hand_id == "2686662183"
        assert h.sb == 0.01
        assert h.bb == 0.02
        assert h.button_seat == 4
        assert h.table_name == "Ideal"
        assert len(h.seats) == 6
        assert h.seats[6] == ("vortexted", 1.87)
        assert h.hero_name == "vortexted"
        assert h.hero_cards == "Ad Qh"

    def test_simple_hand_actions(self):
        lines = SAMPLE_HAND_SIMPLE.strip().split("\n")
        h = parse_hand(lines)
        pf = h.preflop_actions()
        # swarna89 folds, Pavelik folds, WoWiWoN folds, dahui36 raises,
        # victoremm folds, vortexted raises, dahui36 folds
        assert len(pf) == 7
        assert pf[0].player == "swarna89"
        assert pf[0].action == "fold"
        assert pf[3].player == "dahui36"
        assert pf[3].action == "raise"
        assert pf[3].amount == 0.04
        assert pf[5].player == "vortexted"
        assert pf[5].action == "raise"
        assert pf[5].amount == 0.08

    def test_simple_hand_positions(self):
        lines = SAMPLE_HAND_SIMPLE.strip().split("\n")
        h = parse_hand(lines)
        assert h.player_position("dahui36") == "BTN"
        assert h.player_position("victoremm") == "SB"
        assert h.player_position("vortexted") == "BB"
        assert h.player_position("swarna89") == "UTG"

    def test_showdown_hand(self):
        lines = SAMPLE_HAND_SHOWDOWN.strip().split("\n")
        h = parse_hand(lines)
        assert h.hand_id == "2686678568"
        assert h.board == ["4s", "3c", "7c", "Kd", "7d"]
        assert "Heebert" in h.showdown_cards
        assert h.showdown_cards["Heebert"] == "2h 2c"
        assert "vla01092019" in h.showdown_cards
        assert "Heebert" in h.winners
        assert h.total_pot == 0.16

    def test_showdown_saw_flop(self):
        lines = SAMPLE_HAND_SHOWDOWN.strip().split("\n")
        h = parse_hand(lines)
        assert h.saw_flop("Heebert") is True
        assert h.saw_flop("vla01092019") is True
        assert h.saw_flop("vortexted") is False
        assert h.went_to_showdown("Heebert") is True
        assert h.went_to_showdown("vortexted") is False

    def test_allin_hand(self):
        lines = SAMPLE_HAND_ALLIN.strip().split("\n")
        h = parse_hand(lines)
        assert h.hand_id == "2686670000"
        pf = h.preflop_actions()
        # vortexted raises, swarna89 folds, Pavelik folds, WoWiWoN raises,
        # dahui36 folds, RYJITSU folds, vortexted raises, WoWiWoN calls
        assert pf[-2].player == "vortexted"
        assert pf[-2].action == "raise"
        assert pf[-1].player == "WoWiWoN"
        assert pf[-1].action == "call"

    def test_cbet_detection(self):
        lines = SAMPLE_HAND_SHOWDOWN.strip().split("\n")
        h = parse_hand(lines)
        # vla01092019 raised preflop, bet the flop = cbet
        flop_actions = [a for a in h.actions if a.street == "flop"]
        assert len(flop_actions) == 3  # check, bet, call
        assert flop_actions[0].action == "check"
        assert flop_actions[1].player == "vla01092019"
        assert flop_actions[1].action == "bet"

    def test_empty_lines(self):
        h = parse_hand([])
        assert h is None

    def test_vpip_tracking(self):
        lines = SAMPLE_HAND_SIMPLE.strip().split("\n")
        h = parse_hand(lines)
        pf = h.preflop_actions()
        # VPIP = called or raised preflop
        vpip = {a.player for a in pf if a.action in ("call", "raise")}
        assert "dahui36" in vpip
        assert "vortexted" in vpip
        assert "swarna89" not in vpip


class TestParseFile:
    def test_parse_real_file(self):
        """Test parsing actual ACR hand history file if available."""
        import os
        hh_dir = os.path.expanduser(
            "~/Downloads/AmericasCardroom/handHistory/vortexted/")
        if not os.path.isdir(hh_dir):
            pytest.skip("No hand history directory found")
        import glob
        files = glob.glob(os.path.join(hh_dir, "HH*.txt"))
        if not files:
            pytest.skip("No hand history files found")
        hands = parse_file(files[0])
        assert len(hands) > 0
        h = hands[0]
        assert h.hand_id
        assert h.bb > 0
        assert len(h.seats) >= 2
