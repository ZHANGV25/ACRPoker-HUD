"""Tests for solver/action_history.py"""

import pytest
from src.game_state import GameState, PlayerState
from solver.range_lookup import RangeLookup
from solver.action_history import (
    reconstruct_preflop,
    determine_solver_inputs,
    HandTracker,
    PreflopAction,
)


def _make_game_state(
    board=None, dealer_seat=1, street="preflop",
    player_overrides=None, hero_cards=None,
    total_bb=None, pot_bb=None, hand_id=None,
):
    """Helper to build a GameState for testing."""
    players = []
    for seat in range(1, 7):
        p = PlayerState(seat=seat, stack_bb=100.0)
        if seat == 5:
            p.is_hero = True
        if player_overrides and seat in player_overrides:
            for k, v in player_overrides[seat].items():
                setattr(p, k, v)
        players.append(p)

    gs = GameState(
        hand_id=hand_id,
        board=board or [],
        dealer_seat=dealer_seat,
        street=street,
        players=players,
        hero_cards=hero_cards or [],
        total_bb=total_bb,
        pot_bb=pot_bb,
    )
    return gs


@pytest.fixture
def rl():
    return RangeLookup()


class TestReconstructPreflop:
    def test_simple_rfi_and_call(self):
        """UTG raises, BB calls, everyone else folds."""
        gs = _make_game_state(dealer_seat=1, player_overrides={
            # seat 1=BTN, 2=SB, 3=BB, 4=UTG, 5=MP, 6=CO
            4: {"action_label": "R", "current_bet_bb": 2.5},
            5: {"action_label": "F", "is_folded": True},
            6: {"action_label": "F", "is_folded": True},
            1: {"action_label": "F", "is_folded": True},
            2: {"action_label": "F", "is_folded": True},
            3: {"action_label": "C", "current_bet_bb": 2.5},
        })
        result = reconstruct_preflop(gs)
        assert result.opener == "UTG"
        assert result.scenario_type == "rfi"
        assert "BB" in result.callers

    def test_3bet_pot(self):
        """CO opens, BTN 3bets, CO calls."""
        gs = _make_game_state(dealer_seat=1, player_overrides={
            # 1=BTN, 2=SB, 3=BB, 4=UTG, 5=MP, 6=CO
            6: {"action_label": "R", "current_bet_bb": 2.5},
            1: {"action_label": "R", "current_bet_bb": 8.0},
            2: {"action_label": "F", "is_folded": True},
            3: {"action_label": "F", "is_folded": True},
            4: {"action_label": "F", "is_folded": True},
            5: {"action_label": "F", "is_folded": True},
        })
        result = reconstruct_preflop(gs)
        assert result.opener == "CO"
        assert result.three_bettor == "BTN"
        assert result.scenario_type == "3bet"

    def test_all_fold_no_action(self):
        """No action labels — returns empty."""
        gs = _make_game_state(dealer_seat=1)
        result = reconstruct_preflop(gs)
        assert result.opener is None

    def test_limped_pot(self):
        """Multiple callers, no raise."""
        gs = _make_game_state(dealer_seat=1, player_overrides={
            2: {"action_label": "C"},
            3: {"action_label": "C"},
            4: {"action_label": "F", "is_folded": True},
            5: {"action_label": "F", "is_folded": True},
            6: {"action_label": "F", "is_folded": True},
            1: {"action_label": "F", "is_folded": True},
        })
        result = reconstruct_preflop(gs)
        assert result.is_limped


class TestReconstructFromPostflop:
    def test_two_players_on_flop(self):
        """BTN and BB on flop — infer BTN opened, BB called."""
        gs = _make_game_state(
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=6.5,
            player_overrides={
                1: {"action_label": "C/X"},  # BTN
                3: {"action_label": "C/X"},  # BB
                2: {"is_folded": True},
                4: {"is_folded": True},
                5: {"is_folded": True},
                6: {"is_folded": True},
            },
        )
        result = reconstruct_preflop(gs)
        assert result.opener == "BTN"
        assert "BB" in result.callers

    def test_big_pot_implies_3bet(self):
        """Two players, large pot on flop -> likely 3bet pot."""
        gs = _make_game_state(
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=22.0,
            player_overrides={
                6: {"action_label": "C/X"},  # CO
                1: {"action_label": "C/X"},  # BTN
                2: {"is_folded": True},
                3: {"is_folded": True},
                4: {"is_folded": True},
                5: {"is_folded": True},
            },
        )
        result = reconstruct_preflop(gs)
        assert result.scenario_type == "3bet"


class TestDetermineSolverInputs:
    def test_basic_flop_hu(self, rl):
        """BTN opens, BB calls, flop dealt."""
        gs = _make_game_state(
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=6.5,
            hero_cards=["Ac", "Qs"],
            player_overrides={
                1: {"action_label": "C/X", "stack_bb": 97.0},  # BTN
                3: {"action_label": "C/X", "stack_bb": 97.0},  # BB
                2: {"is_folded": True},
                4: {"is_folded": True},
                5: {"is_folded": True, "stack_bb": 100.0},
                6: {"is_folded": True},
            },
        )
        result = determine_solver_inputs(gs, rl)
        assert result is not None
        assert result["board"] == ["Ah", "Kd", "7c"]
        assert result["oop_position"] == "BB"
        assert result["ip_position"] == "BTN"
        assert result["effective_stack"] == 97.0
        assert len(result["oop_range"]) > 0
        assert len(result["ip_range"]) > 0

    def test_returns_none_for_preflop(self, rl):
        """No board = preflop, solver needs postflop."""
        gs = _make_game_state(board=[], street="preflop")
        assert determine_solver_inputs(gs, rl) is None

    def test_hero_position_set(self, rl):
        """Hero's IP/OOP is correctly identified."""
        # Hero is seat 5 = MP (dealer=1), in pot with BB (seat 3)
        gs = _make_game_state(
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=6.5,
            hero_cards=["Jh", "Jd"],
            player_overrides={
                5: {"action_label": "R", "stack_bb": 95.0, "current_bet_bb": 2.5},
                3: {"action_label": "C", "stack_bb": 97.0},
                1: {"is_folded": True},
                2: {"is_folded": True},
                4: {"is_folded": True},
                6: {"is_folded": True},
            },
        )
        result = determine_solver_inputs(gs, rl)
        assert result is not None
        # MP is later than BB in postflop order, so MP is IP
        assert result["hero_position"] == "ip"

    def test_multiway_picks_largest_villain(self, rl):
        """3 players on flop: hero + 2 villains, picks biggest stack."""
        gs = _make_game_state(
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=9.0,
            hero_cards=["Qh", "Qd"],
            player_overrides={
                5: {"stack_bb": 95.0},  # MP, hero
                3: {"stack_bb": 90.0},  # BB
                6: {"stack_bb": 50.0},  # CO
                1: {"is_folded": True},
                2: {"is_folded": True},
                4: {"is_folded": True},
            },
        )
        result = determine_solver_inputs(gs, rl)
        assert result is not None
        # Should pick BB (90bb) as villain, not CO (50bb)


class TestHandTracker:
    def test_tracks_preflop_through_streets(self, rl):
        """Tracker remembers preflop action when board comes."""
        tracker = HandTracker()

        # Preflop snapshot
        gs_pre = _make_game_state(
            hand_id="hand1",
            dealer_seat=1,
            street="preflop",
            player_overrides={
                1: {"action_label": "R", "current_bet_bb": 2.5},  # BTN opens
                2: {"action_label": "F", "is_folded": True},
                3: {"action_label": "C"},  # BB calls
                4: {"action_label": "F", "is_folded": True},
                5: {"action_label": "F", "is_folded": True},
                6: {"action_label": "F", "is_folded": True},
            },
        )
        tracker.update(gs_pre)
        assert tracker.preflop_action is not None
        assert tracker.preflop_action.opener == "BTN"

        # Flop snapshot — action labels now reflect flop play
        gs_flop = _make_game_state(
            hand_id="hand1",
            dealer_seat=1,
            board=["Ah", "Kd", "7c"],
            street="flop",
            total_bb=6.5,
            hero_cards=["Qh", "Qd"],
            player_overrides={
                1: {"action_label": "C/X", "stack_bb": 97.0},
                3: {"action_label": "R/B", "stack_bb": 97.0, "current_bet_bb": 3.0},
                2: {"is_folded": True},
                4: {"is_folded": True},
                5: {"is_folded": True},
                6: {"is_folded": True},
            },
        )
        result = tracker.get_solver_inputs(gs_flop, rl)
        assert result is not None
        # Preflop action should still show BTN opened
        assert tracker.preflop_action.opener == "BTN"

    def test_resets_on_new_hand(self, rl):
        tracker = HandTracker()

        gs1 = _make_game_state(hand_id="hand1", dealer_seat=1, player_overrides={
            1: {"action_label": "R", "current_bet_bb": 2.5},
            3: {"action_label": "C"},
            2: {"action_label": "F", "is_folded": True},
            4: {"action_label": "F", "is_folded": True},
            5: {"action_label": "F", "is_folded": True},
            6: {"action_label": "F", "is_folded": True},
        })
        tracker.update(gs1)
        assert tracker.preflop_action.opener == "BTN"

        # dealer=6: seat1=SB, seat2=BB, seat3=UTG, seat4=MP, seat5=CO, seat6=BTN
        gs2 = _make_game_state(hand_id="hand2", dealer_seat=6, player_overrides={
            4: {"action_label": "R", "current_bet_bb": 2.5},
        })
        tracker.update(gs2)
        assert tracker.current_hand_id == "hand2"
        assert tracker.preflop_action.opener == "MP"
