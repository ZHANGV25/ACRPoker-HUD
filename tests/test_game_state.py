"""Tests for game_state.py — position inference and serialization."""

import json
import pytest
from src.game_state import GameState, PlayerState


def _make_state(dealer_seat, active_seats):
    """Helper: create a GameState with given dealer and active seat numbers."""
    state = GameState(dealer_seat=dealer_seat)
    for s in active_seats:
        state.players.append(PlayerState(seat=s, stack_bb=100.0))
    return state


class TestInferStreet:
    def test_preflop(self):
        s = GameState(board=[])
        s.infer_street()
        assert s.street == "preflop"

    def test_flop(self):
        s = GameState(board=["2h", "9c", "3s"])
        s.infer_street()
        assert s.street == "flop"

    def test_turn(self):
        s = GameState(board=["2h", "9c", "3s", "Kd"])
        s.infer_street()
        assert s.street == "turn"

    def test_river(self):
        s = GameState(board=["2h", "9c", "3s", "Kd", "Ah"])
        s.infer_street()
        assert s.street == "river"


class TestInferPositions:
    def test_6max_full(self):
        state = _make_state(dealer_seat=3, active_seats=[1, 2, 3, 4, 5, 6])
        pos = state.infer_positions()
        assert pos[3] == "BTN"
        assert pos[4] == "SB"
        assert pos[5] == "BB"
        assert pos[6] == "UTG"
        assert pos[1] == "MP"
        assert pos[2] == "CO"

    def test_5_players(self):
        state = _make_state(dealer_seat=1, active_seats=[1, 2, 3, 4, 5])
        pos = state.infer_positions()
        assert pos[1] == "BTN"
        assert pos[2] == "SB"
        assert pos[3] == "BB"
        assert pos[4] == "UTG"
        assert pos[5] == "CO"

    def test_3_players(self):
        state = _make_state(dealer_seat=2, active_seats=[2, 4, 6])
        pos = state.infer_positions()
        assert pos[2] == "BTN"
        assert pos[4] == "SB"
        assert pos[6] == "BB"

    def test_heads_up(self):
        state = _make_state(dealer_seat=1, active_seats=[1, 5])
        pos = state.infer_positions()
        assert pos[1] == "BTN"
        assert pos[5] == "BB"

    def test_no_dealer(self):
        state = _make_state(dealer_seat=None, active_seats=[1, 2, 3])
        assert state.infer_positions() == {}

    def test_sitting_out_excluded(self):
        state = GameState(dealer_seat=1)
        state.players.append(PlayerState(seat=1, stack_bb=100.0))
        state.players.append(PlayerState(seat=2, stack_bb=100.0, is_sitting_out=True))
        state.players.append(PlayerState(seat=3, stack_bb=100.0))
        pos = state.infer_positions()
        assert 2 not in pos
        assert pos[1] == "BTN"
        assert pos[3] == "BB"


class TestSerialization:
    def test_to_json_roundtrip(self):
        state = GameState(
            hand_id="123",
            hero_cards=["Ah", "Kd"],
            board=["Qh", "Jh", "Th"],
            total_bb=15.0,
            street="flop",
            dealer_seat=3,
        )
        state.players.append(PlayerState(seat=1, stack_bb=100.0, is_hero=True))
        j = state.to_json()
        data = json.loads(j)
        assert data["hand_id"] == "123"
        assert data["hero_cards"] == ["Ah", "Kd"]
        assert data["board"] == ["Qh", "Jh", "Th"]
        assert data["street"] == "flop"
        assert data["dealer_seat"] == 3

    def test_to_solver_input_skips_sitting_out(self):
        state = GameState()
        state.players.append(PlayerState(seat=1, stack_bb=100.0))
        state.players.append(PlayerState(seat=2, is_sitting_out=True))
        data = state.to_solver_input()
        assert len(data["players"]) == 1
        assert data["players"][0]["seat"] == 1
