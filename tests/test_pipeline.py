"""Integration tests for the full OCR pipeline.

These tests require macOS with Vision framework (pyobjc) installed.
They use reference screenshots to verify the full pipeline output.
Mark with @pytest.mark.macos to allow skipping on non-macOS systems.
"""

import os
import sys
import pytest
import cv2

# Skip entire module if not on macOS or Vision not available
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Pipeline tests require macOS Vision framework"
)

try:
    import Vision  # noqa: F401
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

if not HAS_VISION:
    pytest.skip("Vision framework not available", allow_module_level=True)

from src.pipeline import process_screenshot
from src.game_state import GameState

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_screenshots")


def _ref(name):
    path = os.path.join(REF_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Reference screenshot not found: {name}")
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Could not load: {name}")
    return img


class TestFullPipeline:
    def test_returns_game_state(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        assert isinstance(state, GameState)

    def test_has_6_players(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        assert len(state.players) == 6

    def test_hero_is_seat_5(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        hero = [p for p in state.players if p.is_hero]
        assert len(hero) == 1
        assert hero[0].seat == 5

    def test_preflop_street(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        assert state.street == "preflop"

    def test_flop_street(self):
        img = _ref("03_flop_hero_check.png")
        state = process_screenshot(img)
        assert state.street == "flop"
        assert len(state.board) == 3

    def test_hero_cards_detected(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        assert len(state.hero_cards) >= 1, "Hero cards not detected"

    def test_dealer_button_found(self):
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        assert state.dealer_seat is not None
        assert 1 <= state.dealer_seat <= 6

    def test_pot_detected(self):
        """On a flop screenshot, pot should be non-zero."""
        img = _ref("03_flop_hero_check.png")
        state = process_screenshot(img)
        # At least one of total_bb or pot_bb should have a value
        assert state.total_bb is not None or state.pot_bb is not None

    def test_action_buttons_on_hero_turn(self):
        """Screenshot 02 is hero's turn — should detect action buttons."""
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        # Should have at least one action available
        action_keys = {"fold", "check", "call", "raise_to", "bet"}
        found = action_keys & set(state.available_actions.keys())
        assert len(found) > 0, f"No actions found. Keys: {state.available_actions.keys()}"

    def test_to_json_produces_valid_output(self):
        import json
        img = _ref("02_preflop_hero_turn.png")
        state = process_screenshot(img)
        j = state.to_json()
        data = json.loads(j)
        assert "hero_cards" in data
        assert "board" in data
        assert "players" in data
        assert "street" in data

    def test_flop_facing_bet(self):
        """Screenshot 04 — facing a bet on the flop."""
        img = _ref("04_flop_facing_bet.png")
        state = process_screenshot(img)
        assert state.street == "flop"
        assert len(state.board) == 3
