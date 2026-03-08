"""Auto-clicker for ACR Poker — translates solver actions into mouse clicks.

Uses macOS Quartz CGEvent API to click buttons on the poker table.
Supports multi-table: each table's window bounds are passed per-click.

Button layout (from screenshots):
  Action row:   [Fold]  [Check/Call]  [Raise To / Bet]
  Presets row:   [Min] [1/2] [3/4] [Pot] [All-In]
  Raise slider + amount to the left of presets
"""

import math
import random
import sys
import time
from typing import Optional, Tuple

import Quartz
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGHIDEventTap,
)

from src.regions import (
    FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON,
    Region,
)


# ── Preset button regions (calibrated from screenshots) ──────────────────────
# The 5 preset buttons sit in a row: Min | 1/2 | 3/4 | Pot | All-In
# Located above the Raise/Bet button, right side of screen.

PRESET_MIN   = Region(0.670, 0.855, 0.045, 0.030)
PRESET_HALF  = Region(0.720, 0.855, 0.040, 0.030)
PRESET_3Q    = Region(0.763, 0.855, 0.040, 0.030)
PRESET_POT   = Region(0.808, 0.855, 0.040, 0.030)
PRESET_ALLIN = Region(0.855, 0.855, 0.055, 0.030)

# ── Humanization constants ───────────────────────────────────────────────────

# Coordinate jitter: random offset in pixels from button center
JITTER_PX = 6

# Reaction delay range (seconds) before first click
REACTION_MIN = 0.3
REACTION_MAX = 1.8

# Delay between multi-click actions (preset → confirm)
INTER_CLICK_MIN = 0.12
INTER_CLICK_MAX = 0.35

# Mouse trail: probability of doing a curved approach before clicking
TRAIL_PROBABILITY = 0.3
TRAIL_STEPS_MIN = 3
TRAIL_STEPS_MAX = 8
TRAIL_STEP_DELAY_MIN = 0.008
TRAIL_STEP_DELAY_MAX = 0.025


# ── Mouse movement helpers ───────────────────────────────────────────────────

def _jitter(x, y, px=JITTER_PX):
    # type: (float, float, int) -> Tuple[float, float]
    """Add random pixel offset to coordinates."""
    return (x + random.uniform(-px, px),
            y + random.uniform(-px, px))


def _get_mouse_pos():
    # type: () -> Tuple[float, float]
    """Get current mouse position."""
    event = Quartz.CGEventCreate(None)
    point = Quartz.CGEventGetLocation(event)
    return point.x, point.y


def _move_to(x, y):
    # type: (float, float) -> None
    """Move mouse to position."""
    point = Quartz.CGPointMake(x, y)
    move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, point, 0)
    CGEventPost(kCGHIDEventTap, move)


def _mouse_trail(target_x, target_y):
    # type: (float, float) -> None
    """Move mouse toward target in a slightly curved path."""
    start_x, start_y = _get_mouse_pos()
    steps = random.randint(TRAIL_STEPS_MIN, TRAIL_STEPS_MAX)

    # Random control point for a quadratic bezier curve (slight arc)
    mid_x = (start_x + target_x) / 2.0 + random.uniform(-40, 40)
    mid_y = (start_y + target_y) / 2.0 + random.uniform(-30, 30)

    for i in range(1, steps + 1):
        t = i / float(steps)
        # Quadratic bezier: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        inv = 1.0 - t
        bx = inv * inv * start_x + 2 * inv * t * mid_x + t * t * target_x
        by = inv * inv * start_y + 2 * inv * t * mid_y + t * t * target_y
        _move_to(bx, by)
        time.sleep(random.uniform(TRAIL_STEP_DELAY_MIN, TRAIL_STEP_DELAY_MAX))


def _region_center(region, win_x, win_y, win_w, win_h):
    # type: (Region, int, int, int, int) -> Tuple[float, float]
    """Convert a ratio-based region to screen-absolute center point."""
    cx_ratio = region.x + region.w / 2.0
    cy_ratio = region.y + region.h / 2.0
    screen_x = win_x + cx_ratio * win_w
    screen_y = win_y + cy_ratio * win_h
    return screen_x, screen_y


def _click_at(x, y):
    # type: (float, float) -> None
    """Click at screen-absolute coordinates using CGEvent."""
    # Apply jitter
    x, y = _jitter(x, y)

    # Occasionally do a curved mouse trail approach
    if random.random() < TRAIL_PROBABILITY:
        _mouse_trail(x, y)
    else:
        # Simple move
        _move_to(x, y)
        time.sleep(random.uniform(0.015, 0.035))

    # Mouse down
    point = Quartz.CGPointMake(x, y)
    down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, point, 0)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(random.uniform(0.04, 0.10))
    # Mouse up
    up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, point, 0)
    CGEventPost(kCGHIDEventTap, up)


def _click_region(region, bounds):
    # type: (Region, dict) -> None
    """Click the center of a region given window bounds."""
    x, y = _region_center(
        region, bounds["x"], bounds["y"], bounds["w"], bounds["h"])
    _click_at(x, y)


# ── Solver action parsing ────────────────────────────────────────────────────

def _parse_solver_action(action_str):
    # type: (str) -> Tuple[str, Optional[float]]
    """Parse solver action string into (action_type, bet_pct_or_none).

    Returns:
        action_type: "fold", "check", "call", "bet", "raise", "allin"
        bet_pct: pot percentage for bets (e.g. 33.0, 66.0), None for others
    """
    s = action_str.strip()
    lower = s.lower()

    if lower == "fold":
        return ("fold", None)
    if lower == "check":
        return ("check", None)
    if lower == "call":
        return ("call", None)

    # "AllIn(X)" or "AllIn"
    if lower.startswith("allin"):
        return ("allin", None)

    # "Bet 33% (X)" or "Bet 66% (12)"
    if lower.startswith("bet "):
        # Extract percentage: "Bet 33% (12)" -> 33
        try:
            pct_str = s.split("%")[0].split()[-1]
            return ("bet", float(pct_str))
        except (ValueError, IndexError):
            return ("bet", None)

    # "Raise(X)"
    if lower.startswith("raise"):
        return ("raise", None)

    return ("unknown", None)


def _pick_preset_for_bet(bet_pct):
    # type: (Optional[float]) -> Region
    """Map a solver bet percentage to the best ACR preset button.

    ACR presets: Min, 1/2 (50%), 3/4 (75%), Pot (100%), All-In

    Solver typically uses 33%, 66% bet sizes (from DEFAULT_BET_SIZES).
    Mapping:
      <=25%  -> Min
      26-55% -> 1/2  (covers 33% and 50%)
      56-85% -> 3/4  (covers 66% and 75%)
      86-120% -> Pot  (covers 100%)
      >120%  -> All-In (covers 150%+)
    """
    if bet_pct is None:
        return PRESET_3Q  # safe default

    if bet_pct <= 25:
        return PRESET_MIN
    elif bet_pct <= 55:
        return PRESET_HALF
    elif bet_pct <= 85:
        return PRESET_3Q
    elif bet_pct <= 120:
        return PRESET_POT
    else:
        return PRESET_ALLIN


# ── Main Clicker class ───────────────────────────────────────────────────────

class Clicker:
    """Executes poker actions by clicking ACR Poker buttons.

    Tracks what has been clicked to prevent double-clicking the same spot.
    Thread-safe: called from the UI worker thread.
    """

    def __init__(self, enabled=False):
        # type: (bool) -> None
        self.enabled = enabled
        self._last_click_key = None  # type: Optional[str]
        self._last_click_time = 0.0
        self.last_action = ""  # type: str  # for debug display

    def _make_key(self, hand_id, street, action_str):
        # type: (Optional[str], str, str) -> str
        """Unique key for a spot+action to prevent re-clicking."""
        return "{}|{}|{}".format(hand_id or "?", street, action_str)

    def execute_postflop(self, picked_action, bounds, game_state):
        # type: (dict, dict, ...) -> bool
        """Click the button for a postflop solver-picked action.

        Args:
            picked_action: {"action": "Bet 66% (12)", "frequency": 0.68, ...}
            bounds: window bounds dict {"x", "y", "w", "h"}
            game_state: GameState object

        Returns True if a click was performed.
        """
        if not self.enabled or not picked_action:
            return False

        action_str = picked_action["action"]
        key = self._make_key(game_state.hand_id, game_state.street, action_str)

        if key == self._last_click_key:
            return False  # already clicked this spot

        action_type, bet_pct = _parse_solver_action(action_str)

        # Remap action if the expected button doesn't exist on screen.
        # E.g. solver says "Check" but hero faces all-in (only Fold + All-In shown).
        aa = game_state.available_actions or {}
        action_type = self._remap_action(action_type, aa)

        clicked = self._do_click(action_type, bet_pct, bounds, aa)
        if clicked:
            self._last_click_key = key
            self._last_click_time = time.time()
            self.last_action = action_str
            sys.stderr.write("[clicker] Clicked: {}\n".format(action_str))

        return clicked

    def execute_preflop(self, advice, bounds, game_state):
        # type: (str, dict, ...) -> bool
        """Click the button for a preflop advice action.

        Args:
            advice: e.g. "RAISE  AKs in CO's open range" or "FOLD ..."
            bounds: window bounds dict
            game_state: GameState object

        Returns True if a click was performed.
        """
        if not self.enabled or not advice:
            return False

        action_word = advice.split()[0].upper() if advice else ""
        key = self._make_key(game_state.hand_id, "preflop", action_word)

        if key == self._last_click_key:
            return False

        if action_word == "FOLD":
            action_type = "fold"
        elif action_word == "CALL":
            action_type = "call"
        elif action_word in ("RAISE", "3-BET", "4-BET"):
            # Preflop raises use Pot preset for standard sizing
            action_type = "raise"
        else:
            return False

        # For preflop raises, use pot-size preset (standard open/3bet sizing)
        bet_pct = 100.0 if action_type == "raise" else None
        aa = game_state.available_actions or {}
        clicked = self._do_click(action_type, bet_pct, bounds, aa)
        if clicked:
            self._last_click_key = key
            self._last_click_time = time.time()
            self.last_action = action_word
            sys.stderr.write("[clicker] Clicked preflop: {}\n".format(action_word))

        return clicked

    def _remap_action(self, action_type, available_actions):
        # type: (str, dict) -> str
        """Remap solver action to match what's actually on screen.

        Handles cases like:
        - Solver says "check" but only fold + all-in available → "call"
        - Solver says "call" but only fold + check available → "check"
        - Solver says "bet"/"raise" but no raise button → "call"
        """
        has_check = available_actions.get("check")
        has_call = available_actions.get("call") is not None
        has_raise = (available_actions.get("raise_to") is not None or
                     available_actions.get("bet") is not None)
        has_fold = available_actions.get("fold")

        if action_type == "fold" and not has_fold:
            # No fold button (e.g. BB with no raise — only Check + Bet shown)
            if has_check:
                return "check"
            elif has_call:
                return "call"
        elif action_type == "check" and not has_check:
            # No check button — click call instead (facing all-in)
            if has_call or has_raise:
                return "call"
        elif action_type == "call" and not has_call:
            if has_check:
                return "check"
        elif action_type in ("bet", "raise") and not has_raise:
            # No raise button — fall back to call/check
            if has_call:
                return "call"
            elif has_check:
                return "check"

        return action_type

    def _do_click(self, action_type, bet_pct, bounds, available_actions=None):
        # type: (str, Optional[float], dict, Optional[dict]) -> bool
        """Perform the actual click(s) for an action type."""
        aa = available_actions or {}

        # Safety: don't click on minimized/tiny windows (ACR minimizes
        # tables when hero is not in action). Normal table is ~900x700+.
        if bounds.get("w", 0) < 350 or bounds.get("h", 0) < 250:
            sys.stderr.write("[clicker] Window too small ({}x{}), skipping\n".format(
                bounds.get("w", 0), bounds.get("h", 0)))
            return False

        # Human-like reaction delay (varies widely)
        time.sleep(random.uniform(REACTION_MIN, REACTION_MAX))

        if action_type == "fold":
            # Only click fold if the fold button actually exists on screen.
            # ACR sometimes shows only Check + Bet (no fold) in BB with no raise.
            if not aa.get("fold"):
                sys.stderr.write("[clicker] No fold button, skipping\n")
                return False
            _click_region(FOLD_BUTTON, bounds)
            return True

        elif action_type in ("check", "call"):
            # When facing all-in with no call/check middle button,
            # ACR puts the All-In button in the raise position (right side).
            has_middle = aa.get("check") or aa.get("call") is not None
            if not has_middle:
                # Only fold + all-in — click right button
                _click_region(RAISE_BET_BUTTON, bounds)
            else:
                _click_region(CALL_CHECK_BUTTON, bounds)
            return True

        elif action_type == "allin":
            # Click All-In preset, then click Raise/Bet button
            _click_region(PRESET_ALLIN, bounds)
            time.sleep(random.uniform(INTER_CLICK_MIN, INTER_CLICK_MAX))
            _click_region(RAISE_BET_BUTTON, bounds)
            return True

        elif action_type in ("bet", "raise"):
            if bet_pct is not None:
                # Click the sizing preset that best matches the solver %
                preset = _pick_preset_for_bet(bet_pct)
                _click_region(preset, bounds)
                time.sleep(random.uniform(INTER_CLICK_MIN, INTER_CLICK_MAX))
            # Click the Raise/Bet button to confirm
            _click_region(RAISE_BET_BUTTON, bounds)
            return True

        return False

    def reset_for_new_hand(self, hand_id):
        # type: (Optional[str]) -> None
        """Reset click tracking when a new hand starts."""
        # Only reset if hand_id actually changed
        if self._last_click_key and hand_id:
            old_hand = self._last_click_key.split("|")[0]
            if old_hand != hand_id:
                self._last_click_key = None
                self.last_action = ""

    def toggle(self):
        # type: () -> bool
        """Toggle enabled state. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled
