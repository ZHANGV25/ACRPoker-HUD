"""Live watcher — continuously captures ACR Poker and shows readable game state.

Runs OCR on every detected state change and prints a formatted summary
including board, hero cards, positions, actions, and solver inputs.

Usage:
    python -m src.watch              # live capture from ACR window
    python -m src.watch -f file.png  # single screenshot (for testing)
    python -m src.watch --all        # capture every poll, not just changes
"""

import signal
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.capture import find_acr_windows, capture_window, capture_from_file
from src.pipeline import process_screenshot
from src.regions import FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON
from src.game_state import GameState
from solver.range_lookup import RangeLookup
from solver.action_history import HandTracker, reconstruct_preflop, determine_solver_inputs

POLL_INTERVAL = 0.8
MIN_WINDOW_WIDTH = 700  # Skip tiled multi-table windows (expanded ~800pt, tiled ~600pt)
SMOOTH_WINDOW = 3       # Number of recent reads to vote on for stabilization


# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoothing — stabilizes flickering card reads
# ─────────────────────────────────────────────────────────────────────────────

class ReadingSmoother:
    """Stabilizes flickering card reads using lock-in + majority vote.

    Once a card is read the same way LOCK_THRESHOLD times, it's locked
    for the rest of the hand. Resets on new hand_id or new street (for board).
    """

    LOCK_THRESHOLD = 2  # reads before locking in

    def __init__(self, window=SMOOTH_WINDOW):
        # type: (int) -> None
        self._window = window
        self._hero_history = []    # type: List[Tuple[Optional[str], ...]]
        self._hero_locked = {}     # type: Dict[int, str]  # position -> locked card
        self._board_history = []   # type: List[Tuple[str, ...]]
        self._board_locked = {}    # type: Dict[int, str]
        self._current_hand_id = None  # type: Optional[str]
        self._current_board_len = 0

    def update(self, gs):
        # type: (GameState) -> GameState
        """Smooth hero cards and board across recent reads. Mutates gs in place."""
        # Reset on new hand
        if gs.hand_id != self._current_hand_id:
            self._hero_history.clear()
            self._hero_locked.clear()
            self._board_history.clear()
            self._board_locked.clear()
            self._current_hand_id = gs.hand_id
            self._current_board_len = 0

        # Reset board locks when new cards appear (street change)
        board_len = len(gs.board) if gs.board else 0
        if board_len > self._current_board_len:
            self._board_history.clear()
            self._board_locked.clear()
            self._current_board_len = board_len

        # Hero cards: lock-in + vote
        if gs.hero_cards:
            self._hero_history.append(tuple(gs.hero_cards))
            if len(self._hero_history) > self._window:
                self._hero_history.pop(0)
            gs.hero_cards = list(self._resolve(
                self._hero_history, len(gs.hero_cards), self._hero_locked))

        # Board cards: lock-in + vote
        if gs.board:
            self._board_history.append(tuple(gs.board))
            if len(self._board_history) > self._window:
                self._board_history.pop(0)
            same_len = [b for b in self._board_history if len(b) == board_len]
            if same_len:
                gs.board = list(self._resolve(
                    same_len, board_len, self._board_locked))

        return gs

    def _resolve(self, history, num_cards, locked):
        # type: (List[Tuple], int, Dict[int, str]) -> Tuple
        """Per-position: return locked value if available, else majority vote."""
        result = []
        for i in range(num_cards):
            if i in locked:
                result.append(locked[i])
                continue

            cards_at_pos = [h[i] for h in history if i < len(h) and h[i] is not None]
            if not cards_at_pos:
                result.append(None)
                continue

            best, count = Counter(cards_at_pos).most_common(1)[0]
            if count >= self.LOCK_THRESHOLD:
                locked[i] = best
            result.append(best)
        return tuple(result)

# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _card_str(cards):
    # type: (list) -> str
    if not cards:
        return "--"
    return " ".join(c if c else "??" for c in cards)


def _format_state(gs, positions, preflop, solver_inputs):
    # type: (GameState, dict, ..., ...) -> str
    """Build a human-readable summary of the game state."""
    lines = []
    lines.append("")
    lines.append("=" * 60)

    # Street + board
    street = gs.street.upper()
    board = _card_str(gs.board) if gs.board else "(none)"
    lines.append("  Street: {:<12} Board: {}".format(street, board))

    # Hero
    hero = _card_str(gs.hero_cards) if gs.hero_cards else "--"
    lines.append("  Hero:   {}".format(hero))

    # Pot
    pot_str = ""
    if gs.total_bb is not None:
        pot_str = "{:.1f} BB".format(gs.total_bb)
    elif gs.pot_bb is not None:
        pot_str = "{:.1f} BB".format(gs.pot_bb)
    else:
        pot_str = "?"
    lines.append("  Pot:    {}".format(pot_str))

    # Dealer
    dealer_pos = positions.get(gs.dealer_seat, "?") if gs.dealer_seat else "?"
    lines.append("  Dealer: seat {} ({})".format(gs.dealer_seat or "?", dealer_pos))

    lines.append("-" * 60)

    # Players table
    lines.append("  {:<5} {:<5} {:<10} {:<8} {:<6} {}".format(
        "Seat", "Pos", "Name", "Stack", "Bet", "Action"))
    lines.append("  " + "-" * 52)

    for p in gs.players:
        if p.is_sitting_out:
            continue
        pos = positions.get(p.seat, "?")
        name = (p.name or "")[:10]
        stack = "{:.1f}".format(p.stack_bb) if p.stack_bb is not None else "?"
        bet = "{:.1f}".format(p.current_bet_bb) if p.current_bet_bb else ""
        action = p.action_label or ""
        if p.is_folded:
            action = "FOLD"
        marker = " *" if p.is_hero else ""
        lines.append("  {:<5} {:<5} {:<10} {:>8} {:>6} {}{}".format(
            p.seat, pos, name, stack, bet, action, marker))

    # Action buttons
    if gs.available_actions:
        lines.append("-" * 60)
        parts = []
        aa = gs.available_actions
        if aa.get("fold"):
            parts.append("Fold")
        if aa.get("check"):
            parts.append("Check")
        if aa.get("call") is not None:
            parts.append("Call {:.1f}".format(aa["call"]))
        if aa.get("raise_to") is not None:
            parts.append("Raise to {:.1f}".format(aa["raise_to"]))
        if aa.get("bet") is not None:
            parts.append("Bet {:.1f}".format(aa["bet"]))
        if parts:
            lines.append("  Actions: {}".format(" | ".join(parts)))
        hs = aa.get("hand_strength")
        if hs:
            lines.append("  Hand:    {}".format(hs))

    # Preflop reconstruction
    if preflop and preflop.opener:
        lines.append("-" * 60)
        pf_parts = ["{} opened".format(preflop.opener)]
        if preflop.callers:
            pf_parts.append("{} called".format(", ".join(preflop.callers)))
        if preflop.three_bettor:
            pf_parts.append("{} 3bet".format(preflop.three_bettor))
        if preflop.three_bet_callers:
            pf_parts.append("{} called 3bet".format(", ".join(preflop.three_bet_callers)))
        if preflop.four_bettor:
            pf_parts.append("{} 4bet".format(preflop.four_bettor))
        lines.append("  Preflop: {}".format(" -> ".join(pf_parts)))
        lines.append("  Type:    {} pot".format(preflop.scenario_type))

    # Solver inputs
    if solver_inputs:
        lines.append("-" * 60)
        lines.append("  SOLVER INPUTS:")
        lines.append("    OOP: {} ({})".format(
            solver_inputs["oop_position"],
            solver_inputs["oop_range"][:60] + "..." if len(solver_inputs["oop_range"]) > 60 else solver_inputs["oop_range"]))
        lines.append("    IP:  {} ({})".format(
            solver_inputs["ip_position"],
            solver_inputs["ip_range"][:60] + "..." if len(solver_inputs["ip_range"]) > 60 else solver_inputs["ip_range"]))
        lines.append("    Pot: {:.1f} BB  Eff stack: {:.1f} BB".format(
            solver_inputs["starting_pot"], solver_inputs["effective_stack"]))
        if solver_inputs["hero_position"]:
            lines.append("    Hero is {}".format(solver_inputs["hero_position"].upper()))

    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# State change detection
# ─────────────────────────────────────────────────────────────────────────────

def _state_fingerprint(gs):
    # type: (GameState) -> str
    """Quick fingerprint to detect meaningful state changes."""
    parts = [
        gs.street,
        str(gs.board),
        str(gs.hero_cards),
        str(gs.dealer_seat),
        str(gs.total_bb),
    ]
    for p in gs.players:
        parts.append("{}:{}:{}:{}".format(
            p.seat, p.action_label, p.current_bet_bb, p.is_folded))
    return "|".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Process one frame
# ─────────────────────────────────────────────────────────────────────────────

def process_and_display(img, tracker, rl):
    # type: (np.ndarray, HandTracker, RangeLookup) -> Optional[str]
    """Run pipeline + action reconstruction on an image. Returns fingerprint."""
    try:
        gs = process_screenshot(img)
    except Exception as exc:
        print("[watch] Pipeline error: {}".format(exc), file=sys.stderr)
        return None

    positions = gs.infer_positions()
    tracker.update(gs)
    preflop = tracker.preflop_action

    solver_inputs = None
    if gs.board and len(gs.board) >= 3:
        try:
            solver_inputs = determine_solver_inputs(gs, rl, hand_tracker=tracker)
        except Exception:
            pass

    print(_format_state(gs, positions, preflop, solver_inputs))
    sys.stdout.flush()

    return _state_fingerprint(gs)


# ─────────────────────────────────────────────────────────────────────────────
# Main loops
# ─────────────────────────────────────────────────────────────────────────────

_running = True

def _handle_sigint(signum, frame):
    global _running
    _running = False


def run_file(filepath):
    # type: (str) -> None
    """Process a single screenshot file."""
    img = capture_from_file(filepath)
    tracker = HandTracker()
    rl = RangeLookup()
    process_and_display(img, tracker, rl)


def run_live(window_index=0, show_all=False):
    # type: (int, bool) -> None
    """Continuously capture and display state changes."""
    global _running
    _running = True
    signal.signal(signal.SIGINT, _handle_sigint)

    tracker = HandTracker()
    smoother = ReadingSmoother()
    rl = RangeLookup()
    last_fingerprint = None
    tracked_window_id = None
    last_printed = None  # type: Optional[str]

    print("[watch] ACR Poker watcher starting. Press Ctrl+C to quit.")

    while _running:
        # Find window — pick the largest ACR window (handles multi-table)
        try:
            windows = find_acr_windows()
        except RuntimeError as exc:
            print("[watch] Fatal: {}".format(exc), file=sys.stderr)
            return

        if not windows:
            if tracked_window_id is not None:
                print("[watch] ACR window lost. Waiting...")
                tracked_window_id = None
            time.sleep(POLL_INTERVAL)
            continue

        # Pick the widest window (the expanded one in multi-table mode)
        win = max(windows, key=lambda w: w["bounds"]["w"])

        if win["id"] != tracked_window_id:
            tracked_window_id = win["id"]
            print("[watch] Tracking: {} ({}x{})".format(
                win["title"], win["bounds"]["w"], win["bounds"]["h"]))

        # Skip small tiled windows (multi-table mode)
        if win["bounds"]["w"] < MIN_WINDOW_WIDTH:
            time.sleep(POLL_INTERVAL)
            continue

        # Capture twice with short delay — only proceed if frame is settled
        img = capture_window(tracked_window_id)
        if img is None:
            tracked_window_id = None
            time.sleep(POLL_INTERVAL)
            continue

        time.sleep(0.15)
        img2 = capture_window(tracked_window_id)
        if img2 is None or img.shape != img2.shape:
            time.sleep(POLL_INTERVAL)
            continue

        # Compare frames: if too different, window is still animating
        diff = np.mean(np.abs(img.astype(float) - img2.astype(float)))
        if diff > 5.0:
            time.sleep(POLL_INTERVAL)
            continue

        # Use the second (more recent) frame
        img = img2

        # Run pipeline
        try:
            gs = process_screenshot(img)
        except Exception as exc:
            print("[watch] Pipeline error: {}".format(exc), file=sys.stderr)
            time.sleep(POLL_INTERVAL)
            continue

        # Apply temporal smoothing to stabilize card reads
        smoother.update(gs)

        fp = _state_fingerprint(gs)

        if show_all or fp != last_fingerprint:
            positions = gs.infer_positions()
            tracker.update(gs)
            preflop = tracker.preflop_action

            solver_inputs = None
            if gs.board and len(gs.board) >= 3:
                try:
                    solver_inputs = determine_solver_inputs(gs, rl, hand_tracker=tracker)
                except Exception:
                    pass

            output = _format_state(gs, positions, preflop, solver_inputs)
            # Suppress duplicate output after smoothing
            if output != last_printed:
                print(output)
                sys.stdout.flush()
                last_printed = output
            last_fingerprint = fp

        time.sleep(POLL_INTERVAL)

    print("\n[watch] Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ACR Poker watcher — live state display with action reconstruction."
    )
    parser.add_argument(
        "-f", "--file",
        help="Process a single screenshot file instead of live capture.",
    )
    parser.add_argument(
        "-w", "--window-index",
        type=int, default=0,
        help="ACR window index (default: 0).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show every capture, not just state changes.",
    )
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
    else:
        run_live(window_index=args.window_index, show_all=args.all)
