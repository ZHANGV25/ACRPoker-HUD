"""Live watcher — continuously captures ACR Poker and shows readable game state.

Runs OCR on every detected state change and prints a formatted summary
including board, hero cards, positions, actions, and solver inputs.
Supports multi-table: tracks all expanded ACR windows independently.

Usage:
    python -m src.watch              # live capture from ACR window(s)
    python -m src.watch -f file.png  # single screenshot (for testing)
    python -m src.watch --all        # capture every poll, not just changes
"""

import json
import os
import signal
import subprocess
import sys
import time
import threading
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
MIN_WINDOW_WIDTH = 750  # Skip tiled multi-table windows (expanded ~800pt, tiled ~600pt)
SMOOTH_WINDOW = 3       # Number of recent reads to vote on for stabilization

# Path to solver binary (built with cargo build --release)
SOLVER_BIN = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "solver", "solver-cli", "target", "release", "solver-cli"
)


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
            # Reject impossible duplicates (e.g. 3s 3s) — unlock both
            if len(gs.hero_cards) == 2 and gs.hero_cards[0] == gs.hero_cards[1]:
                self._hero_locked.pop(0, None)
                self._hero_locked.pop(1, None)

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
# Solver integration
# ─────────────────────────────────────────────────────────────────────────────

class SolverRunner:
    """Runs the Rust solver CLI in a background thread."""

    def __init__(self):
        # type: () -> None
        self._lock = threading.Lock()
        self._result = None  # type: Optional[dict]
        self._pending_key = None  # type: Optional[str]
        self._running = False

    def request(self, solver_inputs, hero_hand, hero_position, board):
        # type: (dict, List[str], str, List[str]) -> None
        """Request a solve in the background. Non-blocking."""
        if not os.path.isfile(SOLVER_BIN):
            return

        # Build a key to avoid re-solving the same spot
        key = "{}|{}|{}".format(
            ",".join(board), ",".join(hero_hand), hero_position)
        with self._lock:
            if key == self._pending_key:
                return  # already solving or solved this spot
            self._pending_key = key
            self._result = None

        solver_input = {
            "board": board,
            "oop_range": solver_inputs["oop_range"],
            "ip_range": solver_inputs["ip_range"],
            "starting_pot": solver_inputs["starting_pot"],
            "effective_stack": solver_inputs["effective_stack"],
            "hero_hand": hero_hand,
            "hero_position": hero_position,
            "max_iterations": 300,
            "bet_sizes_oop": "66%, a",
            "bet_sizes_ip": "66%, a",
            "raise_sizes_oop": "2.5x",
            "raise_sizes_ip": "2.5x",
        }

        thread = threading.Thread(
            target=self._solve, args=(key, solver_input), daemon=True)
        thread.start()

    def _solve(self, key, solver_input):
        # type: (str, dict) -> None
        """Run solver-cli subprocess."""
        try:
            proc = subprocess.run(
                [SOLVER_BIN],
                input=json.dumps(solver_input),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                result = json.loads(proc.stdout)
                with self._lock:
                    if self._pending_key == key:
                        self._result = result
            else:
                sys.stderr.write("[solver] Error: {}\n".format(
                    proc.stderr.strip()))
        except subprocess.TimeoutExpired:
            sys.stderr.write("[solver] Timed out after 120s\n")
        except Exception as exc:
            sys.stderr.write("[solver] Exception: {}\n".format(exc))

    def get_result(self):
        # type: () -> Optional[dict]
        """Return the latest solver result, or None if not ready."""
        with self._lock:
            return self._result


# ─────────────────────────────────────────────────────────────────────────────
# Per-table state — each ACR window gets its own tracker
# ─────────────────────────────────────────────────────────────────────────────

class TableState:
    """Tracks state for a single ACR table window."""

    def __init__(self, window_id, title, rl):
        # type: (int, str, RangeLookup) -> None
        self.window_id = window_id
        self.title = title
        self.tracker = HandTracker()
        self.smoother = ReadingSmoother()
        self.solver = SolverRunner()
        self.rl = rl
        self.last_fingerprint = None  # type: Optional[str]
        self.last_printed = None  # type: Optional[str]
        self.last_board_len = 0

    def process_frame(self, img, show_all=False):
        # type: (np.ndarray, bool) -> None
        """Process a single frame for this table."""
        try:
            gs = process_screenshot(img)
        except Exception as exc:
            print("[table {}] Pipeline error: {}".format(
                self.window_id, exc), file=sys.stderr)
            return

        # Apply temporal smoothing
        self.smoother.update(gs)

        fp = _state_fingerprint(gs)
        if not show_all and fp == self.last_fingerprint:
            return

        positions = gs.infer_positions()
        self.tracker.update(gs)
        preflop = self.tracker.preflop_action

        solver_inputs = None
        if gs.board and len(gs.board) >= 3:
            try:
                solver_inputs = self.tracker.get_solver_inputs(gs, self.rl)
            except Exception:
                pass

        # Trigger solver when we have inputs + hero cards + board
        if solver_inputs and gs.hero_cards and len(gs.hero_cards) == 2:
            hero_pos = solver_inputs.get("hero_position", "")
            if hero_pos and all(gs.hero_cards):
                board_len = len(gs.board)
                # Only solve turn+ (flop needs precomputed tables)
                if board_len >= 4:
                    self.solver.request(
                        solver_inputs, gs.hero_cards, hero_pos, gs.board)
                # New street → new solve needed
                if board_len > self.last_board_len:
                    self.last_board_len = board_len

        # Get solver result if available
        solver_result = self.solver.get_result()

        output = _format_state(
            gs, positions, preflop, solver_inputs, solver_result)

        # Suppress duplicate output
        if output != self.last_printed:
            print(output)
            sys.stdout.flush()
            self.last_printed = output
        self.last_fingerprint = fp


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _card_str(cards):
    # type: (list) -> str
    if not cards:
        return "--"
    return " ".join(c if c else "??" for c in cards)


def _format_state(gs, positions, preflop, solver_inputs, solver_result=None):
    # type: (GameState, dict, ..., ..., ...) -> str
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

    # Solver result
    if solver_result:
        lines.append("-" * 60)
        lines.append("  SOLVER STRATEGY:")
        lines.append("    EV: {:.2f} BB  Equity: {:.1f}%".format(
            solver_result["ev"], solver_result["equity"] * 100))
        for a in solver_result["actions"]:
            if a["frequency"] > 0.005:
                lines.append("    {:>20s}  {:.0f}%".format(
                    a["action"], a["frequency"] * 100))

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
# Process one frame (for single-file mode)
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


def _capture_settled(window_id):
    # type: (int) -> Optional[np.ndarray]
    """Capture a window twice and return frame only if settled (not animating)."""
    img = capture_window(window_id)
    if img is None:
        return None

    time.sleep(0.15)
    img2 = capture_window(window_id)
    if img2 is None or img.shape != img2.shape:
        return None

    diff = np.mean(np.abs(img.astype(float) - img2.astype(float)))
    if diff > 5.0:
        return None

    return img2


def run_live(show_all=False):
    # type: (bool) -> None
    """Continuously capture and display state changes for all ACR tables."""
    global _running
    _running = True
    signal.signal(signal.SIGINT, _handle_sigint)

    rl = RangeLookup()
    tables = {}  # type: Dict[int, TableState]

    if os.path.isfile(SOLVER_BIN):
        print("[watch] Solver binary found: {}".format(SOLVER_BIN))
    else:
        print("[watch] Solver binary not found — solving disabled.")
        print("[watch] Build with: cd solver/solver-cli && cargo build --release")

    print("[watch] ACR Poker watcher starting (multi-table). Press Ctrl+C to quit.")

    while _running:
        # Find all ACR windows
        try:
            windows = find_acr_windows()
        except RuntimeError as exc:
            print("[watch] Fatal: {}".format(exc), file=sys.stderr)
            return

        # Filter to expanded windows only (skip tiled)
        expanded = [w for w in windows if w["bounds"]["w"] >= MIN_WINDOW_WIDTH]

        if not expanded:
            if tables:
                print("[watch] All ACR windows lost. Waiting...")
                tables.clear()
            time.sleep(POLL_INTERVAL)
            continue

        # Track new windows, remove stale ones
        active_ids = {w["id"] for w in expanded}
        for wid in list(tables.keys()):
            if wid not in active_ids:
                print("[watch] Table {} lost.".format(wid))
                del tables[wid]

        for win in expanded:
            wid = win["id"]
            if wid not in tables:
                tables[wid] = TableState(wid, win["title"], rl)
                print("[watch] Tracking table: {} ({}x{})".format(
                    win["title"], win["bounds"]["w"], win["bounds"]["h"]))

        # Process each table
        for wid, table in tables.items():
            img = _capture_settled(wid)
            if img is None:
                continue
            table.process_frame(img, show_all=show_all)

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
        "--all",
        action="store_true",
        help="Show every capture, not just state changes.",
    )
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
    else:
        run_live(show_all=args.all)
