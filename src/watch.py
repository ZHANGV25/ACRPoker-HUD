"""Live monitor — continuously captures target application and shows readable game state.

Runs OCR on every detected state change and prints a formatted summary
including board, hero cards, positions, actions, and engine inputs.
Supports multi-table: tracks all expanded windows independently.

Usage:
    python -m src.watch              # live capture from target window(s)
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

from src.capture import find_target_windows, capture_window, capture_from_file
from src.pipeline import process_screenshot
from src.regions import FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON
from src.game_state import GameState
from solver.range_lookup import RangeLookup
from solver.action_history import HandTracker, reconstruct_preflop, determine_solver_inputs

POLL_INTERVAL = 0.8
MIN_WINDOW_WIDTH = 400  # Skip lobby/minimized windows; small tables can be ~467px
SMOOTH_WINDOW = 5       # Number of recent reads to vote on for stabilization

# Path to engine binary (built with cargo build --release)
ENGINE_BIN = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "solver", "solver-cli", "target", "release", "tbl-engine"
)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoothing — stabilizes flickering card reads
# ─────────────────────────────────────────────────────────────────────────────

class ReadingSmoother:
    """Stabilizes flickering card reads using lock-in + majority vote.

    Once a card is read the same way LOCK_THRESHOLD times, it's locked
    for the rest of the hand. Resets on new hand_id or new street (for board).
    """

    LOCK_THRESHOLD = 2  # reads before locking in (board cards)
    HERO_LOCK_THRESHOLD = 2  # lock hero cards after 2 consistent reads

    def __init__(self, window=SMOOTH_WINDOW):
        # type: (int) -> None
        self._window = window
        self._hero_history = []    # type: List[Tuple[Optional[str], ...]]
        self._hero_locked = {}     # type: Dict[int, str]  # position -> locked card
        self._board_history = []   # type: List[Tuple[str, ...]]
        self._board_locked = {}    # type: Dict[int, str]
        self._current_hand_id = None  # type: Optional[str]
        self._current_board_len = 0
        self._hero_action_skip = 0  # frames to skip after action appears

    def update(self, gs, hero_has_action=False):
        # type: (GameState, bool) -> GameState
        """Smooth hero cards and board across recent reads. Mutates gs in place.

        hero_has_action: only accumulate/lock hero cards when True.
        Before hero's turn the view is small and OCR is unreliable.
        """
        # Reset on new hand
        if gs.hand_id != self._current_hand_id:
            self._board_history.clear()
            self._board_locked.clear()
            self._current_hand_id = gs.hand_id
            self._current_board_len = 0

            # Always clear hero state on hand change to prevent stale locks
            self._hero_history.clear()
            self._hero_locked.clear()

        # Reset board locks when new cards appear (street change)
        board_len = len(gs.board) if gs.board else 0
        if board_len > self._current_board_len:
            self._board_history.clear()
            self._board_locked.clear()
            self._current_board_len = board_len

        # Hero cards: only accumulate/lock when hero has action (view is big)
        # Skip first frame after action appears (settle check filters animations)
        if not hero_has_action:
            self._hero_action_skip = 0
        elif self._hero_action_skip < 1:
            self._hero_action_skip += 1
            gs.hero_cards = None

        if gs.hero_cards and hero_has_action:
            self._hero_history.append(tuple(gs.hero_cards))
            if len(self._hero_history) > self._window:
                self._hero_history.pop(0)
            gs.hero_cards = list(self._resolve(
                self._hero_history, len(gs.hero_cards), self._hero_locked,
                threshold=self.HERO_LOCK_THRESHOLD))
            # Reject impossible duplicates (e.g. 3s 3s) — unlock both
            if len(gs.hero_cards) == 2 and gs.hero_cards[0] == gs.hero_cards[1]:
                self._hero_locked.pop(0, None)
                self._hero_locked.pop(1, None)
        elif gs.hero_cards and self._hero_locked:
            # Not hero's turn but we have prior locks — use them
            gs.hero_cards = list(self._resolve(
                [tuple(gs.hero_cards)], len(gs.hero_cards), self._hero_locked,
                threshold=self.HERO_LOCK_THRESHOLD))

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

    def _resolve(self, history, num_cards, locked, threshold=None):
        # type: (List[Tuple], int, Dict[int, str], Optional[int]) -> Tuple
        """Per-position: return locked value if available, else majority vote.

        Only locks when the last `threshold` reads are consecutively identical.
        This prevents wrong locks from a run of bad reads.
        Always shows the majority vote while waiting for lock.
        """
        if threshold is None:
            threshold = self.LOCK_THRESHOLD
        unlock_after = threshold + 1  # consecutive disagreements to override lock
        result = []
        for i in range(num_cards):
            if i in locked:
                # Check if recent reads consistently disagree with lock
                cards_at_pos = [h[i] for h in history if i < len(h) and h[i] is not None]
                if len(cards_at_pos) >= unlock_after:
                    recent = cards_at_pos[-unlock_after:]
                    if (all(c == recent[0] for c in recent)
                            and recent[0] != locked[i]):
                        # Consistent new read overrides stale lock
                        locked[i] = recent[0]
                result.append(locked[i])
                continue

            cards_at_pos = [h[i] for h in history if i < len(h) and h[i] is not None]
            if not cards_at_pos:
                result.append(None)
                continue

            best, _count = Counter(cards_at_pos).most_common(1)[0]
            # Lock only when last `threshold` reads are all identical (consecutive)
            if len(cards_at_pos) >= threshold:
                recent = cards_at_pos[-threshold:]
                if all(c == recent[0] for c in recent):
                    locked[i] = recent[0]
                    best = recent[0]
            result.append(best)
        return tuple(result)


# ─────────────────────────────────────────────────────────────────────────────
# Solver integration
# ─────────────────────────────────────────────────────────────────────────────

class EngineRunner:
    """Runs the Rust engine CLI in a background thread."""

    def __init__(self):
        # type: () -> None
        self._lock = threading.Lock()
        self._result = None  # type: Optional[dict]
        self._pending_key = None  # type: Optional[str]
        self._running = False
        self.status = "idle"  # type: str  # idle, running, done, error
        self.last_solve_time = 0.0
        self.last_error = ""  # type: str
        self._last_board_len = 0

    def invalidate_on_street_change(self, board_len):
        # type: (int) -> None
        """Clear stale result when a new street card appears."""
        if board_len > self._last_board_len:
            with self._lock:
                self._result = None
                self._pending_key = None
                self.status = "idle"
            self._last_board_len = board_len

    def request(self, solver_inputs, hero_hand, hero_position, board,
                bet_sizes=None, street_actions=None):
        # type: (dict, List[str], str, List[str], Optional[dict], Optional[list]) -> None
        """Request a solve in the background. Non-blocking."""
        if not os.path.isfile(ENGINE_BIN):
            self.status = "error"
            self.last_error = "binary not found"
            return

        sa_key = str(street_actions or [])
        # Key includes pot/stack/street_actions so we re-solve on action changes
        key = "{}|{}|{}|{:.0f}|{:.0f}|{}".format(
            ",".join(board), ",".join(hero_hand), hero_position,
            solver_inputs["starting_pot"], solver_inputs["effective_stack"],
            sa_key)
        with self._lock:
            if key == self._pending_key:
                return  # already solving or solved this spot
            self._pending_key = key
            self._result = None

        bs = bet_sizes or {}
        solver_input = {
            "board": board,
            "oop_range": solver_inputs["oop_range"],
            "ip_range": solver_inputs["ip_range"],
            "starting_pot": solver_inputs["starting_pot"],
            "effective_stack": solver_inputs["effective_stack"],
            "hero_hand": hero_hand,
            "hero_position": hero_position,
            "max_iterations": 300,
            "bet_sizes_oop": bs.get("bet_sizes_oop", "66%, a"),
            "bet_sizes_ip": bs.get("bet_sizes_ip", "66%, a"),
            "raise_sizes_oop": bs.get("raise_sizes_oop", "2.5x"),
            "raise_sizes_ip": bs.get("raise_sizes_ip", "2.5x"),
            "street_actions": street_actions or [],
        }

        self.status = "running"
        thread = threading.Thread(
            target=self._solve, args=(key, solver_input), daemon=True)
        thread.start()

    def _solve(self, key, solver_input):
        # type: (str, dict) -> None
        """Run engine subprocess."""
        t0 = time.time()
        try:
            proc = subprocess.run(
                [ENGINE_BIN],
                input=json.dumps(solver_input),
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                result = json.loads(proc.stdout)
                with self._lock:
                    if self._pending_key == key:
                        self._result = result
                        self.status = "done"
                        self.last_solve_time = elapsed
                        self.last_error = ""
            else:
                err = proc.stderr.strip()
                sys.stderr.write("[engine] Error: {}\n".format(err))
                with self._lock:
                    self.status = "error"
                    self.last_error = err[:120]
        except subprocess.TimeoutExpired:
            sys.stderr.write("[engine] Timed out after 120s\n")
            with self._lock:
                self.status = "error"
                self.last_error = "timeout (120s)"
        except Exception as exc:
            sys.stderr.write("[engine] Exception: {}\n".format(exc))
            with self._lock:
                self.status = "error"
                self.last_error = str(exc)[:120]

    def get_result(self):
        # type: () -> Optional[dict]
        """Return the latest engine result, or None if not ready."""
        with self._lock:
            return self._result


# ─────────────────────────────────────────────────────────────────────────────
# Per-table state — each target window gets its own tracker
# ─────────────────────────────────────────────────────────────────────────────

class TableState:
    """Tracks state for a single table window."""

    def __init__(self, window_id, title, rl):
        # type: (int, str, RangeLookup) -> None
        self.window_id = window_id
        self.title = title
        self.tracker = HandTracker()
        self.smoother = ReadingSmoother()
        self.engine = EngineRunner()
        self.rl = rl
        self.last_fingerprint = None  # type: Optional[str]
        self.last_printed = None  # type: Optional[str]

    def process_frame(self, img, show_all=False):
        # type: (np.ndarray, bool) -> None
        """Process a single frame for this table."""
        t_start = time.time()
        try:
            gs = process_screenshot(img)
        except Exception as exc:
            print("[table {}] Pipeline error: {}".format(
                self.window_id, exc), file=sys.stderr)
            return
        ocr_ms = (time.time() - t_start) * 1000

        # Hero-has-action gating: must be computed before smoother
        hero_has_action = bool(gs.available_actions and (
            gs.available_actions.get("fold") or
            gs.available_actions.get("check") or
            gs.available_actions.get("call") is not None))

        # When hero doesn't have action, view is small — suppress unreliable hero reads
        if not hero_has_action and not self.smoother._hero_locked:
            gs.hero_cards = None

        # Apply temporal smoothing (hero cards only smoothed when view is big)
        self.smoother.update(gs, hero_has_action=hero_has_action)

        fp = _state_fingerprint(gs)
        if not show_all and fp == self.last_fingerprint:
            return

        positions = gs.infer_positions()
        self.tracker.update(gs)
        preflop = self.tracker.preflop_action

        solver_inputs = None
        solver_exc = None
        if gs.board and len(gs.board) >= 3:
            try:
                solver_inputs = self.tracker.get_solver_inputs(gs, self.rl)
            except Exception as e:
                solver_exc = str(e)

        # Invalidate stale solver result on street change
        board_len = len(gs.board) if gs.board else 0
        self.engine.invalidate_on_street_change(board_len)

        # Trigger solver when we have inputs + hero cards + board + hero has action
        # Only solve turn/river — flop is too slow for live play
        if solver_inputs and gs.hero_cards and len(gs.hero_cards) == 2:
            hero_pos = solver_inputs.get("hero_position", "")
            if hero_pos and all(gs.hero_cards) and hero_has_action and board_len >= 4:
                bet_sizes = self.tracker.get_bet_sizes()
                street_actions = self.tracker.get_street_actions(gs, solver_inputs)
                self.engine.request(
                    solver_inputs, gs.hero_cards, hero_pos, gs.board,
                    bet_sizes=bet_sizes, street_actions=street_actions)

        # Get solver result if available
        solver_result = self.engine.get_result()

        # Log engine block reason for debugging
        if self.engine.status == "idle" and hero_has_action and board_len >= 4:
            reasons = []
            if solver_exc:
                reasons.append("solver_exc:" + solver_exc[:60])
            if not solver_inputs:
                reasons.append("no_solver_inputs")
            if not gs.hero_cards or len(gs.hero_cards) != 2:
                reasons.append("no_hero_cards")
            elif not all(gs.hero_cards):
                reasons.append("hero_cards_partial:{}".format(gs.hero_cards))
            if solver_inputs and not solver_inputs.get("hero_position"):
                reasons.append("no_hero_pos")
            if reasons:
                sys.stderr.write("[engine-block] {}\n".format(" | ".join(reasons)))

        # Build debug info
        debug = {
            "ocr_ms": ocr_ms,
            "engine_status": self.engine.status,
            "engine_time": self.engine.last_solve_time,
            "engine_error": self.engine.last_error,
            "hero_has_action": hero_has_action,
            "hand_id": gs.hand_id,
            "tracked_bets": len(self.tracker._observed_bet_pcts),
        }

        output = _format_state(
            gs, positions, preflop, solver_inputs, solver_result, debug=debug)

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


def _format_state(gs, positions, preflop, solver_inputs, solver_result=None, debug=None):
    # type: (GameState, dict, ..., ..., ..., ...) -> str
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
        lines.append("  ENGINE STRATEGY:")
        lines.append("    EV: {:.2f} BB  Equity: {:.1f}%".format(
            solver_result["ev"], solver_result["equity"] * 100))
        for a in solver_result["actions"]:
            if a["frequency"] > 0.005:
                lines.append("    {:>20s}  {:.0f}%".format(
                    a["action"], a["frequency"] * 100))

    # Debug info
    if debug:
        lines.append("-" * 60)
        status = debug.get("engine_status", "?")
        ocr_ms = debug.get("ocr_ms", 0)
        solve_t = debug.get("engine_time", 0)
        has_action = debug.get("hero_has_action", False)
        hand_id = debug.get("hand_id", "?")
        err = debug.get("engine_error", "")
        n_bets = debug.get("tracked_bets", 0)

        status_str = status
        if status == "done" and solve_t > 0:
            status_str = "done ({:.1f}s)".format(solve_t)
        elif status == "error" and err:
            status_str = "ERR: {}".format(err[:60])

        lines.append("  DEBUG  ocr:{:.0f}ms  engine:{}  action:{}  bets:{}  hand:{}".format(
            ocr_ms, status_str,
            "YES" if has_action else "no",
            n_bets, hand_id or "?"))

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
        print("[monitor] Pipeline error: {}".format(exc), file=sys.stderr)
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

    time.sleep(0.25)
    img2 = capture_window(window_id)
    if img2 is None or img.shape != img2.shape:
        return None

    diff = np.mean(np.abs(img.astype(float) - img2.astype(float)))
    if diff > 3.0:
        return None

    return img2


def run_live(show_all=False):
    # type: (bool) -> None
    """Continuously capture and display state changes for all target tables."""
    global _running
    _running = True
    signal.signal(signal.SIGINT, _handle_sigint)

    rl = RangeLookup()
    tables = {}  # type: Dict[int, TableState]

    if os.path.isfile(ENGINE_BIN):
        print("[monitor] Engine binary found: {}".format(ENGINE_BIN))
    else:
        print("[monitor] Engine binary not found — engine disabled.")
        print("[monitor] Build with: cd solver/solver-cli && cargo build --release")

    print("[monitor] Monitor starting (multi-table). Press Ctrl+C to quit.")

    while _running:
        # Find all target windows
        try:
            windows = find_target_windows()
        except RuntimeError as exc:
            print("[monitor] Fatal: {}".format(exc), file=sys.stderr)
            return

        # Filter to expanded windows only (skip tiled)
        expanded = [w for w in windows if w["bounds"]["w"] >= MIN_WINDOW_WIDTH]

        if not expanded:
            if tables:
                print("[monitor] All target windows lost. Waiting...")
                tables.clear()
            time.sleep(POLL_INTERVAL)
            continue

        # Track new windows, remove stale ones
        active_ids = {w["id"] for w in expanded}
        for wid in list(tables.keys()):
            if wid not in active_ids:
                print("[monitor] Table {} lost.".format(wid))
                del tables[wid]

        for win in expanded:
            wid = win["id"]
            if wid not in tables:
                tables[wid] = TableState(wid, win["title"], rl)
                print("[monitor] Tracking table: {} ({}x{})".format(
                    win["title"], win["bounds"]["w"], win["bounds"]["h"]))

        # Process each table
        for wid, table in tables.items():
            img = _capture_settled(wid)
            if img is None:
                continue
            table.process_frame(img, show_all=show_all)

        time.sleep(POLL_INTERVAL)

    print("\n[monitor] Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live monitor — state display with action reconstruction."
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
