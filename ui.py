#!/usr/bin/env python3
"""Poker HUD overlay — native macOS AppKit GUI with styled text."""

import sys
import os
import random
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

import objc
import AppKit
from AppKit import (
    NSApplication, NSWindow, NSScrollView, NSTextView,
    NSBackingStoreBuffered, NSMakeRect, NSColor, NSFont,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable, NSWindowStyleMaskMiniaturizable,
    NSFloatingWindowLevel,
)
from Foundation import NSMutableAttributedString, NSRange
from AppKit import NSForegroundColorAttributeName, NSFontAttributeName
import numpy as np

from src.capture import find_target_windows, capture_window
from src.pipeline import process_screenshot
from src.watch import ReadingSmoother, EngineRunner, ENGINE_BIN, MIN_WINDOW_WIDTH
from solver.range_lookup import RangeLookup, preflop_advice
from solver.action_history import HandTracker
from solver.hh_watcher import HHWatcher
from solver.exploitative import adjust_advice, adjust_solver_ranges

POLL_INTERVAL = 1.0
HERO_NAME = "vortexted"
HERO_SEAT = 5  # ACR always puts hero at seat 5

# ─── Colors ───────────────────────────────────────────────────────────────────

def _rgb(r, g, b, a=1.0):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)

CLR_BG        = _rgb(0.08, 0.08, 0.14, 0.95)
CLR_TEXT      = _rgb(0.82, 0.82, 0.82)
CLR_DIM       = _rgb(0.50, 0.50, 0.55)
CLR_GREEN     = _rgb(0.30, 0.85, 0.40)
CLR_BLUE      = _rgb(0.40, 0.70, 1.00)
CLR_RED       = _rgb(1.00, 0.35, 0.35)
CLR_YELLOW    = _rgb(1.00, 0.85, 0.30)
CLR_ORANGE    = _rgb(1.00, 0.60, 0.20)
CLR_WHITE     = _rgb(1.00, 1.00, 1.00)
CLR_BAR_BG    = _rgb(0.20, 0.20, 0.28)
CLR_HERO      = _rgb(0.55, 0.80, 1.00)
CLR_EXPLOIT   = _rgb(1.00, 0.75, 0.30)

FONT_ACTION   = NSFont.fontWithName_size_("Menlo-Bold", 28)
FONT_HEADER   = NSFont.fontWithName_size_("Menlo-Bold", 13)
FONT_BODY     = NSFont.fontWithName_size_("Menlo", 12)
FONT_SMALL    = NSFont.fontWithName_size_("Menlo", 10)
FONT_BAR      = NSFont.fontWithName_size_("Menlo-Bold", 12)

ACTION_COLORS = {
    "raise": CLR_GREEN, "bet": CLR_GREEN, "4-bet": CLR_GREEN, "3-bet": CLR_GREEN,
    "call": CLR_BLUE, "check": CLR_BLUE,
    "fold": CLR_RED,
}


# ─── RNG Action Picker ───────────────────────────────────────────────────────

class ActionPicker:
    """Picks a single action from solver frequencies using RNG.

    Locks the pick per spot — only re-rolls when the solver result changes.
    """

    def __init__(self):
        self._last_result_id = None  # type: object
        self._picked = None  # type: dict  # {"action": ..., "frequency": ...}

    def pick(self, solver_result):
        # type: (dict) -> dict
        """Pick an action from solver result. Returns the chosen action dict."""
        if solver_result is None:
            self._picked = None
            self._last_result_id = None
            return None

        # Use id() to detect new result objects
        result_id = id(solver_result)
        if result_id == self._last_result_id and self._picked:
            return self._picked

        self._last_result_id = result_id
        actions = solver_result.get("actions", [])
        if not actions:
            self._picked = None
            return None

        # Roll RNG and pick based on cumulative frequency
        roll = random.random()
        cumulative = 0.0
        for a in actions:
            cumulative += a["frequency"]
            if roll <= cumulative:
                self._picked = a
                return a

        # Fallback: pick highest frequency
        self._picked = max(actions, key=lambda x: x["frequency"])
        return self._picked

    def reset(self):
        self._picked = None
        self._last_result_id = None


# ─── Styled Text Builder ─────────────────────────────────────────────────────

class StyledText:
    """Builds NSMutableAttributedString incrementally."""

    def __init__(self):
        self._parts = []  # type: list  # (text, font, color)

    def add(self, text, font=None, color=None):
        # type: (str, ..., ...) -> StyledText
        self._parts.append((text, font or FONT_BODY, color or CLR_TEXT))
        return self

    def nl(self):
        return self.add("\n")

    def line(self, text, font=None, color=None):
        return self.add(text + "\n", font, color)

    def build(self):
        # type: () -> NSMutableAttributedString
        result = NSMutableAttributedString.alloc().init()
        for text, font, color in self._parts:
            attrs = {
                NSForegroundColorAttributeName: color,
                NSFontAttributeName: font,
            }
            seg = NSMutableAttributedString.alloc().initWithString_attributes_(text, attrs)
            result.appendAttributedString_(seg)
        return result


# ─── Frame Processor ──────────────────────────────────────────────────────────

def _action_color(action_name):
    # type: (str) -> ...
    """Get color for an action name."""
    lower = action_name.lower()
    for key, color in ACTION_COLORS.items():
        if key in lower:
            return color
    return CLR_TEXT


def _bar_str(pct, width=15):
    # type: (int, int) -> str
    filled = max(0, min(width, int(pct / 100.0 * width)))
    return "\u2593" * filled + "\u2591" * (width - filled)


def process_frame(img, tracker, smoother, engine, rl, name_cache,
                  hh_watcher, action_picker, table_name=""):
    # type: (...) -> StyledText
    """Process one frame and return styled text for the overlay."""
    t0 = time.time()
    gs = process_screenshot(img)
    ocr_ms = (time.time() - t0) * 1000

    raw_hero = list(gs.hero_cards) if gs.hero_cards else []

    aa = gs.available_actions or {}
    hero_has_action = bool(aa and (
        aa.get("fold") or aa.get("check") or aa.get("call") is not None))

    if not hero_has_action and not smoother._hero_locked:
        gs.hero_cards = None

    smoother.update(gs, hero_has_action=hero_has_action)

    # Hero name is always known — hardcode it
    for p in gs.players:
        if p.seat == HERO_SEAT:
            p.name = HERO_NAME
            p.is_hero = True
            name_cache[HERO_SEAT] = HERO_NAME

    # Seed name cache from hand history seat mapping (ground truth from disk)
    if hh_watcher and table_name and not name_cache.get("_hh_seeded"):
        hh_names = hh_watcher.get_table_names(table_name)
        if hh_names:
            for seat, hh_name in hh_names.items():
                if seat != HERO_SEAT:
                    name_cache[seat] = hh_name
            name_cache["_hh_seeded"] = True

    # Stabilize opponent names: only update from big-view reads,
    # and resolve through HH fuzzy matching for accuracy
    if hero_has_action:
        for p in gs.players:
            if p.seat == HERO_SEAT:
                continue
            if p.name and len(p.name) >= 3:
                # Resolve OCR name to known HH name
                if hh_watcher:
                    resolved = hh_watcher._resolve_name(p.name)
                    if resolved:
                        name_cache[p.seat] = resolved
                    elif p.seat not in name_cache:
                        name_cache[p.seat] = p.name
                elif p.seat not in name_cache:
                    name_cache[p.seat] = p.name
    for p in gs.players:
        if p.seat in name_cache and not isinstance(name_cache.get(p.seat), bool):
            p.name = name_cache[p.seat]

    positions = gs.infer_positions()
    tracker.update(gs)

    solver_inputs = None
    solver_err = ""
    if gs.board and len(gs.board) >= 3:
        try:
            solver_inputs = tracker.get_solver_inputs(gs, rl)
        except Exception as e:
            solver_err = str(e)[:80]

    board_len = len(gs.board) if gs.board else 0
    engine.invalidate_on_street_change(board_len)

    if solver_inputs and gs.hero_cards and len(gs.hero_cards) == 2:
        hero_pos = solver_inputs.get("hero_position", "")
        if hero_pos and all(gs.hero_cards) and hero_has_action:
            if hh_watcher:
                for p in gs.players:
                    if p.is_sitting_out or p.is_folded or p.is_hero:
                        continue
                    if p.name:
                        vs = hh_watcher.get_player_stats(p.name)
                        solver_inputs = adjust_solver_ranges(solver_inputs, vs)
                        break

            if board_len >= 4:
                bet_sizes = tracker.get_bet_sizes()
                street_actions = tracker.get_street_actions(gs, solver_inputs)
                engine.request(solver_inputs, gs.hero_cards, hero_pos, gs.board,
                               bet_sizes=bet_sizes, street_actions=street_actions)

    result = engine.get_result()
    status = engine.status

    # ─── Build styled output ───

    st = StyledText()

    # Hero info line
    hero_str = " ".join(c if c else "??" for c in gs.hero_cards) if gs.hero_cards else "--"
    hero_pos_str = ""
    if solver_inputs and solver_inputs.get("hero_position"):
        hero_pos_str = solver_inputs["hero_position"].upper()
    else:
        for p in gs.players:
            if p.is_hero and p.seat in positions:
                hero_pos_str = positions[p.seat]
                break

    pot = gs.total_bb or gs.pot_bb
    pot_str = "{:.1f}".format(pot) if pot else "?"

    st.add("  {} ".format(hero_str), FONT_HEADER, CLR_WHITE)
    if hero_pos_str:
        st.add("({})".format(hero_pos_str), FONT_HEADER, CLR_YELLOW)
    st.add("   Pot {} BB".format(pot_str), FONT_BODY, CLR_DIM)
    st.nl()

    # Street + board
    street = gs.street.upper() if gs.street else "?"
    board_str = " ".join(c if c else "??" for c in gs.board) if gs.board else "--"
    st.add("  {} ".format(street), FONT_BODY, CLR_DIM)
    st.add(board_str, FONT_HEADER, CLR_WHITE)
    st.nl()

    # ─── Main action area ───
    st.nl()

    if gs.street == "preflop" and hero_has_action:
        _render_preflop(st, gs, positions, rl, hh_watcher)
    elif result and hero_has_action:
        _render_postflop(st, result, engine, action_picker)
    elif status == "running":
        st.add("      ", FONT_ACTION, CLR_DIM)
        st.line("...", FONT_ACTION, CLR_DIM)
    elif gs.street == "flop" and hero_has_action:
        st.line("  (flop -- no live solve)", FONT_BODY, CLR_DIM)
    elif not hero_has_action:
        st.line("  Waiting for action...", FONT_BODY, CLR_DIM)
    elif status == "error":
        st.line("  Engine: {}".format(engine.last_error[:50]), FONT_SMALL, CLR_RED)
    elif solver_err:
        st.line("  {}".format(solver_err[:50]), FONT_SMALL, CLR_DIM)
    else:
        st.line("  (idle)", FONT_BODY, CLR_DIM)

    # ─── Players ───
    st.nl()
    st.line("  \u2500\u2500\u2500 players \u2500\u2500\u2500", FONT_SMALL, CLR_DIM)
    for p in gs.players:
        if p.is_sitting_out:
            continue
        pos = positions.get(p.seat, "?")
        name = (p.name or "")[:12]
        stack = "{:.1f}".format(p.stack_bb) if p.stack_bb is not None else "?"
        action = p.action_label or ""
        if p.is_folded:
            action = "FOLD"

        if p.is_hero:
            st.add("  {:<4}".format(pos), FONT_SMALL, CLR_HERO)
            st.add("{:<12}".format(name), FONT_SMALL, CLR_HERO)
            st.add("{:>6}".format(stack), FONT_SMALL, CLR_HERO)
            st.add(" \u25c0", FONT_SMALL, CLR_HERO)
        else:
            st.add("  {:<4}".format(pos), FONT_SMALL, CLR_DIM)
            st.add("{:<12}".format(name), FONT_SMALL, CLR_TEXT)
            st.add("{:>6}".format(stack), FONT_SMALL, CLR_DIM)

            # Player stats
            if hh_watcher and p.name:
                ps = hh_watcher.get_player_stats(p.name)
                if ps.hands >= 5:
                    arch_color = _archetype_color(ps.archetype)
                    st.add("  {:.0f}/{:.0f}".format(ps.vpip, ps.pfr), FONT_SMALL, CLR_DIM)
                    st.add(" {}".format(ps.archetype), FONT_SMALL, arch_color)

        if action and not p.is_hero:
            st.add(" {}".format(action), FONT_SMALL, CLR_DIM)
        st.nl()

    # ─── Debug ───
    st.nl()
    st.line("  \u2500\u2500\u2500 debug \u2500\u2500\u2500", FONT_SMALL, CLR_DIM)
    raw_h = " ".join(c if c else "??" for c in raw_hero) if raw_hero else "--"
    locked_h = " ".join(smoother._hero_locked.get(i, "?") for i in range(2))
    hero_ocr_on = hero_has_action or bool(smoother._hero_locked)

    debug_parts = ["{}ms".format(int(ocr_ms))]
    if status == "done" and engine.last_solve_time:
        debug_parts.append("solve:{:.1f}s".format(engine.last_solve_time))
    elif status != "idle":
        debug_parts.append(status)
    debug_parts.append("hand:{}".format((gs.hand_id or "?")[-6:]))
    if hh_watcher:
        debug_parts.append("hh:{}".format(hh_watcher.total_hands))
    debug_parts.append("ocr:{}".format("ON" if hero_ocr_on else "OFF"))
    st.add("  {}".format(" | ".join(debug_parts)), FONT_SMALL, CLR_DIM)
    st.nl()
    st.add("  raw:{} lock:{}".format(raw_h, locked_h), FONT_SMALL, CLR_DIM)
    if solver_err:
        st.nl()
        st.add("  {}".format(solver_err[:50]), FONT_SMALL, CLR_RED)
    st.nl()

    # Detailed player stats in debug
    if hh_watcher:
        st.nl()
        st.line("  \u2500\u2500\u2500 player stats \u2500\u2500\u2500", FONT_SMALL, CLR_DIM)
        st.add("  {:<12} {:>5} {:>5} {:>4} {:>5} {:>4}  {}".format(
            "Name", "VPIP", "PFR", "3B", "FdCB", "#H", "Type"), FONT_SMALL, CLR_DIM)
        st.nl()
        for p in gs.players:
            if p.is_sitting_out or p.is_hero or not p.name:
                continue
            ps = hh_watcher.get_player_stats(p.name)
            if ps.hands < 1:
                continue
            arch_color = _archetype_color(ps.archetype)
            st.add("  {:<12}".format((p.name or "")[:12]), FONT_SMALL, CLR_TEXT)
            st.add(" {:>4.0f}%".format(ps.vpip), FONT_SMALL, CLR_DIM)
            st.add(" {:>4.0f}%".format(ps.pfr), FONT_SMALL, CLR_DIM)
            st.add(" {:>3.0f}%".format(ps.three_bet), FONT_SMALL, CLR_DIM)
            if ps.fold_to_cbet > 0:
                st.add(" {:>4.0f}%".format(ps.fold_to_cbet), FONT_SMALL, CLR_DIM)
            else:
                st.add("    -", FONT_SMALL, CLR_DIM)
            st.add(" {:>4}".format(ps.hands), FONT_SMALL, CLR_DIM)
            st.add("  {}".format(ps.archetype), FONT_SMALL, arch_color)
            st.nl()

    return st


def _render_preflop(st, gs, positions, rl, hh_watcher):
    # type: (StyledText, ..., dict, RangeLookup, ...) -> None
    """Render preflop advice in the action area."""
    hero_pos_name = None
    for p in gs.players:
        if p.is_hero and p.seat in positions:
            hero_pos_name = positions[p.seat]
            break

    advice = None
    if hero_pos_name and gs.hero_cards:
        advice = preflop_advice(gs.hero_cards, hero_pos_name, gs, rl)

    exploit_tip = None
    if advice and hh_watcher:
        villain_name = None
        for p in gs.players:
            if p.is_sitting_out or p.is_folded or p.is_hero:
                continue
            bet = p.current_bet_bb or 0
            pos = positions.get(p.seat, "")
            if pos not in ("SB", "BB") and bet > 1.0:
                villain_name = p.name
                break
            if pos == "BB" and bet > 1.0:
                villain_name = p.name
                break
        if villain_name:
            vs = hh_watcher.get_player_stats(villain_name)
            adjusted = adjust_advice(advice, vs)
            if adjusted != advice:
                # Extract the exploit tip part
                exploit_tip = adjusted[len(advice):]
                advice = adjusted

    if not advice:
        st.line("  (waiting for preflop data)", FONT_BODY, CLR_DIM)
        return

    # Parse the action word from advice (e.g. "RAISE  AKs in CO's open range")
    action_word = advice.split()[0] if advice else ""
    color = _action_color(action_word)
    detail = advice[len(action_word):].strip() if len(advice) > len(action_word) else ""

    # Big action
    st.add("       ", FONT_ACTION)
    st.line(action_word, FONT_ACTION, color)

    # Detail line
    if detail:
        # Split off exploit tip if it was appended
        base_detail = detail
        tip = ""
        if exploit_tip:
            base_detail = detail[:-(len(exploit_tip))].strip()
            tip = exploit_tip.strip()
        st.add("  {}".format(base_detail), FONT_BODY, CLR_TEXT)
        if tip:
            st.nl()
            st.add("  {}".format(tip), FONT_BODY, CLR_EXPLOIT)
        st.nl()


def _render_postflop(st, result, engine, action_picker):
    # type: (StyledText, dict, EngineRunner, ActionPicker) -> None
    """Render postflop solver result with RNG-picked action."""
    picked = action_picker.pick(result)
    if not picked:
        st.line("  (no action)", FONT_BODY, CLR_DIM)
        return

    action_name = picked["action"]
    color = _action_color(action_name)

    # Big action recommendation
    # Clean up action name for display (e.g. "Raise 66%" -> "RAISE")
    display_name = action_name.split()[0].upper() if action_name else "?"
    st.add("       ", FONT_ACTION)
    st.line(display_name, FONT_ACTION, color)

    # EV + equity line
    st.add("  EV ", FONT_BODY, CLR_DIM)
    ev = result.get("ev", 0)
    ev_color = CLR_GREEN if ev > 0 else CLR_RED if ev < 0 else CLR_TEXT
    st.add("{:+.2f} BB".format(ev), FONT_BODY, ev_color)
    st.add("  Equity ", FONT_BODY, CLR_DIM)
    st.add("{:.0f}%".format(result.get("equity", 0) * 100), FONT_BODY, CLR_TEXT)
    st.nl()
    st.nl()

    # Frequency bars
    actions = sorted(result.get("actions", []), key=lambda a: -a["frequency"])
    for a in actions:
        freq = a["frequency"]
        if freq < 0.005:
            continue
        pct = int(freq * 100)
        bar_color = _action_color(a["action"])
        is_picked = (a["action"] == picked["action"])

        marker = "\u25b6 " if is_picked else "  "
        st.add("  {}".format(marker), FONT_BAR, CLR_WHITE if is_picked else CLR_DIM)
        st.add(_bar_str(pct), FONT_BAR, bar_color)
        st.add(" {:>3d}% ".format(pct), FONT_BAR, bar_color)
        st.add("{}".format(a["action"]), FONT_SMALL, CLR_TEXT if is_picked else CLR_DIM)
        st.nl()


def _archetype_color(archetype):
    # type: (str) -> ...
    m = {
        "fish": CLR_GREEN, "whale": CLR_GREEN, "calling_station": CLR_GREEN,
        "nit": CLR_BLUE, "TAG": CLR_TEXT,
        "LAG": CLR_ORANGE, "maniac": CLR_RED,
        "unknown": CLR_DIM,
    }
    return m.get(archetype, CLR_DIM)


# ─── Capture ──────────────────────────────────────────────────────────────────

def _extract_table_name(title):
    # type: (str) -> str
    """Extract table name from ACR window title.

    Title format: 'Heyburn - No Limit - $0.01 / $0.02 Hold'em ...'
    Returns: 'Heyburn'
    """
    if not title:
        return ""
    # Table name is the first word before ' - '
    parts = title.split(" - ", 1)
    return parts[0].strip() if parts else ""


def _capture_settled():
    """Find table and capture settled frame. Returns (img, title, table_name)."""
    try:
        windows = find_target_windows()
    except RuntimeError:
        return None, "", ""
    expanded = [w for w in windows if w["bounds"]["w"] >= MIN_WINDOW_WIDTH]
    if not expanded:
        return None, "", ""
    win = expanded[0]
    title = "{} ({}x{})".format(win["title"][:40], win["bounds"]["w"], win["bounds"]["h"])
    table_name = _extract_table_name(win["title"])

    img1 = capture_window(win["id"])
    if img1 is None:
        return None, title, table_name
    time.sleep(0.12)
    img2 = capture_window(win["id"])
    if img2 is None or img1.shape != img2.shape:
        return None, title, table_name
    diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
    if diff > 5.0:
        return None, title, table_name
    return img2, title, table_name


# ─── App Delegate ─────────────────────────────────────────────────────────────

class OverlayDelegate(AppKit.NSObject):
    """App delegate that creates the overlay window and runs the poll loop."""

    def applicationDidFinishLaunching_(self, notification):
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(20, 20, 380, 480), style, NSBackingStoreBuffered, False)
        self.window.setTitle_("Poker HUD")
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(CLR_BG)

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 380, 480))
        scroll.setHasVerticalScroller_(True)
        scroll.setDrawsBackground_(False)

        self.text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 380, 480))
        self.text_view.setEditable_(False)
        self.text_view.setSelectable_(True)
        self.text_view.setDrawsBackground_(True)
        self.text_view.setBackgroundColor_(CLR_BG)

        scroll.setDocumentView_(self.text_view)
        self.window.setContentView_(scroll)
        self.window.makeKeyAndOrderFront_(None)

        # State
        self.tracker = HandTracker()
        self.smoother = ReadingSmoother()
        self.engine = EngineRunner()
        self.rl = RangeLookup()
        self.hh_watcher = HHWatcher()
        self.hh_watcher.start()
        self.action_picker = ActionPicker()
        self.last_text_hash = None

        # Queue for styled text from worker thread
        self._queue = []  # type: list
        self._queue_lock = threading.Lock()

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

        self._start_timer()

    def _start_timer(self):
        AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.3, self, objc.selector(self.onTimer_, signature=b'v@:@'), None, True)

    @objc.python_method
    def _worker(self):
        name_cache = {}
        while True:
            try:
                img, title, table_name = _capture_settled()
                if img is None:
                    st = StyledText()
                    st.nl()
                    st.line("  Waiting for table...", FONT_HEADER, CLR_DIM)
                    if title:
                        st.line("  {}".format(title), FONT_SMALL, CLR_DIM)
                else:
                    st = process_frame(
                        img, self.tracker, self.smoother, self.engine,
                        self.rl, name_cache, self.hh_watcher, self.action_picker,
                        table_name=table_name)

                with self._queue_lock:
                    self._queue.append(st)
            except Exception as e:
                st = StyledText()
                st.line("Error: {}".format(str(e)[:100]), FONT_BODY, CLR_RED)
                with self._queue_lock:
                    self._queue.append(st)
            time.sleep(POLL_INTERVAL)

    def onTimer_(self, timer):
        with self._queue_lock:
            if not self._queue:
                return
            st = self._queue[-1]
            self._queue.clear()

        attr_str = st.build()
        text_hash = attr_str.string()
        if text_hash == self.last_text_hash:
            return
        self.last_text_hash = text_hash

        storage = self.text_view.textStorage()
        storage.setAttributedString_(attr_str)


def main():
    app = NSApplication.sharedApplication()
    delegate = OverlayDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)
    app.run()


if __name__ == "__main__":
    main()
