#!/usr/bin/env python3
"""Compact always-on-top overlay using native macOS AppKit."""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

import objc
import AppKit
from AppKit import (
    NSApplication, NSWindow, NSTextView, NSScrollView,
    NSBackingStoreBuffered, NSMakeRect, NSColor, NSFont,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable, NSWindowStyleMaskMiniaturizable,
    NSFloatingWindowLevel, NSAttributedString,
)
from Foundation import NSMutableAttributedString, NSRange, NSDictionary
import numpy as np

from src.capture import find_target_windows, capture_window
from src.pipeline import process_screenshot
from src.watch import ReadingSmoother, EngineRunner, ENGINE_BIN, MIN_WINDOW_WIDTH
from solver.range_lookup import RangeLookup, preflop_advice
from solver.action_history import HandTracker
from solver.hh_watcher import HHWatcher
from solver.exploitative import adjust_advice, adjust_solver_ranges

POLL_INTERVAL = 1.0  # seconds


def _capture_settled():
    """Find table and capture settled frame."""
    try:
        windows = find_target_windows()
    except RuntimeError:
        return None, ""
    expanded = [w for w in windows if w["bounds"]["w"] >= MIN_WINDOW_WIDTH]
    if not expanded:
        return None, ""
    win = expanded[0]
    title = "{} ({}x{})".format(win["title"][:40], win["bounds"]["w"], win["bounds"]["h"])

    img1 = capture_window(win["id"])
    if img1 is None:
        return None, title
    time.sleep(0.12)
    img2 = capture_window(win["id"])
    if img2 is None or img1.shape != img2.shape:
        return None, title
    diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
    if diff > 5.0:
        return None, title
    return img2, title


def _build_display(img, tracker, smoother, engine, rl, name_cache, hh_watcher=None):
    """Process frame and return formatted text string.

    name_cache: dict of seat -> locked name, persists across frames.
    Resets when hand_id changes.
    hh_watcher: HHWatcher instance for player stats (optional).
    """
    t0 = time.time()
    gs = process_screenshot(img)
    ocr_ms = (time.time() - t0) * 1000

    # Capture raw OCR reads before smoothing
    raw_hero = list(gs.hero_cards) if gs.hero_cards else []
    raw_board = list(gs.board) if gs.board else []

    aa = gs.available_actions or {}
    hero_has_action = bool(aa and (
        aa.get("fold") or aa.get("check") or aa.get("call") is not None))

    # When hero doesn't have action, view is small — suppress unreliable hero reads
    if not hero_has_action and not smoother._hero_locked:
        gs.hero_cards = None

    smoother.update(gs, hero_has_action=hero_has_action)

    # Stabilize player names: lock first non-empty read per seat per hand
    hand_id = gs.hand_id
    if name_cache.get("_hand_id") != hand_id:
        name_cache.clear()
        name_cache["_hand_id"] = hand_id
    for p in gs.players:
        if p.name and p.seat not in name_cache:
            name_cache[p.seat] = p.name
        elif p.seat in name_cache:
            p.name = name_cache[p.seat]

    positions = gs.infer_positions()
    tracker.update(gs)
    preflop = tracker.preflop_action

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
            # Exploitative: adjust villain range based on archetype
            if hh_watcher:
                for p in gs.players:
                    if p.is_sitting_out or p.is_folded or p.is_hero:
                        continue
                    if p.name:
                        vs = hh_watcher.get_player_stats(p.name)
                        solver_inputs = adjust_solver_ranges(solver_inputs, vs)
                        break  # adjust for primary villain only

            # Only solve turn/river — flop is too slow for live play
            if board_len >= 4:
                bet_sizes = tracker.get_bet_sizes()
                street_actions = tracker.get_street_actions(gs, solver_inputs)
                engine.request(solver_inputs, gs.hero_cards, hero_pos, gs.board,
                               bet_sizes=bet_sizes, street_actions=street_actions)

    result = engine.get_result()

    # Format output
    lines = []
    status = engine.status

    # ── Header: hero info ──
    hero_str = " ".join(c if c else "??" for c in gs.hero_cards) if gs.hero_cards else "--"
    hero_pos_str = ""
    if solver_inputs and solver_inputs.get("hero_position"):
        hero_pos_str = solver_inputs["hero_position"].upper()
    else:
        for p in gs.players:
            if p.is_hero and p.seat in positions:
                hero_pos_str = positions[p.seat]
                break

    street = gs.street.upper() if gs.street else "?"
    board_str = " ".join(c if c else "??" for c in gs.board) if gs.board else "--"
    pot = gs.total_bb or gs.pot_bb
    pot_str = "{:.1f}".format(pot) if pot else "?"

    if hero_pos_str:
        lines.append("{} ({})  |  Pot {} BB".format(hero_str, hero_pos_str, pot_str))
    else:
        lines.append("{}  |  Pot {} BB".format(hero_str, pot_str))
    lines.append("{} | {}".format(street, board_str))

    # ── Strategy (the main event) ──
    lines.append("")
    if gs.street == "preflop" and hero_has_action:
        # Preflop chart lookup — big and clear
        hero_pos_name = None
        for p in gs.players:
            if p.is_hero and p.seat in positions:
                hero_pos_name = positions[p.seat]
                break
        advice = None
        if hero_pos_name and gs.hero_cards:
            advice = preflop_advice(gs.hero_cards, hero_pos_name, gs, rl)

        # Exploitative adjustment: find the main villain (opener or 3bettor)
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
                advice = adjust_advice(advice, vs)

        if advice:
            lines.append("\u2550" * 38)
            lines.append("  \u25b6  {}".format(advice))
            lines.append("\u2550" * 38)
        else:
            lines.append("  (waiting for preflop data)")
    elif result:
        lines.append("EV {:.2f} BB  |  Equity {:.0f}%".format(
            result["ev"], result["equity"] * 100))
        lines.append("")
        for a in result["actions"]:
            freq = a["frequency"]
            if freq < 0.005:
                continue
            pct = int(freq * 100)
            bar = "\u2588" * max(1, pct // 5)
            lines.append("  {:>3d}%  {}  {}".format(pct, bar, a["action"]))
    elif status == "running":
        lines.append("  Solving...")
    elif status == "error":
        lines.append("  Engine: {}".format(engine.last_error[:60]))
    elif gs.street == "flop":
        lines.append("  (flop — no live solve)")
    elif solver_err:
        lines.append("  {}".format(solver_err[:50]))
    else:
        lines.append("  (idle)")

    # ── Actions ──
    act_parts = []
    if aa:
        if aa.get("fold"):
            act_parts.append("Fold")
        if aa.get("check"):
            act_parts.append("Check")
        if aa.get("call") is not None:
            act_parts.append("Call {:.1f}".format(aa["call"]))
        if aa.get("raise_to") is not None:
            act_parts.append("Raise {:.1f}".format(aa["raise_to"]))
        if aa.get("bet") is not None:
            act_parts.append("Bet {:.1f}".format(aa["bet"]))
    if act_parts:
        lines.append("")
        lines.append(" | ".join(act_parts))

    # ── Players ──
    lines.append("")
    lines.append("\u2500" * 38)
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
        hero_mark = " \u25c0" if p.is_hero else ""

        # Player stats from hand history
        stat_str = ""
        if hh_watcher and p.name and not p.is_hero:
            ps = hh_watcher.get_player_stats(p.name)
            if ps.hands >= 5:
                stat_str = " {:.0f}/{:.0f} {}".format(
                    ps.vpip, ps.pfr, ps.archetype)

        lines.append("{:<4} {:<10} {:>6} {:>5} {}{}{}".format(
            pos, name, stack, bet, action, hero_mark, stat_str))

    # ── Debug (compact, at bottom) ──
    lines.append("")
    lines.append("\u2500 debug \u2500")
    raw_h = " ".join(c if c else "??" for c in raw_hero) if raw_hero else "--"
    locked_h = " ".join(smoother._hero_locked.get(i, "?") for i in range(2))
    hero_ocr_on = hero_has_action or bool(smoother._hero_locked)
    lines.append("raw:{} lock:{} ocr:{}".format(raw_h, locked_h,
        "ON" if hero_ocr_on else "OFF"))
    parts = ["{}ms".format(int(ocr_ms)), status]
    if status == "done" and engine.last_solve_time:
        parts[-1] = "done({:.1f}s)".format(engine.last_solve_time)
    parts.append("hand:{}".format((gs.hand_id or "?")[-6:]))
    if hh_watcher:
        parts.append("hh:{}".format(hh_watcher.total_hands))
    if solver_err:
        parts.append(solver_err[:40])
    lines.append(" ".join(parts))

    return "\n".join(lines)


class OverlayDelegate(AppKit.NSObject):
    """App delegate that creates the overlay window and runs the poll loop."""

    def applicationDidFinishLaunching_(self, notification):
        # Create window
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(20, 20, 420, 340), style, NSBackingStoreBuffered, False)
        self.window.setTitle_("TBL Monitor")
        # Normal window level (not always on top)
        self.window.setLevel_(0)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.1, 0.1, 0.18, 0.95))

        # Scroll view + text view
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 420, 340))
        scroll.setHasVerticalScroller_(True)

        self.text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 420, 340))
        self.text_view.setEditable_(False)
        self.text_view.setSelectable_(True)
        self.text_view.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.1, 0.1, 0.18, 1.0))
        self.text_view.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.88, 0.88, 0.88, 1.0))
        self.text_view.setFont_(NSFont.fontWithName_size_("Menlo", 12))

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
        self.last_text = ""

        # Start background capture thread
        self._text_queue = []
        self._queue_lock = threading.Lock()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

        # Poll for updates from background thread
        self._start_timer()

    def _start_timer(self):
        AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.3, self, objc.selector(self.onTimer_, signature=b'v@:@'), None, True)

    @objc.python_method
    def _worker(self):
        name_cache = {}
        while True:
            try:
                img, title = _capture_settled()
                if img is None:
                    text = "Waiting for table...\n{}".format(title) if title else "Waiting for table..."
                else:
                    text = _build_display(img, self.tracker, self.smoother, self.engine, self.rl, name_cache, self.hh_watcher)

                with self._queue_lock:
                    self._text_queue.append(text)
            except Exception as e:
                with self._queue_lock:
                    self._text_queue.append("Error: {}".format(str(e)[:100]))
            time.sleep(POLL_INTERVAL)

    def onTimer_(self, timer):
        with self._queue_lock:
            if not self._text_queue:
                return
            text = self._text_queue[-1]
            self._text_queue.clear()

        if text == self.last_text:
            return
        self.last_text = text

        self.text_view.setString_(text)


def main():
    app = NSApplication.sharedApplication()
    delegate = OverlayDelegate.alloc().init()
    app.setDelegate_(delegate)
    # Don't show in dock
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)
    app.run()


if __name__ == "__main__":
    main()
