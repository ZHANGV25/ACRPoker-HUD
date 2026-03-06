"""Live window capture loop for ACR Poker OCR pipeline.

Polls for ACR Poker table windows, detects when it is the hero's turn
(action buttons visible), runs the full OCR pipeline, and prints the
resulting game state as JSON.

Usage:
    python -m src.live
"""

import signal
import sys
import time

import cv2
import numpy as np

from src.capture import find_acr_windows, capture_window
from src.pipeline import process_screenshot
from src.regions import FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL = 0.5        # seconds between capture polls
COOLDOWN_INTERVAL = 0.3    # seconds between checks while waiting for buttons to disappear
BUTTON_BRIGHTNESS_THRESH = 60  # min mean brightness to consider a button region "active"
BUTTON_SATURATION_THRESH = 30  # min mean saturation for coloured button detection

# ---------------------------------------------------------------------------
# Lightweight button detection (avoids full OCR on every poll)
# ---------------------------------------------------------------------------


def _region_is_active(img: np.ndarray, region) -> bool:
    """Check if a button region contains a visible button.

    ACR action buttons are brightly coloured rectangles (green / blue / red).
    When no buttons are shown the region is dark table felt.  A simple
    brightness + saturation check is enough as a pre-filter.
    """
    crop = region.crop(img)
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))

    return mean_v > BUTTON_BRIGHTNESS_THRESH and mean_s > BUTTON_SATURATION_THRESH


def buttons_visible(img: np.ndarray) -> bool:
    """Return True if at least two of the three action button regions look active.

    Requiring two out of three guards against false positives from stray UI
    elements while still being tolerant of partial occlusion.
    """
    active_count = sum(
        _region_is_active(img, region)
        for region in (FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON)
    )
    return active_count >= 2


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


def run(window_index: int = 0) -> None:
    """Run the live capture loop.

    Args:
        window_index: Which ACR window to track when multiple are found
                      (0 = first window discovered).
    """
    global _running
    _running = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print("[live] ACR Poker live capture starting.  Press Ctrl+C to quit.")

    tracked_window_id = None
    tracked_window_title = None

    while _running:
        # ---- 1. Find / re-find ACR windows --------------------------------
        try:
            windows = find_acr_windows()
        except RuntimeError as exc:
            print(f"[live] Fatal: {exc}", file=sys.stderr)
            return

        if not windows:
            if tracked_window_id is not None:
                print("[live] ACR window lost.  Waiting for a new window...")
                tracked_window_id = None
                tracked_window_title = None
            time.sleep(POLL_INTERVAL)
            continue

        # Pick the requested window (clamp index to range)
        idx = min(window_index, len(windows) - 1)
        win = windows[idx]

        if win["id"] != tracked_window_id:
            tracked_window_id = win["id"]
            tracked_window_title = win["title"]
            print(
                f"[live] Tracking window [{tracked_window_id}] "
                f"{tracked_window_title!r} "
                f"({win['bounds']['w']}x{win['bounds']['h']})"
            )

        # ---- 2. Capture and check for action buttons ----------------------
        img = capture_window(tracked_window_id)
        if img is None:
            # Window may have closed between listing and capture
            tracked_window_id = None
            time.sleep(POLL_INTERVAL)
            continue

        if not buttons_visible(img):
            time.sleep(POLL_INTERVAL)
            continue

        # ---- 3. Buttons detected — run full pipeline ----------------------
        print("[live] Action buttons detected — running OCR pipeline...")
        try:
            state = process_screenshot(img)
        except Exception as exc:
            print(f"[live] Pipeline error: {exc}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)
            continue

        # ---- 4. Output game state JSON ------------------------------------
        print(state.to_json())
        sys.stdout.flush()

        # ---- 5. Wait for buttons to disappear (hero acted) ----------------
        print("[live] Waiting for action to complete...")
        while _running:
            time.sleep(COOLDOWN_INTERVAL)
            img = capture_window(tracked_window_id)
            if img is None:
                break
            if not buttons_visible(img):
                print("[live] Action completed.  Resuming polling.")
                break

    print("\n[live] Stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ACR Poker live capture — polls table window and runs OCR on hero's turn."
    )
    parser.add_argument(
        "-w", "--window-index",
        type=int,
        default=0,
        help="Index of the ACR window to track when multiple are open (default: 0).",
    )
    args = parser.parse_args()

    run(window_index=args.window_index)
