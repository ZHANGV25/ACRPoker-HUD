"""Calibration tool: draws all defined regions on a reference screenshot.

Usage:
    python calibrate.py reference_screenshots/02_preflop_hero_turn.png

Opens a window showing the screenshot with colored rectangles for each OCR region.
Press any key to close.
"""

import sys
import cv2
import numpy as np
from src.regions import (
    TITLE_BAR, HAND_ID, POT_TOTAL, POT_COMMITTED,
    BOARD_CARD_SLOTS, SEATS, HERO_CARD_1, HERO_CARD_2,
    FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON,
    SIZING_PRESETS, HAND_STRENGTH, Region,
)


COLORS = {
    "title": (255, 255, 0),       # Cyan
    "hand_id": (255, 200, 0),     # Light blue
    "pot": (0, 255, 255),         # Yellow
    "board": (0, 255, 0),         # Green
    "seat_name": (255, 100, 100), # Blue
    "seat_stack": (100, 255, 100),# Green
    "seat_action": (100, 100, 255),# Red
    "seat_bet": (0, 200, 255),    # Orange
    "hero_cards": (255, 0, 255),  # Magenta
    "buttons": (0, 165, 255),     # Orange
    "hand_str": (200, 200, 200),  # Gray
}


def draw_region(img: np.ndarray, region: Region, color: tuple, label: str = ""):
    """Draw a rectangle for a region on the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region.to_pixels(w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def calibrate(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    overlay = img.copy()

    # Title and hand ID
    draw_region(overlay, TITLE_BAR, COLORS["title"], "TITLE")
    draw_region(overlay, HAND_ID, COLORS["hand_id"], "HAND_ID")

    # Pot
    draw_region(overlay, POT_TOTAL, COLORS["pot"], "POT_TOTAL")
    draw_region(overlay, POT_COMMITTED, COLORS["pot"], "POT_COMMITTED")

    # Board cards
    for i, slot in enumerate(BOARD_CARD_SLOTS):
        draw_region(overlay, slot, COLORS["board"], f"CARD_{i+1}")

    # Seats
    for seat_num, regions in SEATS.items():
        draw_region(overlay, regions["name"], COLORS["seat_name"], f"S{seat_num}_NAME")
        draw_region(overlay, regions["stack"], COLORS["seat_stack"], f"S{seat_num}_STACK")
        draw_region(overlay, regions["action"], COLORS["seat_action"], f"S{seat_num}_ACT")
        draw_region(overlay, regions["bet"], COLORS["seat_bet"], f"S{seat_num}_BET")

    # Hero cards
    draw_region(overlay, HERO_CARD_1, COLORS["hero_cards"], "HERO_C1")
    draw_region(overlay, HERO_CARD_2, COLORS["hero_cards"], "HERO_C2")

    # Action buttons
    draw_region(overlay, FOLD_BUTTON, COLORS["buttons"], "FOLD")
    draw_region(overlay, CALL_CHECK_BUTTON, COLORS["buttons"], "CALL/CHECK")
    draw_region(overlay, RAISE_BET_BUTTON, COLORS["buttons"], "RAISE/BET")
    draw_region(overlay, SIZING_PRESETS, COLORS["buttons"], "SIZING")

    # Hand strength
    draw_region(overlay, HAND_STRENGTH, COLORS["hand_str"], "HAND_STR")

    # Save output
    out_path = image_path.replace(".png", "_calibrated.png")
    cv2.imwrite(out_path, overlay)
    print(f"Saved calibrated image to: {out_path}")

    # Try to display
    try:
        cv2.imshow("Calibration", overlay)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("(Could not open display window - check saved image instead)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <screenshot_path>")
        print("Example: python calibrate.py reference_screenshots/02_preflop_hero_turn.png")
        sys.exit(1)

    calibrate(sys.argv[1])
