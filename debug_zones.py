"""Draw all region zones on a screenshot for visual debugging."""
import cv2
import numpy as np
import sys

from src.regions import (
    SEATS, TITLE_BAR, HAND_ID, POT_TOTAL, POT_COMMITTED, BOARD_CARDS,
    HERO_CARDS, FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON,
    HAND_STRENGTH, BOARD_CARD_SLOTS, extract_table_area,
)

COLORS = {
    "name": (0, 255, 0),      # green
    "stack": (0, 255, 255),    # yellow
    "action": (255, 0, 255),   # magenta
    "bet": (255, 128, 0),      # orange
}

def draw_region(img, region, color, label=""):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region.to_pixels(w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main(path):
    img = cv2.imread(path)
    img = extract_table_area(img)
    overlay = img.copy()

    # Seats
    for seat_num, regions in SEATS.items():
        for key, region in regions.items():
            color = COLORS.get(key, (255, 255, 255))
            draw_region(overlay, region, color, f"S{seat_num} {key}")

    # Global regions
    draw_region(overlay, TITLE_BAR, (255, 255, 255), "TITLE")
    draw_region(overlay, HAND_ID, (200, 200, 200), "HAND_ID")
    draw_region(overlay, POT_TOTAL, (0, 200, 255), "POT_TOTAL")
    draw_region(overlay, POT_COMMITTED, (0, 200, 255), "POT_COMMITTED")
    draw_region(overlay, BOARD_CARDS, (255, 200, 0), "BOARD")
    draw_region(overlay, HERO_CARDS, (0, 0, 255), "HERO_CARDS")
    draw_region(overlay, FOLD_BUTTON, (0, 0, 200), "FOLD")
    draw_region(overlay, CALL_CHECK_BUTTON, (0, 0, 200), "CALL/CHECK")
    draw_region(overlay, RAISE_BET_BUTTON, (0, 0, 200), "RAISE/BET")
    draw_region(overlay, HAND_STRENGTH, (200, 200, 0), "HAND_STR")

    for i, slot in enumerate(BOARD_CARD_SLOTS):
        draw_region(overlay, slot, (100, 255, 100), f"BC{i+1}")

    out = path.replace(".png", "_zones.png")
    cv2.imwrite(out, overlay)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main(sys.argv[1])
