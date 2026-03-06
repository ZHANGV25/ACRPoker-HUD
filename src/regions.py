"""Region definitions for ACR Poker 6-max table layout.

All coordinates are defined as ratios (0.0 - 1.0) of the window dimensions,
so they work at any resolution. Multiply by window width/height to get pixels.

Calibrated from ACR Poker BB mode screenshots (~1590x1190).
"""

from dataclasses import dataclass


@dataclass
class Region:
    """A rectangular region defined as ratios of window dimensions."""
    x: float  # left edge ratio
    y: float  # top edge ratio
    w: float  # width ratio
    h: float  # height ratio

    def to_pixels(self, img_w: int, img_h: int) -> tuple:
        """Convert to pixel coordinates: (x1, y1, x2, y2)."""
        x1 = int(self.x * img_w)
        y1 = int(self.y * img_h)
        x2 = int((self.x + self.w) * img_w)
        y2 = int((self.y + self.h) * img_h)
        return x1, y1, x2, y2

    def crop(self, img):
        """Crop this region from an image (numpy array)."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.to_pixels(w, h)
        return img[y1:y2, x1:x2]


# -- Title bar (stakes, table name in window title) --
TITLE_BAR = Region(0.0, 0.0, 0.45, 0.04)

# -- Hand ID "Current:XXXXXXXXXX" --
HAND_ID = Region(0.005, 0.05, 0.16, 0.025)

# -- Pot total "Total : X BB" (above board cards, center) --
POT_TOTAL = Region(0.36, 0.39, 0.20, 0.025)

# -- Committed pot "Pot X BB" (below board cards) --
POT_COMMITTED = Region(0.37, 0.56, 0.13, 0.03)

# -- Board cards (community cards area) --
BOARD_CARDS = Region(0.28, 0.42, 0.30, 0.16)

# -- Individual board card slots (white card faces) --
BOARD_CARD_1 = Region(0.29, 0.42, 0.085, 0.155)
BOARD_CARD_2 = Region(0.375, 0.42, 0.085, 0.155)
BOARD_CARD_3 = Region(0.458, 0.42, 0.085, 0.155)
BOARD_CARD_4 = Region(0.543, 0.42, 0.085, 0.155)
BOARD_CARD_5 = Region(0.628, 0.42, 0.085, 0.155)
BOARD_CARD_SLOTS = [BOARD_CARD_1, BOARD_CARD_2, BOARD_CARD_3, BOARD_CARD_4, BOARD_CARD_5]

# -- 6-max seat layout --
# Action labels appear directly above each player's name box.

# Seat 1: Top-left (e.g. Burgboy)
SEAT1_NAME = Region(0.08, 0.28, 0.16, 0.03)
SEAT1_STACK = Region(0.10, 0.32, 0.10, 0.03)
SEAT1_ACTION = Region(0.09, 0.24, 0.07, 0.04)
SEAT1_BET = Region(0.16, 0.36, 0.08, 0.03)

# Seat 2: Top-center (e.g. SITTING OUT / facey01)
SEAT2_NAME = Region(0.40, 0.20, 0.14, 0.025)
SEAT2_STACK = Region(0.44, 0.22, 0.10, 0.025)
SEAT2_ACTION = Region(0.44, 0.175, 0.06, 0.025)
SEAT2_BET = Region(0.44, 0.26, 0.08, 0.03)

# Seat 3: Top-right (e.g. yagizongi)
SEAT3_NAME = Region(0.82, 0.26, 0.14, 0.03)
SEAT3_STACK = Region(0.81, 0.32, 0.10, 0.03)
SEAT3_ACTION = Region(0.84, 0.22, 0.08, 0.04)
SEAT3_BET = Region(0.73, 0.29, 0.07, 0.04)

# Seat 4: Bottom-right (e.g. AJHPoker)
SEAT4_NAME = Region(0.70, 0.73, 0.17, 0.03)
SEAT4_STACK = Region(0.73, 0.76, 0.10, 0.03)
SEAT4_ACTION = Region(0.76, 0.70, 0.07, 0.03)
SEAT4_BET = Region(0.63, 0.60, 0.08, 0.03)

# Seat 5: Bottom-center (HERO - e.g. vortexted)
SEAT5_NAME = Region(0.38, 0.84, 0.14, 0.025)
SEAT5_STACK = Region(0.47, 0.86, 0.08, 0.03)
SEAT5_ACTION = Region(0.42, 0.80, 0.06, 0.03)
SEAT5_BET = Region(0.44, 0.70, 0.08, 0.03)

# Seat 6: Bottom-left (e.g. cmooraces)
SEAT6_NAME = Region(0.10, 0.73, 0.17, 0.03)
SEAT6_STACK = Region(0.12, 0.77, 0.10, 0.03)
SEAT6_ACTION = Region(0.18, 0.70, 0.05, 0.03)
SEAT6_BET = Region(0.22, 0.60, 0.08, 0.03)

SEATS = {
    1: {"name": SEAT1_NAME, "stack": SEAT1_STACK, "action": SEAT1_ACTION, "bet": SEAT1_BET},
    2: {"name": SEAT2_NAME, "stack": SEAT2_STACK, "action": SEAT2_ACTION, "bet": SEAT2_BET},
    3: {"name": SEAT3_NAME, "stack": SEAT3_STACK, "action": SEAT3_ACTION, "bet": SEAT3_BET},
    4: {"name": SEAT4_NAME, "stack": SEAT4_STACK, "action": SEAT4_ACTION, "bet": SEAT4_BET},
    5: {"name": SEAT5_NAME, "stack": SEAT5_STACK, "action": SEAT5_ACTION, "bet": SEAT5_BET},
    6: {"name": SEAT6_NAME, "stack": SEAT6_STACK, "action": SEAT6_ACTION, "bet": SEAT6_BET},
}

HERO_SEAT = 5

# -- Hero's hole cards (combined area for both cards) --
HERO_CARDS = Region(0.42, 0.70, 0.14, 0.14)

# -- Dealer button (search area - covers the full table felt where the button can appear) --
DEALER_BUTTON = Region(0.05, 0.15, 0.90, 0.70)

# -- Action buttons (bottom of screen) --
FOLD_BUTTON = Region(0.44, 0.93, 0.14, 0.05)
CALL_CHECK_BUTTON = Region(0.60, 0.93, 0.16, 0.05)
RAISE_BET_BUTTON = Region(0.82, 0.93, 0.14, 0.05)

# -- Bet sizing presets (Min/1/4, 1/2, 3/4, Pot, All-In) --
SIZING_PRESETS = Region(0.63, 0.85, 0.30, 0.03)

# -- Hand strength text (bottom right, e.g. "Jack high") --
HAND_STRENGTH = Region(0.91, 0.80, 0.08, 0.03)
