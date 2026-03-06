"""Card identification for ACR Poker — rank and suit detection.

Uses template matching for rank identification (most reliable for fixed-font game client)
and color/shape analysis for suit detection.
"""

from typing import Optional, List, Tuple
import os
import cv2
import numpy as np

RANK_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "templates", "card_ranks")

# Loaded templates: rank_char -> binary image (white on black, 40px height)
_templates = None


def _load_templates() -> dict:
    """Load card rank templates from disk."""
    global _templates
    if _templates is not None:
        return _templates
    _templates = {}
    if not os.path.exists(RANK_TEMPLATE_DIR):
        return _templates
    for f in os.listdir(RANK_TEMPLATE_DIR):
        if not f.endswith(".png"):
            continue
        rank = f.replace(".png", "")
        img = cv2.imread(os.path.join(RANK_TEMPLATE_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _templates[rank] = img
    return _templates


def _match_rank(char_img: np.ndarray) -> Tuple[str, float]:
    """Match a binary character image against rank templates.

    Args:
        char_img: Binary image (white on black), normalized to 40px height.

    Returns:
        (rank, confidence) where rank is '2'-'9','T','J','Q','K','A'.
    """
    templates = _load_templates()
    if not templates:
        return '?', 0.0

    target_h = 40
    best_rank = '?'
    best_score = -1.0

    for rank, tmpl in templates.items():
        max_w = max(char_img.shape[1], tmpl.shape[1])
        pad1 = np.zeros((target_h, max_w), dtype=np.uint8)
        pad2 = np.zeros((target_h, max_w), dtype=np.uint8)
        o1 = (max_w - char_img.shape[1]) // 2
        o2 = (max_w - tmpl.shape[1]) // 2
        pad1[:, o1:o1 + char_img.shape[1]] = char_img
        pad2[:, o2:o2 + tmpl.shape[1]] = tmpl

        if pad1.std() == 0 or pad2.std() == 0:
            continue
        score = cv2.matchTemplate(pad1, pad2, cv2.TM_CCOEFF_NORMED).max()
        if score > best_score:
            best_score = score
            best_rank = rank

    return best_rank, best_score


def _extract_char_from_roi(roi: np.ndarray) -> Optional[np.ndarray]:
    """Extract the rank character as a normalized binary image from a card corner ROI.

    Uses HSV-based detection to handle both black and red rank text on white cards.

    Args:
        roi: BGR image of the card's top-left corner region.

    Returns:
        Binary image (white on black), 40px height, or None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(int)  # saturation (high for colored text)
    v = hsv[:, :, 2].astype(int)  # value (low for dark text)

    # Non-white pixels: dark (low value) OR highly colored (high saturation)
    text_mask = ((v < 160) | (s > 60)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if h < 5 or w < 3:
        return None

    char_img = text_mask[y:y + h, x:x + w]

    # Normalize to 40px height
    target_h = 40
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(char_img, (new_w, target_h), interpolation=cv2.INTER_NEAREST)


def _find_face_start(card_img: np.ndarray) -> Tuple[int, int]:
    """Find where the white card face starts (x, y offset).

    Returns (face_x, face_y).
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape

    # Sample row at 15% height
    sample_row = min(max(int(ch * 0.15), 1), ch - 1)
    white_cols = np.where(gray[sample_row, :] > 200)[0]
    face_x = int(white_cols[0]) if len(white_cols) > 0 else 0

    center_col = min(max(cw // 2, 0), cw - 1)
    white_rows = np.where(gray[:, center_col] > 200)[0]
    face_y = int(white_rows[0]) if len(white_rows) > 0 else 0

    return face_x, face_y


def _read_rank(card_img: np.ndarray) -> Optional[str]:
    """Read the rank from a single card image using template matching.

    Args:
        card_img: BGR image of one card (may include dark border).

    Returns:
        Standard rank: '2'-'9','T','J','Q','K','A', or None.
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)

    # Rank area: top-left of the card face
    rx1 = face_x + 2
    ry1 = face_y + 2
    rx2 = min(rx1 + max(int(cw * 0.35), 45), cw)
    ry2 = min(ry1 + max(int(ch * 0.30), 42), ch)

    rank_roi = card_img[ry1:ry2, rx1:rx2]
    if rank_roi.size == 0:
        return None

    char_norm = _extract_char_from_roi(rank_roi)
    if char_norm is None:
        return None

    rank, score = _match_rank(char_norm)
    if score < 0.5:
        return None

    # Disambiguate 6/9 if needed
    if rank in ('6', '9') and score < 0.95:
        rank = _disambiguate_6_9(rank_roi)

    return rank


def _disambiguate_6_9(rank_roi: np.ndarray) -> str:
    """Distinguish 6 from 9 by ink distribution (9 has bulk at top)."""
    gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    rh = mask.shape[0]
    top_ink = cv2.countNonZero(mask[:rh // 2, :])
    bottom_ink = cv2.countNonZero(mask[rh // 2:, :])

    return '9' if top_ink > bottom_ink else '6'


def _detect_suit(card_img: np.ndarray) -> str:
    """Detect the suit of a card using color and shape analysis.

    Returns: 'h', 'd', 'c', or 's'.
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)

    face_w = cw - face_x
    face_h = ch - face_y

    # Suit symbol: top-right area of card face
    sx1 = face_x + max(int(face_w * 0.50), 1)
    sx2 = min(face_x + int(face_w * 0.95), cw)
    sy1 = face_y + 2
    sy2 = min(face_y + max(int(face_h * 0.30), 10), ch)

    suit_roi = card_img[sy1:sy2, sx1:sx2]
    if suit_roi.size == 0:
        return 's'

    # Color detection
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_px = int(cv2.countNonZero(red1) + cv2.countNonZero(red2))

    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 120, 120]))
    dark_px = int(cv2.countNonZero(dark))

    is_red = red_px > dark_px and red_px > suit_roi.shape[0] * suit_roi.shape[1] * 0.03

    # Shape analysis for specific suit
    gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    _, suit_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    sh, sw = suit_mask.shape

    if sh < 5 or sw < 5:
        return 'h' if is_red else 's'

    # Width profile: compare width at top vs middle
    def _width_at(pct):
        row = min(int(sh * pct), sh - 1)
        nz = np.where(suit_mask[row, :] > 0)[0]
        return int(nz[-1] - nz[0]) if len(nz) >= 2 else 0

    w_top = _width_at(0.15)
    w_mid = _width_at(0.50)

    if w_mid == 0:
        return 'h' if is_red else 's'

    ratio = w_top / w_mid

    if is_red:
        # Hearts: wide at top (two humps), ratio > 0.6
        # Diamonds: narrow at top (point), ratio < 0.6
        return 'h' if ratio > 0.6 else 'd'
    else:
        # Clubs have 3 separate lobes that appear as separate pixel groups
        # at ~35-45% height. Spades have a smooth continuous outline.
        for pct in [0.35, 0.40, 0.45]:
            row = min(int(sh * pct), sh - 1)
            nz = np.where(suit_mask[row, :] > 0)[0]
            if len(nz) >= 2:
                groups = 1
                for j in range(1, len(nz)):
                    if nz[j] - nz[j - 1] > 2:
                        groups += 1
                if groups >= 3:
                    return 'c'
        return 's'


def identify_card(card_img: np.ndarray) -> Optional[str]:
    """Identify a single card from its image.

    Args:
        card_img: BGR image of a single card.

    Returns:
        Card string like '2h', '9c', 'Ts', 'Ad', or None.
    """
    rank = _read_rank(card_img)
    if rank is None:
        return None
    suit = _detect_suit(card_img)
    return rank + suit


def _find_merged_card_area(region_img: np.ndarray,
                           min_area: int = 1000) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of the merged white card area.

    Returns (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 55, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    return cv2.boundingRect(largest)


def detect_and_identify_board(board_img: np.ndarray) -> List[Optional[str]]:
    """Detect and identify community cards on the board.

    Args:
        board_img: BGR image of the board area (BOARD_CARDS region crop).

    Returns:
        List of card strings. Empty if no cards detected.
    """
    rect = _find_merged_card_area(board_img, min_area=5000)
    if rect is None:
        return []

    x, y, w, h = rect

    # Estimate number of cards from aspect ratio
    # Single card aspect ~0.65-0.85. Merged cards: width / (height * single_aspect)
    single_card_aspect = 0.75
    expected_card_w = h * single_card_aspect
    num_cards = max(1, round(w / expected_card_w))
    num_cards = min(num_cards, 5)  # Max 5 community cards

    card_width = w // num_cards
    cards = []

    for i in range(num_cards):
        cx1 = x + i * card_width
        cx2 = x + (i + 1) * card_width
        card_img = board_img[y:y + h, cx1:cx2]
        card_str = identify_card(card_img)
        cards.append(card_str)

    return cards


def detect_and_identify_hero(hero_img: np.ndarray) -> List[Optional[str]]:
    """Detect and identify hero's hole cards.

    Args:
        hero_img: BGR image of the hero card area.

    Returns:
        List of 0-2 card strings.
    """
    rect = _find_merged_card_area(hero_img, min_area=500)
    if rect is None:
        return []

    x, y, w, h = rect

    # Hero cards: 2 cards, slightly overlapping
    # Split at ~48% for cleaner separation (card 1 is in front)
    if w < h * 0.8:
        # Looks like a single card
        card_img = hero_img[y:y + h, x:x + w]
        result = identify_card(card_img)
        return [result] if result else []

    # Two cards
    split = int(w * 0.48)
    cards = []

    # Card 1: left portion
    c1 = hero_img[y:y + h, x:x + split]
    cards.append(identify_card(c1))

    # Card 2: right portion — rank is in its own top-left corner
    # The card 2 starts where card 1 ends, but its white face
    # extends from about the split point to x+w
    c2 = hero_img[y:y + h, x + split:x + w]
    # For card 2, the rank character may be further right due to overlap
    # Try to identify normally first
    c2_result = identify_card(c2)
    if c2_result is None:
        # Try extracting rank from the visible top portion with adjusted ROI
        c2_result = _identify_hero_card2(hero_img, x, y, w, h, split)
    cards.append(c2_result)

    return cards


def detect_dealer_button(img: np.ndarray) -> Optional[int]:
    """Detect the dealer button and return which seat (1-6) it belongs to.

    The ACR dealer button is a small white/bright circle with a dark "D" on it,
    positioned on the table felt between a player's seat and the center of the table.

    Detection approach:
      1. Threshold for bright pixels (the button face is white, gray > 190).
      2. Morphological close to fill the dark "D" letter gap.
      3. Find contours that are roughly circular (circularity > 0.55, aspect 0.6-1.6).
      4. Verify each candidate has the "D" letter pattern: a mix of bright face pixels
         (> 30%) and dark text/border pixels (5-50%).
      5. Pick the best candidate (highest circularity, then largest area).
      6. Map the button center to the nearest of the 6 seat positions.

    Args:
        img: BGR image of the full table screenshot.

    Returns:
        Seat number (1-6) that has the dealer button, or None if not found.
    """
    ih, iw = img.shape[:2]

    # Search the full table area (wider than the DEALER_BUTTON region in regions.py,
    # since the button can appear near any of the 6 seats).
    sx1 = int(0.05 * iw)
    sy1 = int(0.15 * ih)
    sx2 = int(0.95 * iw)
    sy2 = int(0.85 * ih)

    search = img[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

    # Step 1: Threshold for bright pixels (button face is white)
    _, bright = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    # Step 2: Morphological close to bridge the dark "D" letter inside the button
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: Find circular contours in the right size range
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # Button is roughly 30-50px diameter at 1590x1190 -> area 700-2000
        # Scale-adaptive: use image diagonal as reference
        diag = (iw ** 2 + ih ** 2) ** 0.5
        min_area = max(200, int((diag / 2000) ** 2 * 400))
        max_area = max(800, int((diag / 2000) ** 2 * 2500))
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        if bh == 0:
            continue
        aspect = bw / bh
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if 0.6 < aspect < 1.6 and circularity > 0.55:
            abs_x = bx + sx1
            abs_y = by + sy1
            candidates.append({
                'area': area,
                'abs_x': abs_x,
                'abs_y': abs_y,
                'w': bw,
                'h': bh,
                'circularity': circularity,
            })

    # Step 4: Verify each candidate has the "D" letter pattern
    verified = []
    for cd in candidates:
        margin = 3
        rx1 = max(0, cd['abs_x'] - margin)
        ry1 = max(0, cd['abs_y'] - margin)
        rx2 = min(iw, cd['abs_x'] + cd['w'] + margin)
        ry2 = min(ih, cd['abs_y'] + cd['h'] + margin)

        roi_gray = cv2.cvtColor(img[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        if roi_gray.size == 0:
            continue

        bright_pct = np.sum(roi_gray > 180) / roi_gray.size
        dark_pct = np.sum(roi_gray < 120) / roi_gray.size

        # The button face should be mostly bright with some dark text
        if bright_pct > 0.30 and 0.05 < dark_pct < 0.50:
            verified.append(cd)

    if not verified:
        return None

    # Step 5: Pick the best candidate (highest circularity, then largest area)
    best = max(verified, key=lambda c: (c['circularity'], c['area']))

    # Step 6: Map button center to nearest seat
    btn_cx = (best['abs_x'] + best['w'] / 2) / iw
    btn_cy = (best['abs_y'] + best['h'] / 2) / ih

    # Seat reference points (approximate center of each player's avatar/name area)
    seat_refs = {
        1: (0.14, 0.24),   # top-left
        2: (0.47, 0.17),   # top-center
        3: (0.89, 0.24),   # top-right
        4: (0.80, 0.73),   # bottom-right
        5: (0.47, 0.86),   # bottom-center (hero)
        6: (0.15, 0.73),   # bottom-left
    }

    nearest_seat = min(
        seat_refs,
        key=lambda s: (btn_cx - seat_refs[s][0]) ** 2 + (btn_cy - seat_refs[s][1]) ** 2
    )

    return nearest_seat


def _identify_hero_card2(hero_img: np.ndarray, x: int, y: int,
                          w: int, h: int, split: int) -> Optional[str]:
    """Fallback identification for hero card 2 (often partially occluded).

    Looks for rank character contours in the right portion of the merged area.
    """
    # Search in the top half of the right portion
    c2_top = hero_img[y:y + h // 2, x + split:x + w]
    if c2_top.size == 0:
        return None

    gray = cv2.cvtColor(c2_top, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find contours that look like rank characters: taller than wide, reasonable size
    rank_candidates = []
    for c in contours:
        cx, cy, cw, ch_c = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if ch_c > 10 and cw < ch_c * 1.5 and area > 20:
            rank_candidates.append((cx, cy, cw, ch_c, area))

    if not rank_candidates:
        return None

    # Pick the tallest character that's not too far right (should be near left of card 2)
    rank_candidates.sort(key=lambda r: r[3], reverse=True)

    for cx, cy, cw, ch_c, area in rank_candidates:
        char_img = mask[cy:cy + ch_c, cx:cx + cw]
        target_h = 40
        scale = target_h / ch_c
        new_w = max(1, int(cw * scale))
        char_norm = cv2.resize(char_img, (new_w, target_h),
                               interpolation=cv2.INTER_NEAREST)

        rank, score = _match_rank(char_norm)
        if score > 0.5:
            # Detect suit from card 2 area
            c2_full = hero_img[y:y + h, x + split:x + w]
            suit = _detect_suit(c2_full)
            return rank + suit

    return None
