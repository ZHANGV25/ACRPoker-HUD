"""Card identification — rank and suit detection.

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
    """Load card rank templates from disk.

    Supports multiple templates per rank: '2.png', '2_v2.png', etc. all
    map to rank '2'.  Returns dict mapping rank -> list of images.
    """
    global _templates
    if _templates is not None:
        return _templates
    _templates = {}
    if not os.path.exists(RANK_TEMPLATE_DIR):
        return _templates
    for f in os.listdir(RANK_TEMPLATE_DIR):
        if not f.endswith(".png"):
            continue
        # Extract rank: 'Q.png' -> 'Q', 'Q_v2.png' -> 'Q'
        base = f.replace(".png", "")
        rank = base.split("_")[0]
        img = cv2.imread(os.path.join(RANK_TEMPLATE_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if rank not in _templates:
                _templates[rank] = []
            _templates[rank].append(img)
    return _templates


def _iou_score(char_bin: np.ndarray, tmpl_bin: np.ndarray,
               target_h: int = 40, margin: int = 4) -> float:
    """Compute best IoU between a character and a template with alignment shifts."""
    char_w = char_bin.shape[1]
    tmpl_w = tmpl_bin.shape[1]

    canvas_w = max(char_w, tmpl_w) + margin * 2
    char_canvas = np.zeros((target_h, canvas_w), dtype=np.float32)
    tmpl_canvas = np.zeros((target_h, canvas_w), dtype=np.float32)
    co = (canvas_w - char_w) // 2
    to = (canvas_w - tmpl_w) // 2
    char_canvas[:, co:co + char_w] = char_bin
    tmpl_canvas[:, to:to + tmpl_w] = tmpl_bin

    best_iou = 0.0
    for shift in range(-margin, margin + 1):
        shifted = np.roll(char_canvas, shift, axis=1)
        intersection = np.sum(shifted * tmpl_canvas)
        union = np.sum(np.clip(shifted + tmpl_canvas, 0, 1))
        iou = intersection / union if union > 0 else 0
        best_iou = max(best_iou, iou)
    return best_iou


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

    best_rank = '?'
    best_score = -1.0

    char_bin = (char_img > 127).astype(np.float32)

    for rank, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            tmpl_bin = (tmpl > 127).astype(np.float32)
            iou = _iou_score(char_bin, tmpl_bin)
            if iou > best_score:
                best_score = iou
                best_rank = rank

    # J is the narrowest rank (15px). If best match is J but the extracted
    # character is wide, it's likely a misread of 6, 9, or another wide rank.
    if best_rank == 'J':
        char_cols = (char_bin > 0.5).any(axis=0)
        char_ink_w = int(char_cols.sum())
        if char_ink_w > 17:
            # Re-score against wide ranks only, pick best
            second_rank = '?'
            second_score = -1.0
            for rank in ('6', '9', '5', '3', '2', '8'):
                if rank in templates:
                    for tmpl in templates[rank]:
                        tmpl_bin = (tmpl > 127).astype(np.float32)
                        iou = _iou_score(char_bin, tmpl_bin)
                        if iou > second_score:
                            second_score = iou
                            second_rank = rank
            if second_score > 0.2:
                return second_rank, second_score

    return best_rank, best_score


def _match_rank_single(char_img: np.ndarray, rank: str) -> float:
    """Get the best IoU score for a specific rank (across all its templates)."""
    templates = _load_templates()
    if rank not in templates:
        return 0.0

    char_bin = (char_img > 127).astype(np.float32)
    best_iou = 0.0
    for tmpl in templates[rank]:
        tmpl_bin = (tmpl > 127).astype(np.float32)
        iou = _iou_score(char_bin, tmpl_bin)
        best_iou = max(best_iou, iou)
    return best_iou


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

    # A valid card rank ROI should have a white card face background.
    # Reject if there aren't enough bright pixels (i.e. no card face visible).
    bright_pct = np.sum(v > 180) / v.size
    if bright_pct < 0.15:
        return None

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


def _find_face_start(card_img: np.ndarray, search_right: bool = False) -> Tuple[int, int]:
    """Find where the white card face starts (x, y offset).

    Args:
        card_img: BGR image of a card.
        search_right: If True, search for the face in the right portion of the
            image (useful for overlapping hero card 2).

    Returns (face_x, face_y).
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape

    if search_right:
        # For overlapping card 2: sample from the right 2/3 of the image
        sample_col = min(max(int(cw * 0.70), 0), cw - 1)
        white_rows = np.where(gray[:, sample_col] > 200)[0]
        face_y = int(white_rows[0]) if len(white_rows) > 0 else 0

        sample_row = min(max(face_y + int(ch * 0.15), 1), ch - 1)
        # Find where card 2's face starts (skip card 1 bleed on left)
        right_half = gray[sample_row, cw // 2:]
        white_cols = np.where(right_half > 200)[0]
        face_x = (cw // 2 + int(white_cols[0])) if len(white_cols) > 0 else cw // 2
    else:
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

    # Rank area: top-left of the card face (allow small margin before face_x
    # to avoid clipping characters that start at the very edge)
    rx1 = max(face_x - 3, 0)
    ry1 = face_y + 2
    rx2 = min(rx1 + max(int(cw * 0.40), 50), cw)
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

    # Check for "10" (T): the "0" digit may match as 9/6/8/Q.
    # Always check Q — the "0" in "10" often scores high as Q.
    # For other similar shapes, only check when score is ambiguous.
    if rank == 'Q' or (rank in ('9', '6', 'T') and score < 0.90):
        if _has_one_digit_left(rank_roi, char_norm):
            return 'T'

    # Disambiguate Q/9: Q and 9 have similar circular shapes.
    # If Q was matched but there's no "1" digit (not a 10), check if 9 scores close.
    if rank == 'Q':
        nine_score = _match_rank_single(char_norm, '9')
        if score - nine_score < 0.08:
            # Scores are close — use ink distribution: 9 has bulk at top, Q has
            # a more vertically centered/bottom-heavy shape with a tail.
            rank = _disambiguate_Q_9(char_norm)

    # Disambiguate 6/9 if needed
    if rank in ('6', '9') and score < 0.95:
        other = '6' if rank == '9' else '9'
        other_score = _match_rank_single(char_norm, other)
        if score - other_score > 0.05:
            pass  # Template clearly favors this rank
        else:
            rank = _disambiguate_6_9(char_norm)

    # Disambiguate 5/6 if needed
    if rank in ('5', '6') and score < 0.95:
        other = '5' if rank == '6' else '6'
        other_score = _match_rank_single(char_norm, other)
        if score - other_score > 0.05:
            pass  # Template clearly favors this rank
        else:
            rank = _disambiguate_5_6(char_norm)

    return rank


def _has_one_digit_left(rank_roi: np.ndarray, char_norm: np.ndarray) -> bool:
    """Check if there's a narrow '1' digit to the left of the extracted char.

    ACR displays 10 as two characters. If _extract_char_from_roi picked up
    only the '0', this detects the '1' next to it, confirming the rank is T.
    """
    hsv = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    text_mask = ((v < 160) | (s > 60)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return False

    # Find the largest contour (the '0') and any tall narrow contour to its left
    largest = max(contours, key=cv2.contourArea)
    lx, ly, lw, lh = cv2.boundingRect(largest)

    for c in contours:
        if c is largest:
            continue
        cx, cy, cw, ch_c = cv2.boundingRect(c)
        # Must be tall enough (absolute minimum to filter tiny border fragments)
        if ch_c < 10:
            continue
        if ch_c < lh * 0.5:
            continue
        # Must be to the left of the largest contour
        if cx + cw > lx:
            continue
        # Must be narrow (like a '1')
        if cw >= ch_c * 0.5:
            continue
        # Must have sufficient fill ratio (real '1' > 0.30, border/artifact < 0.25)
        area = cv2.contourArea(c)
        if area / (cw * ch_c) < 0.30:
            continue
        # Must be close to the '0' (gap < 40% of '0' width, and not overlapping)
        gap = lx - (cx + cw)
        if gap < 0 or gap > lw * 0.4:
            continue
        # Must be vertically aligned with the '0' (top within 25% of '0' height)
        if abs(cy - ly) > lh * 0.25:
            continue
        # The '0' should be round-ish (width > 45% of height)
        if lw < lh * 0.45:
            continue
        return True

    return False


def _disambiguate_6_9(char_img: np.ndarray) -> str:
    """Distinguish 6 from 9 by ink distribution on the extracted character.

    A '9' has its closed loop (bulk) at the top; a '6' at the bottom.

    Args:
        char_img: Binary image (white on black), 40px height.
    """
    rh = char_img.shape[0]
    top_ink = cv2.countNonZero(char_img[:rh // 2, :])
    bottom_ink = cv2.countNonZero(char_img[rh // 2:, :])

    return '9' if top_ink > bottom_ink else '6'


def _disambiguate_Q_9(char_img: np.ndarray) -> str:
    """Distinguish Q from 9 by ink distribution.

    A '9' has its closed loop at the top with a descending tail (top-heavy).
    A 'Q' has a large circular body that spans more evenly, with a tail at
    bottom-right. The key difference: 9's bottom quarter has very little ink
    (just the thin tail), while Q's bottom quarter has the tail plus the
    bottom of the circle.

    Args:
        char_img: Binary image (white on black), 40px height.
    """
    rh = char_img.shape[0]
    rw = char_img.shape[1]

    # Compare ink in top 40% vs bottom 40%
    top_ink = cv2.countNonZero(char_img[:int(rh * 0.4), :])
    bottom_ink = cv2.countNonZero(char_img[int(rh * 0.6):, :])

    # 9 is very top-heavy: top ink >> bottom ink
    # Q is more balanced: both top and bottom have significant ink
    if bottom_ink == 0:
        return '9'
    ratio = top_ink / max(bottom_ink, 1)

    # 9 typically has ratio > 2.0 (bulk at top, thin tail at bottom)
    # Q typically has ratio < 2.0 (circle extends through bottom)
    return '9' if ratio > 1.8 else 'Q'


def _disambiguate_5_6(char_img: np.ndarray) -> str:
    """Distinguish 5 from 6 by top-row ink distribution.

    '5' has a wide horizontal bar at the very top spanning most of the width.
    '6' curves from the right — the top rows are narrower.
    """
    h, w = char_img.shape[:2]
    # Look at top 20% of character
    top_rows = max(h // 5, 3)
    top = char_img[:top_rows, :]
    # Count how many columns have ink in the top portion
    ink_cols = (top > 127).any(axis=0)
    ink_span = int(ink_cols.sum())
    # '5' has a horizontal bar spanning >55% of width at top
    if ink_span > w * 0.55:
        return '5'
    return '6'


def _suit_from_rank_color(rank_roi: np.ndarray,
                          four_color: bool = False) -> Optional[str]:
    """Detect suit from the color of the rank text (4-color deck).

    In a 4-color deck the rank text is colored per suit:
    green=clubs, blue=diamonds, red=hearts, black=spades.

    Green and blue are always returned (unique to 4-color decks).
    Red ('h') and black ('s') are only returned when four_color=True,
    since in 2-color decks red=hearts/diamonds and black=clubs/spades.
    """
    hsv = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    text_px = (s > 30) | (v < 170)
    total = int(np.sum(text_px))
    if total < 10:
        return None

    green = int(np.sum((h >= 35) & (h <= 85) & (s > 40) & (v > 40) & text_px))
    blue = int(np.sum((h >= 100) & (h <= 135) & (s > 40) & (v > 40) & text_px))
    red1 = int(np.sum((h <= 10) & (s > 50) & (v > 50) & text_px))
    red2 = int(np.sum((h >= 170) & (s > 50) & (v > 50) & text_px))
    red = red1 + red2
    black = int(np.sum((v < 120) & (s < 80) & text_px))

    threshold = max(total * 0.15, 5)
    # Green/blue must dominate — relaxed for small windows where dark edges
    # inflate black count.
    if green > threshold and green > blue and green > black * 2:
        return 'c'
    if blue > threshold and blue > green and blue > black * 2:
        return 'd'
    if four_color:
        if red > threshold and red > black:
            return 'h'
        if black > threshold:
            return 's'
    return None


def _check_4color_pip(roi: np.ndarray) -> Optional[str]:
    """Check for 4-color deck suit colors (green=clubs, blue=diamonds).

    In a 4-color deck, clubs are green and diamonds are blue — colors that
    never appear in a standard 2-color deck.  If either is detected, the
    suit is unambiguous.  For red (hearts) and black (spades), returns None
    so the caller can fall back to existing logic (which works for both
    2-color and 4-color decks since hearts are always red and spades always
    black).
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Only look at non-white pixels (the colored pip, not card background)
    non_white = (s > 30) | (v < 170)
    total = int(np.sum(non_white))
    if total < 5:
        return None

    green = int(np.sum((h >= 35) & (h <= 85) & (s > 50) & (v > 50) & non_white))
    blue = int(np.sum((h >= 90) & (h <= 140) & (s > 40) & (v > 40) & non_white))
    black = int(np.sum((v < 120) & (s < 80) & non_white))

    threshold = max(total * 0.10, 3)
    # Green/blue must exceed both threshold AND black count to avoid
    # false positives from dark spade/club pixels with slight color cast.
    if green > threshold and green > blue and green > black:
        return 'c'
    if blue > threshold and blue > green and blue > black:
        return 'd'

    return None


def _detect_suit(card_img: np.ndarray) -> str:
    """Detect the suit of a card using the top-right corner pip.

    ACR card layout: rank in top-left, suit pip in top-right.

    Returns: 'h', 'd', 'c', or 's'.
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)

    face_w = cw - face_x
    face_h = ch - face_y

    # Top-right suit pip: ~55-95% of face width, 3-22% of face height
    sx1 = face_x + max(int(face_w * 0.55), 1)
    sx2 = min(face_x + int(face_w * 0.95), cw)
    sy1 = face_y + max(int(face_h * 0.03), 1)
    sy2 = min(face_y + int(face_h * 0.22), ch)

    suit_roi = card_img[sy1:sy2, sx1:sx2]
    if suit_roi.size == 0:
        return 's'

    # 4-color deck: green=clubs, blue=diamonds — check first
    fc = _check_4color_pip(suit_roi)
    if fc is not None:
        return fc

    # Color detection
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_px = int(cv2.countNonZero(red1) + cv2.countNonZero(red2))

    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 120, 120]))
    dark_px = int(cv2.countNonZero(dark))

    is_red = red_px > dark_px and red_px > suit_roi.shape[0] * suit_roi.shape[1] * 0.03

    # Shape analysis on the pip contour
    gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    _, suit_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find the largest contour (the pip symbol)
    contours, _ = cv2.findContours(suit_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid:
        return 'h' if is_red else 's'

    pip_c = max(valid, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(pip_c)
    pip_mask = suit_mask[by:by + bh, bx:bx + bw]
    ph, pw = pip_mask.shape

    if ph < 5 or pw < 5:
        return 'h' if is_red else 's'

    def _width_at(pct):
        row = min(int(ph * pct), ph - 1)
        nz = np.where(pip_mask[row, :] > 0)[0]
        return int(nz[-1] - nz[0]) if len(nz) >= 2 else 0

    if is_red:
        # Hearts: two humps at top → wide at 10%, wide at 50%
        # Diamonds: point at top → narrow at 10%, wide at 50%
        w_top = _width_at(0.10)
        w_mid = _width_at(0.50)
        if w_mid == 0:
            return 'h' if w_top > 0 else 'd'
        return 'h' if w_top / w_mid > 0.5 else 'd'
    else:
        # Clubs vs spades: two approaches for robustness
        # 1. Multiple pixel groups at 50-65% height (3+ groups = clubs)
        for pct in [0.50, 0.55, 0.60, 0.65]:
            row = min(int(ph * pct), ph - 1)
            nz = np.where(pip_mask[row, :] > 0)[0]
            if len(nz) >= 2:
                groups = 1
                for j in range(1, len(nz)):
                    if nz[j] - nz[j - 1] > 2:
                        groups += 1
                if groups >= 3:
                    return 'c'

        # 2. Waist-to-lower width ratio: clubs have a narrow waist (25-45%)
        #    then dramatic widening (50-65%) due to three lobes + stem.
        #    Spades widen monotonically. Ratio >2.0 = clubs.
        waist_widths = []
        lower_widths = []
        for pct in [0.25, 0.30, 0.35, 0.40, 0.45]:
            row = min(int(ph * pct), ph - 1)
            nz = np.where(pip_mask[row, :] > 0)[0]
            if len(nz) >= 2:
                waist_widths.append(int(nz[-1] - nz[0]))
        for pct in [0.50, 0.55, 0.60, 0.65]:
            row = min(int(ph * pct), ph - 1)
            nz = np.where(pip_mask[row, :] > 0)[0]
            if len(nz) >= 2:
                lower_widths.append(int(nz[-1] - nz[0]))
        if waist_widths and lower_widths:
            min_waist = min(waist_widths)
            max_lower = max(lower_widths)
            if min_waist > 0 and max_lower / min_waist > 2.0:
                return 'c'

        return 's'


def _pip_roi(card_img: np.ndarray, search_right: bool = False) -> np.ndarray:
    """Extract the below-rank pip area from a card image."""
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img, search_right=search_right)
    face_w = cw - face_x
    face_h = ch - face_y
    py1 = face_y + max(int(face_h * 0.25), 3)
    py2 = face_y + min(int(face_h * 0.45), ch)
    px1 = face_x
    px2 = face_x + max(int(face_w * 0.35), 5)
    roi = card_img[py1:py2, px1:px2]
    if roi.size == 0:
        return card_img[0:1, 0:1]  # tiny fallback
    return roi


def _detect_suit_hero(card_img: np.ndarray) -> str:
    """Detect suit of hero card 1 (front card, top-right pip may be cut off).

    Uses rank text color for red/black, then the below-rank pip area for
    specific suit (clubs have multiple lobes = multiple contours).
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)
    face_w = cw - face_x
    face_h = ch - face_y

    # Red vs black from rank text color
    ry2 = face_y + max(int(face_h * 0.30), 5)
    rx2 = face_x + max(int(face_w * 0.45), 5)
    rank_area = card_img[face_y:ry2, face_x:rx2]
    hsv_r = cv2.cvtColor(rank_area, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv_r, np.array([0, 80, 80]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv_r, np.array([170, 80, 80]), np.array([180, 255, 255]))
    rank_red = int(cv2.countNonZero(r1) + cv2.countNonZero(r2))
    dark_mask = cv2.inRange(hsv_r, np.array([0, 0, 0]), np.array([180, 80, 150]))
    rank_dark = int(cv2.countNonZero(dark_mask))
    is_red = rank_red > rank_dark * 0.3 and rank_red > 10

    # Below-rank pip area: 25-45% of face height, left 35% of face width
    py1 = face_y + max(int(face_h * 0.25), 3)
    py2 = face_y + min(int(face_h * 0.45), ch)
    px1 = face_x
    px2 = face_x + max(int(face_w * 0.35), 5)
    pip = card_img[py1:py2, px1:px2]
    if pip.size == 0:
        return 'h' if is_red else 's'

    # 4-color deck: green=clubs, blue=diamonds
    fc = _check_4color_pip(pip)
    if fc is not None:
        return fc

    hsv = cv2.cvtColor(pip, cv2.COLOR_BGR2HSV)

    if is_red:
        # Red suits: check pip shape for hearts vs diamonds
        m1 = cv2.inRange(hsv, np.array([0, 50, 80]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 50, 80]), np.array([180, 255, 255]))
        suit_mask = cv2.bitwise_or(m1, m2)
        contours, _ = cv2.findContours(suit_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]
        if not valid:
            return 'h'
        pip_c = max(valid, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(pip_c)
        # Diamonds are taller than wide (aspect < 1), hearts are wider
        if bh > 0 and bw / bh < 0.9:
            return 'd'
        return 'h'
    else:
        # Black suits: clubs have multiple lobes (2+ separate contours)
        dark_m = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 120, 120]))
        contours, _ = cv2.findContours(dark_m, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]
        if len(valid) >= 2:
            return 'c'
        return 's'


def _detect_suit_hero_best(card_img: np.ndarray) -> str:
    """Best-effort suit detection for hero card 1.

    Tries the top-right pip method first (_detect_suit). If the pip area
    has enough content, uses that result. Otherwise falls back to
    _detect_suit_hero (rank text color + below-rank pip).
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)
    face_w = cw - face_x
    face_h = ch - face_y

    # Check if top-right pip area has enough dark/colored content
    sx1 = face_x + max(int(face_w * 0.55), 1)
    sx2 = min(face_x + int(face_w * 0.95), cw)
    sy1 = face_y + max(int(face_h * 0.03), 1)
    sy2 = min(face_y + int(face_h * 0.22), ch)

    if sx2 > sx1 + 3 and sy2 > sy1 + 3:
        suit_roi = card_img[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 15]
        if valid:
            pip_c = max(valid, key=cv2.contourArea)
            pip_area = cv2.contourArea(pip_c)
            roi_area = suit_roi.shape[0] * suit_roi.shape[1]
            # If pip occupies >5% of ROI, the pip is sufficiently visible
            if pip_area > roi_area * 0.05:
                return _detect_suit(card_img)

    return _detect_suit_hero(card_img)


def _detect_suit_pip(card_img: np.ndarray, search_right: bool = False) -> str:
    """Detect suit from the small corner pip below the rank character.

    Used for hero cards where the center suit symbol is occluded by overlap.
    """
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img, search_right=search_right)
    face_w = cw - face_x
    face_h = ch - face_y

    # Pip is below the rank: ~28-48% of face height, left 40% of face width
    py1 = face_y + max(int(face_h * 0.28), 5)
    py2 = face_y + min(int(face_h * 0.50), ch)
    px1 = face_x + 2
    px2 = face_x + max(int(face_w * 0.42), 10)

    pip = card_img[py1:py2, px1:px2]
    if pip.size == 0:
        return 's'

    # 4-color deck: green=clubs, blue=diamonds
    fc = _check_4color_pip(pip)
    if fc is not None:
        return fc

    # Color detection (2-color fallback)
    hsv = cv2.cvtColor(pip, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_px = int(cv2.countNonZero(red1) + cv2.countNonZero(red2))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 120, 120]))
    dark_px = int(cv2.countNonZero(dark))
    total = pip.shape[0] * pip.shape[1]

    is_red = red_px > dark_px and red_px > total * 0.03

    # Shape analysis on the pip
    gray = cv2.cvtColor(pip, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    sh, sw = mask.shape

    if sh < 4 or sw < 4:
        return 'h' if is_red else 's'

    def _width_at(pct):
        row = min(int(sh * pct), sh - 1)
        nz = np.where(mask[row, :] > 0)[0]
        return int(nz[-1] - nz[0]) if len(nz) >= 2 else 0

    if is_red:
        w_top = _width_at(0.15)
        w_mid = _width_at(0.50)
        if w_mid == 0:
            return 'h'
        return 'h' if w_top / w_mid > 0.6 else 'd'
    else:
        # Clubs vs spades: use waist-to-lower width ratio
        # (more reliable than pixel groups for small pips)
        contours_p, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        valid_p = [c for c in contours_p if cv2.contourArea(c) > 3]
        if valid_p:
            pip_c = max(valid_p, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(pip_c)
            pip_mask = mask[by:by + bh, bx:bx + bw]
            ph, pw = pip_mask.shape
            if ph >= 6 and pw >= 4:
                waist_widths = []
                lower_widths = []
                for pct in [0.25, 0.30, 0.35, 0.40, 0.45]:
                    row = min(int(ph * pct), ph - 1)
                    nz = np.where(pip_mask[row, :] > 0)[0]
                    if len(nz) >= 2:
                        waist_widths.append(int(nz[-1] - nz[0]))
                for pct in [0.50, 0.55, 0.60, 0.65]:
                    row = min(int(ph * pct), ph - 1)
                    nz = np.where(pip_mask[row, :] > 0)[0]
                    if len(nz) >= 2:
                        lower_widths.append(int(nz[-1] - nz[0]))
                if waist_widths and lower_widths:
                    min_waist = min(waist_widths)
                    max_lower = max(lower_widths)
                    if min_waist > 0 and max_lower / min_waist > 1.8:
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
    # Try rank text color first (4-color deck), fall back to pip shape
    ch, cw = card_img.shape[:2]
    face_x, face_y = _find_face_start(card_img)
    face_w, face_h = cw - face_x, ch - face_y
    rank_area = card_img[face_y:face_y + max(int(face_h * 0.30), 5),
                         face_x:face_x + max(int(face_w * 0.50), 5)]
    suit = _suit_from_rank_color(rank_area)
    if suit is None:
        suit = _detect_suit(card_img)
    return rank + suit


def _identify_card_right(card_img: np.ndarray,
                         four_color: bool = False) -> Optional[str]:
    """Identify a card where the face is in the right portion (overlapping card 2)."""
    face_x, face_y = _find_face_start(card_img, search_right=True)
    ch, cw = card_img.shape[:2]

    rx1 = max(face_x - 3, 0)
    ry1 = face_y + 2
    rx2 = min(rx1 + max(int(cw * 0.40), 50), cw)
    ry2 = min(ry1 + max(int(ch * 0.30), 42), ch)

    rank_roi = card_img[ry1:ry2, rx1:rx2]
    if rank_roi.size == 0:
        return None

    char_norm = _extract_char_from_roi(rank_roi)
    if char_norm is None:
        return None

    rank, score = _match_rank(char_norm)
    if score < 0.65:
        return None

    # Reject suspiciously narrow characters — likely a partial read from
    # card overlap. Even J (narrowest rank) is 15px at 40px height.
    char_bin = (char_norm > 127).astype(np.float32)
    char_cols = (char_bin > 0.5).any(axis=0)
    char_ink_w = int(char_cols.sum())
    if char_ink_w < 14:
        return None

    # Check for "10" (T)
    if rank == 'Q' or (rank in ('9', '6', 'T') and score < 0.90):
        if _has_one_digit_left(rank_roi, char_norm):
            rank = 'T'

    # Disambiguate Q/9 for hero cards too
    if rank == 'Q':
        nine_score = _match_rank_single(char_norm, '9')
        if score - nine_score < 0.08:
            rank = _disambiguate_Q_9(char_norm)

    # Try rank text color for suit (most reliable for 4-color deck)
    suit = _suit_from_rank_color(rank_roi, four_color=four_color)
    if suit is None:
        suit = _detect_suit_pip(card_img, search_right=True)
    return rank + suit


def _find_merged_card_area(region_img: np.ndarray,
                           min_area: int = 1000,
                           strict: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of the merged white card area.

    Args:
        strict: Use a smaller morphological kernel (for hero cards where
                the default kernel merges non-card UI elements).

    Returns (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 55, 255]))

    if strict:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    return cv2.boundingRect(largest)


def detect_and_identify_board(board_img: np.ndarray,
                               full_img: Optional[np.ndarray] = None) -> List[Optional[str]]:
    """Detect and identify community cards on the board.

    When full_img is provided, uses calibrated BOARD_CARD_SLOTS for precise
    per-card cropping (avoids turn/river splitting errors). Falls back to
    merged-area splitting from board_img if full_img is not given.

    Args:
        board_img: BGR image of the board area (BOARD_CARDS region crop).
            Used for card-presence detection and as fallback.
        full_img: Full table image (after extract_table_area). When provided,
            individual card slots are cropped from this image.

    Returns:
        List of card strings. Empty if no cards detected.
    """
    if full_img is not None:
        return _detect_board_from_slots(full_img)

    # Fallback: merged-area approach (for backward compat / tests)
    rect = _find_merged_card_area(board_img, min_area=5000)
    if rect is None:
        return []

    x, y, w, h = rect
    single_card_aspect = 0.68
    expected_card_w = h * single_card_aspect
    num_cards = max(1, round(w / expected_card_w))
    num_cards = min(num_cards, 5)

    card_width = w // num_cards
    cards = []
    for i in range(num_cards):
        cx1 = x + i * card_width
        cx2 = x + (i + 1) * card_width
        card_img = board_img[y:y + h, cx1:cx2]
        card_str = identify_card(card_img)
        cards.append(card_str)
    return cards


def _detect_board_from_slots(full_img: np.ndarray) -> List[Optional[str]]:
    """Detect board cards using calibrated per-card slot regions."""
    from src.regions import BOARD_CARD_SLOTS

    WHITE_MIN_PCT = 0.10
    ih, iw = full_img.shape[:2]
    cards = []

    for slot in BOARD_CARD_SLOTS:
        x1, y1, x2, y2 = slot.to_pixels(iw, ih)
        card_crop = full_img[y1:y2, x1:x2]

        # Check if a card is present (white card face)
        hsv = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)
        white_pct = np.sum(
            (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 170)
        ) / (card_crop.shape[0] * card_crop.shape[1])

        if white_pct < WHITE_MIN_PCT:
            continue

        result = identify_card(card_crop)
        if result is not None:
            # A card can only appear once on the board
            if result not in cards:
                cards.append(result)

    return cards


def _find_hero_cards_no_close(hero_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find hero card area without morphological closing.

    Used when the default closing merges non-card UI elements.
    Finds the two largest white contours that overlap in y-position (the cards).
    """
    hsv = cv2.cvtColor(hero_img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # Filter to contours that could be cards (area > 500, reasonable aspect)
    card_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Cards are roughly taller than wide
        if bh > bw * 0.5:
            card_contours.append((bx, by, bw, bh, area))

    if len(card_contours) < 2:
        # Fallback: just use the two largest contours
        all_valid = [(cv2.boundingRect(c), cv2.contourArea(c))
                     for c in contours if cv2.contourArea(c) > 500]
        all_valid.sort(key=lambda x: x[1], reverse=True)
        if len(all_valid) >= 2:
            (x1, y1, w1, h1), _ = all_valid[0]
            (x2, y2, w2, h2), _ = all_valid[1]
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1 + w1, x2 + w2)
            max_y = max(y1 + h1, y2 + h2)
            return min_x, min_y, max_x - min_x, max_y - min_y
        return None

    # Sort by area descending, take top 2
    card_contours.sort(key=lambda c: c[4], reverse=True)
    cards = card_contours[:2]

    # Merge bounding boxes
    min_x = min(c[0] for c in cards)
    min_y = min(c[1] for c in cards)
    max_x = max(c[0] + c[2] for c in cards)
    max_y = max(c[1] + c[3] for c in cards)

    return min_x, min_y, max_x - min_x, max_y - min_y


def detect_and_identify_hero(hero_img: np.ndarray) -> List[Optional[str]]:
    """Detect and identify hero's hole cards.

    Args:
        hero_img: BGR image of the hero card area.

    Returns:
        List of 0-2 card strings.
    """
    rh, rw = hero_img.shape[:2]
    rect = _find_merged_card_area(hero_img, min_area=500)
    if rect is None:
        return []

    x, y, w, h = rect

    # If default kernel merged non-card UI elements (area >70% of crop),
    # find the two card contours without morphological closing
    if w * h > rw * rh * 0.70:
        rect = _find_hero_cards_no_close(hero_img)
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

    # Helper: get rank area for a card crop
    def _rank_area(card_crop):
        ch_c, cw_c = card_crop.shape[:2]
        fx, fy = _find_face_start(card_crop)
        fw, fh = cw_c - fx, ch_c - fy
        return card_crop[fy:fy + max(int(fh * 0.30), 5),
                         fx:fx + max(int(fw * 0.50), 5)]

    c1 = hero_img[y:y + h, x:x + split]
    c2 = hero_img[y:y + h, x + split:x + w]

    # Always 4-color deck — rank text color determines suit:
    # green=clubs, blue=diamonds, red=hearts, black=spades.
    # Pip-based detection is unreliable for hero cards because ACR overlays
    # green action labels on the card area, contaminating pip regions.
    c1_rank_roi = _rank_area(c1)
    c1_color = _suit_from_rank_color(c1_rank_roi, four_color=True)

    face_x2, face_y2 = _find_face_start(c2, search_right=True)
    ch2, cw2 = c2.shape[:2]
    fw2, fh2 = cw2 - face_x2, ch2 - face_y2
    c2_rank_roi = c2[face_y2:face_y2 + max(int(fh2 * 0.30), 5),
                     face_x2:face_x2 + max(int(fw2 * 0.50), 5)]
    c2_color = _suit_from_rank_color(c2_rank_roi, four_color=True)
    is_4color = True

    cards = []

    # Card 1: left portion (front card — top-right pip may be cut off)
    c1_rank = _read_rank(c1)
    if c1_rank is not None:
        c1_suit = c1_color  # Already computed above
        if c1_suit is None:
            c1_suit = _suit_from_rank_color(c1_rank_roi, four_color=is_4color)
        if c1_suit is None:
            c1_suit = _detect_suit_hero_best(c1)
        cards.append(c1_rank + c1_suit)
    else:
        cards.append(None)

    # Card 2: right portion — overlapped by card 1 on the left
    # Try with search_right=True since card 1 bleeds into the left of c2
    c2_result = _identify_card_right(c2, four_color=is_4color)
    if c2_result is None:
        # Skip normal identify_card — it picks up card 1's content on the left
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

    # Step 1: Threshold for bright pixels (button face is white/light gray)
    _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

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

    # Step 4: Verify each candidate has the "D" letter pattern + color neutrality
    verified = []
    for cd in candidates:
        margin = 3
        rx1 = max(0, cd['abs_x'] - margin)
        ry1 = max(0, cd['abs_y'] - margin)
        rx2 = min(iw, cd['abs_x'] + cd['w'] + margin)
        ry2 = min(ih, cd['abs_y'] + cd['h'] + margin)

        roi = img[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        bright_pct = np.sum(roi_gray > 180) / roi_gray.size
        dark_pct = np.sum(roi_gray < 120) / roi_gray.size

        # Color neutrality: real D button is gray/white, action circles are colored
        mean_b = float(roi[:, :, 0].mean())
        mean_g = float(roi[:, :, 1].mean())
        mean_r = float(roi[:, :, 2].mean())
        mean_all = (mean_b + mean_g + mean_r) / 3.0
        color_dev = max(abs(mean_b - mean_all), abs(mean_g - mean_all),
                        abs(mean_r - mean_all))

        # The button face should be mostly bright with some dark text,
        # and color-neutral (not green/red/blue action circles)
        if bright_pct > 0.30 and 0.05 < dark_pct < 0.55 and color_dev < 20:
            cd['color_dev'] = color_dev
            verified.append(cd)

    if not verified:
        return None

    # Step 5: Pick the best candidate — prefer most color-neutral, then circularity
    best = max(verified, key=lambda c: (-c['color_dev'], c['circularity'], c['area']))

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

    # Find contours that look like rank characters: taller than wide, reasonable size.
    # Skip contours touching the left edge (card 1 bleed-through).
    rank_candidates = []
    for c in contours:
        cx, cy, cw, ch_c = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if ch_c > 10 and cw < ch_c * 1.5 and area > 20 and cx > 2:
            rank_candidates.append((cx, cy, cw, ch_c, area))

    if not rank_candidates:
        return None

    # Try all candidates and pick the one with the best match score
    best_result = None
    best_score = 0.0

    for cx, cy, cw, ch_c, area in rank_candidates:
        char_img = mask[cy:cy + ch_c, cx:cx + cw]
        target_h = 40
        scale = target_h / ch_c
        new_w = max(1, int(cw * scale))
        char_norm = cv2.resize(char_img, (new_w, target_h),
                               interpolation=cv2.INTER_NEAREST)

        rank, score = _match_rank(char_norm)
        if score > best_score and score > 0.5:
            best_score = score
            best_result = rank

    if best_result is not None:
        c2_full = hero_img[y:y + h, x + split:x + w]
        suit = _detect_suit_pip(c2_full, search_right=True)
        return best_result + suit

    return None
