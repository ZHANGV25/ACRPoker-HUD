"""OCR module for reading text and cards from ACR Poker screenshots."""

from typing import Optional, List, Dict
import re
import cv2
import numpy as np
import pytesseract


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Preprocess a cropped region for Tesseract OCR.

    Strategy: extract the green/white text channels specifically, then threshold.
    Green text: high G relative to R and B.
    White text: all channels high.
    """
    if len(img.shape) == 3:
        b, g, r = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)

        # Green-dominant text: G is significantly higher than R and B
        green_text = np.clip(g - np.maximum(r, b) * 0.6, 0, 255).astype(np.uint8)

        # White/bright text: all channels high AND similar
        # (filters out gold border which has high R, medium G, low B)
        min_ch = np.minimum(np.minimum(b, g), r)
        max_ch = np.maximum(np.maximum(b, g), r)
        spread = max_ch - min_ch
        white_text = np.where((min_ch > 140) & (spread < 60), min_ch, 0).astype(np.uint8)

        gray = np.maximum(green_text, white_text)
    else:
        gray = img

    # Upscale small regions
    h, w = gray.shape[:2]
    if h < 50:
        scale = max(3, 50 // max(h, 1))
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Otsu's threshold works well now that background noise is suppressed
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def read_text(img: np.ndarray, whitelist: str = "", psm: int = 7) -> str:
    """Read text from a cropped image region.

    Args:
        img: BGR image crop
        whitelist: allowed characters (e.g. "0123456789.$" for amounts)
        psm: page segmentation mode (7 = single line, 6 = block)
    """
    processed = preprocess_for_ocr(img)

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    text = pytesseract.image_to_string(processed, config=config).strip()
    return text


def read_bb_amount(img: np.ndarray) -> Optional[float]:
    """Read a BB amount from a cropped region (e.g. '99 BB', '4.5 BB', '330.5 BB').

    Uses digits-only whitelist since Tesseract misreads 'BB' as 'RR'/'BR' in this font.
    The BB suffix is expected but not required for matching.
    """
    text = read_text(img, whitelist="0123456789. ")
    # Extract the number - could be "330 5" (missed decimal) or "330.5" or "97"
    # Clean up spaces that should be decimals
    text = text.strip()
    # Common Tesseract errors: space instead of decimal point
    # If we see "330 5" it's really "330.5", "87 5" is "87.5"
    match = re.search(r"(\d+)[.\s](\d)\s*$", text)
    if match:
        try:
            return float(match.group(1) + "." + match.group(2))
        except ValueError:
            return None
    # Simple integer like "97" or "100"
    match = re.search(r"(\d+)\s*$", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def read_dollar_amount(img: np.ndarray) -> Optional[float]:
    """Read a dollar amount from a cropped region (e.g. '$15.67', '$1,269.20')."""
    text = read_text(img, whitelist="0123456789.$,")
    match = re.search(r"\$?([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def read_pot(img: np.ndarray, bb_mode: bool = True) -> Optional[float]:
    """Read pot amount from the pot region."""
    if bb_mode:
        text = read_text(img, whitelist="0123456789. ")
        # Extract any number
        match = re.search(r"(\d+\.?\d*)", text)
    else:
        text = read_text(img, whitelist="0123456789.$, ")
        match = re.search(r"\$?([\d,]+\.?\d*)", text)

    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def read_action_label(img: np.ndarray) -> Optional[str]:
    """Read action label from a seat (e.g. 'F', 'R', 'C/X', 'R/B')."""
    text = read_text(img, whitelist="FRCBX/")
    text = text.strip()
    if text and all(c in "FRCBX/" for c in text):
        return text
    return None


def read_action_buttons(img: np.ndarray) -> dict:
    """Read the action buttons region to determine available actions.

    Returns dict like:
        {"fold": True, "check": True}
        {"fold": True, "call": 4.5, "raise_to": 9.0}
    """
    text = read_text(img, psm=6).lower()
    actions = {}

    if "fold" in text:
        actions["fold"] = True
    if "check" in text:
        actions["check"] = True

    call_match = re.search(r"call\s*([\d.]+)", text)
    if call_match:
        try:
            actions["call"] = float(call_match.group(1))
        except ValueError:
            pass

    raise_match = re.search(r"raise\s*(?:to)?\s*([\d.]+)", text)
    if raise_match:
        try:
            actions["raise_to"] = float(raise_match.group(1))
        except ValueError:
            pass

    bet_match = re.search(r"bet\s*([\d.]+)", text)
    if bet_match:
        try:
            actions["bet"] = float(bet_match.group(1))
        except ValueError:
            pass

    return actions


def detect_cards_by_color(img: np.ndarray) -> List[Dict]:
    """Detect card regions in an image by finding white rectangles (card faces).

    Returns list of bounding boxes for detected cards.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White card face detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Card-like proportions: roughly 0.6-0.85 aspect ratio, reasonable size
        if 0.5 < aspect_ratio < 1.0 and area > 500:
            cards.append({"x": x, "y": y, "w": w, "h": h})

    # Sort left to right
    cards.sort(key=lambda c: c["x"])
    return cards


# Card suit color ranges in HSV
SUIT_COLORS = {
    "hearts": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},  # Red
    "diamonds": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},  # Red
    "clubs": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 50, 80])},  # Black
    "spades": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 50, 80])},  # Black
}

RANK_MAP = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "J": "J", "Q": "Q", "K": "K", "A": "A",
}

SUIT_MAP = {
    "hearts": "h", "diamonds": "d", "clubs": "c", "spades": "s",
}
