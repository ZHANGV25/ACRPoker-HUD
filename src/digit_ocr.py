"""Template-based digit OCR for ACR Poker.

Instead of using Tesseract (which misreads this font), we extract digit templates
from reference screenshots and match against them. This is 100% reliable since
the poker client uses a fixed font.
"""

from typing import Optional, List, Tuple
import cv2
import numpy as np
import os
import json

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def extract_green_channel(img: np.ndarray) -> np.ndarray:
    """Extract green-dominant text from ACR screenshot crop."""
    if len(img.shape) == 3:
        b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        result = np.clip(g - np.maximum(r, b) * 0.5, 0, 255).astype(np.uint8)
        return result
    return img


def extract_white_channel(img: np.ndarray) -> np.ndarray:
    """Extract white/bright text from ACR screenshot crop."""
    if len(img.shape) == 3:
        gray = np.min(img, axis=2)
        return gray
    return img


def binarize(img: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Convert to binary using Otsu if threshold is 0."""
    if threshold == 0:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def find_character_boxes(binary: np.ndarray, min_height: int = 5) -> List[Tuple[int, int, int, int]]:
    """Find bounding boxes of individual characters in a binary image.

    Returns list of (x, y, w, h) sorted left to right.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= min_height and w >= 2:
            boxes.append((x, y, w, h))

    # Sort left to right
    boxes.sort(key=lambda b: b[0])

    # Merge overlapping boxes
    merged = []
    for box in boxes:
        if merged and box[0] < merged[-1][0] + merged[-1][2] + 2:
            prev = merged[-1]
            x = min(prev[0], box[0])
            y = min(prev[1], box[1])
            x2 = max(prev[0] + prev[2], box[0] + box[2])
            y2 = max(prev[1] + prev[3], box[1] + box[3])
            merged[-1] = (x, y, x2 - x, y2 - y)
        else:
            merged.append(box)

    return merged


def find_decimal_points(binary: np.ndarray, char_boxes: List[Tuple[int, int, int, int]]) -> List[int]:
    """Find x positions where decimal points exist between character boxes.

    The decimal point in ACR's font is a tiny 2-3px blob that's too small
    to detect as a character. Instead, find tiny contours between digit boxes.

    Returns list of indices: decimal point appears AFTER box at that index.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find tiny blobs (potential decimal points)
    dots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 4 and w <= 5 and h >= 1 and w >= 1:
            dots.append((x + w // 2, y))  # center x, y

    decimal_positions = []
    for dot_x, dot_y in dots:
        # Check which character boxes this dot falls between
        for i in range(len(char_boxes) - 1):
            box_end = char_boxes[i][0] + char_boxes[i][2]
            next_start = char_boxes[i + 1][0]
            if box_end - 2 <= dot_x <= next_start + 2:
                decimal_positions.append(i)
                break

    return decimal_positions


def crop_and_normalize(binary: np.ndarray, box: Tuple[int, int, int, int],
                       target_height: int = 32) -> np.ndarray:
    """Crop a character from binary image and normalize to fixed height."""
    x, y, w, h = box
    char_img = binary[y:y + h, x:x + w]

    # Scale to target height maintaining aspect ratio
    scale = target_height / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(char_img, (new_w, target_height), interpolation=cv2.INTER_NEAREST)
    return resized


def save_templates_from_crop(crop: np.ndarray, text: str, template_dir: str = TEMPLATE_DIR,
                             color: str = "green"):
    """Extract and save character templates from a known crop.

    Args:
        crop: BGR image crop containing known text
        text: the actual text content (e.g. "330.5")
        template_dir: where to save templates
        color: "green" for stack text, "white" for button text
    """
    os.makedirs(template_dir, exist_ok=True)

    if color == "green":
        gray = extract_green_channel(crop)
    else:
        gray = extract_white_channel(crop)

    binary = binarize(gray)
    boxes = find_character_boxes(binary)

    if len(boxes) != len(text):
        print("Warning: found %d boxes but text has %d chars: %r" % (len(boxes), len(text), text))
        print("Boxes: %s" % boxes)
        return

    for char, box in zip(text, boxes):
        char_img = crop_and_normalize(binary, box)

        # Use safe filename
        if char == ".":
            char_name = "dot"
        elif char == "/":
            char_name = "slash"
        elif char == " ":
            continue
        else:
            char_name = char

        path = os.path.join(template_dir, "%s_%s.png" % (char_name, color))
        # If template already exists, check if this one is better (larger)
        if os.path.exists(path):
            existing = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if existing is not None and existing.shape[1] >= char_img.shape[1]:
                continue

        cv2.imwrite(path, char_img)
        print("Saved template: %s (%dx%d)" % (path, char_img.shape[1], char_img.shape[0]))


def load_templates(template_dir: str = TEMPLATE_DIR) -> dict:
    """Load all character templates.

    Returns dict mapping character -> list of template images.
    """
    templates = {}
    if not os.path.exists(template_dir):
        return templates

    for fname in os.listdir(template_dir):
        if not fname.endswith(".png"):
            continue
        parts = fname.replace(".png", "").split("_")
        char = parts[0]
        if char == "dot":
            char = "."
        elif char == "slash":
            char = "/"

        img = cv2.imread(os.path.join(template_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if char not in templates:
                templates[char] = []
            templates[char].append(img)

    return templates


def match_character(char_img: np.ndarray, templates: dict) -> Tuple[str, float]:
    """Match a character image against templates.

    Returns (best_char, confidence) tuple.
    """
    best_char = "?"
    best_score = -1

    for char, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            # Resize template to match character height
            if char_img.shape[0] != tmpl.shape[0]:
                scale = char_img.shape[0] / tmpl.shape[0]
                new_w = max(1, int(tmpl.shape[1] * scale))
                tmpl_resized = cv2.resize(tmpl, (new_w, char_img.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
            else:
                tmpl_resized = tmpl

            # Pad the smaller image to match widths
            w1, w2 = char_img.shape[1], tmpl_resized.shape[1]
            if w1 != w2:
                max_w = max(w1, w2)
                pad1 = np.zeros((char_img.shape[0], max_w), dtype=np.uint8)
                pad2 = np.zeros((char_img.shape[0], max_w), dtype=np.uint8)
                # Center both
                offset1 = (max_w - w1) // 2
                offset2 = (max_w - w2) // 2
                pad1[:, offset1:offset1 + w1] = char_img
                pad2[:, offset2:offset2 + w2] = tmpl_resized
            else:
                pad1 = char_img
                pad2 = tmpl_resized

            # Normalized cross-correlation
            if pad1.std() == 0 or pad2.std() == 0:
                continue
            score = cv2.matchTemplate(pad1, pad2, cv2.TM_CCOEFF_NORMED)
            max_score = score.max()

            if max_score > best_score:
                best_score = max_score
                best_char = char

    return best_char, best_score


def read_text_template(img: np.ndarray, templates: dict,
                       color: str = "green") -> str:
    """Read text from an image crop using template matching.

    Args:
        img: BGR image crop
        templates: loaded templates dict
        color: "green" for stack/pot text, "white" for button text
    """
    if color == "green":
        gray = extract_green_channel(img)
    else:
        gray = extract_white_channel(img)

    binary = binarize(gray)
    boxes = find_character_boxes(binary)

    result = []
    prev_x2 = 0
    for box in boxes:
        x, y, w, h = box

        # Detect spaces (large gap between characters)
        if prev_x2 > 0 and x - prev_x2 > w * 0.8:
            result.append(" ")

        char_img = crop_and_normalize(binary, box)
        char, confidence = match_character(char_img, templates)
        result.append(char)
        prev_x2 = x + w

    return "".join(result)


def read_bb_amount(img: np.ndarray, templates: dict,
                   color: str = "green") -> Optional[float]:
    """Read a BB amount from an image crop (e.g. '330.5 BB').

    Handles decimal point detection since the '.' in ACR's font is too small
    for contour detection as a character. Stops reading at 'BB'.
    """
    if color == "green":
        gray = extract_green_channel(img)
    else:
        gray = extract_white_channel(img)

    binary = binarize(gray)
    boxes = find_character_boxes(binary)
    decimals = find_decimal_points(binary, boxes)

    if not boxes:
        return None

    # Match each character
    chars = []
    for i, box in enumerate(boxes):
        char_img = crop_and_normalize(binary, box)
        char, confidence = match_character(char_img, templates)
        chars.append((char, i))

    # Build the number string, inserting decimals and stopping at BB
    result = ""
    for char, idx in chars:
        if char == "B":
            break  # Hit the BB suffix, stop
        result += char
        if idx in decimals:
            result += "."

    if result:
        try:
            return float(result)
        except ValueError:
            return None
    return None
