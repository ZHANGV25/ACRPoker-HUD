"""OCR using macOS Vision framework — accurate text recognition for game client."""

from typing import Optional, List, Tuple
import re
import os
import cv2
import numpy as np

import Vision
from Quartz import CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex
from Foundation import NSURL

# Temp file for passing crops to Vision API
_TMP_PATH = "/tmp/_vmon_crop.png"


def _vision_ocr_file(img_path: str) -> List[Tuple[str, float]]:
    """Run Vision OCR on an image file. Returns list of (text, confidence)."""
    url = NSURL.fileURLWithPath_(os.path.abspath(img_path))
    source = CGImageSourceCreateWithURL(url, None)
    if source is None:
        return []
    cg_image = CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        return []

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        return []

    results = []
    for obs in request.results():
        candidates = obs.topCandidates_(1)
        if candidates:
            text = candidates[0].string()
            conf = candidates[0].confidence()
            results.append((text, conf))
    return results


def ocr_crop(img: np.ndarray, tmp_path: str = _TMP_PATH) -> str:
    """Run Vision OCR on a numpy image crop. Returns the recognized text."""
    cv2.imwrite(tmp_path, img)
    results = _vision_ocr_file(tmp_path)
    if results:
        return results[0][0]
    return ""


def ocr_crop_all(img: np.ndarray, tmp_path: str = _TMP_PATH) -> List[str]:
    """Run Vision OCR and return all text lines found."""
    cv2.imwrite(tmp_path, img)
    results = _vision_ocr_file(tmp_path)
    return [text for text, conf in results]


def _fix_bb_text(text: str) -> str:
    """Fix common OCR misreads in the client font.

    The client font causes consistent misreads:
    - 'BR' should be 'BB'
    - 'RR' should be 'BB'
    - 'R8' should be 'BB'
    """
    text = re.sub(r'\bBR\b', 'BB', text)
    text = re.sub(r'\bRR\b', 'BB', text)
    text = re.sub(r'\bR8\b', 'BB', text)
    return text


def parse_bb_amount(text: str) -> Optional[float]:
    """Parse a BB amount from OCR text like '330 5 BB', '97 BB', '87.5 BB'.

    Handles:
    - Space instead of decimal: '330 5' -> 330.5
    - Missing decimal: '3305' -> 330.5 (if followed by BB)
    - Clean reads: '97 BB' -> 97.0
    """
    text = _fix_bb_text(text)

    # Remove 'BB' suffix
    text = re.sub(r'\s*BB\s*$', '', text, flags=re.IGNORECASE).strip()

    # Normalize comma as decimal separator
    text = text.replace(',', '.')

    # Handle leading zero with missing decimal: "05" -> "0.5", "075" -> "0.75"
    match = re.match(r'^0(\d+)$', text)
    if match:
        try:
            return float("0." + match.group(1))
        except ValueError:
            pass

    # Try direct float parse
    try:
        return float(text)
    except ValueError:
        pass

    # Handle "330 5" -> "330.5" (space as decimal separator)
    match = re.match(r'^(\d+)\s+(\d)$', text)
    if match:
        try:
            return float(match.group(1) + "." + match.group(2))
        except ValueError:
            pass

    # Handle just digits
    match = re.match(r'^(\d+)$', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def read_bb_amount(img: np.ndarray) -> Optional[float]:
    """Read a BB amount from an image crop.

    Upscales 2x before OCR to prevent font misreads (e.g. '3' as '2').
    """
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = ocr_crop(upscaled)
    return parse_bb_amount(text)


def read_pot(img: np.ndarray) -> Optional[float]:
    """Read pot amount from the pot region. Handles 'Total: X BB' format."""
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = ocr_crop(upscaled)
    text = _fix_bb_text(text)

    # Extract number from "Total: 13.5 BB" or "Pot 9 BB"
    match = re.search(r'([\d.]+)\s*(?:\d\s)?BB', text, re.IGNORECASE)
    if not match:
        # Try "Total: X Y BB" where Y is decimal part
        match = re.search(r'(\d+)\s+(\d)\s*BB', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1) + "." + match.group(2))
            except ValueError:
                pass

        # Try just finding a number
        match = re.search(r'([\d.]+)', text)

    if match:
        try:
            return float(match.group(1) if len(match.groups()) == 1
                         else match.group(1) + "." + match.group(2))
        except ValueError:
            pass
    return None


def read_action_label(img: np.ndarray) -> Optional[str]:
    """Read action label from a seat (e.g. 'F', 'R', 'C/X', 'R/B')."""
    text = ocr_crop(img).strip().upper()
    # Fix common misreads
    text = text.replace("\\", "/").replace("|", "/")
    # Validate it looks like an action label
    cleaned = text.replace(" ", "")
    if cleaned and all(c in "FRCBX/" for c in cleaned):
        return cleaned
    return None


def read_action_buttons(fold_img: np.ndarray, call_img: np.ndarray,
                        raise_img: np.ndarray) -> dict:
    """Read the three action button regions.

    Vision returns multiple lines per button (e.g. 'Call' and '4.5 BB' separately),
    so we join all lines for each button and parse from the combined text.

    Button crops are upscaled 2x before OCR to prevent decimal point loss in
    small text (e.g. '4.5 BB' reading as '45 PR' at native resolution).
    """
    actions = {}

    def _upscale(img):
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    fold_lines = ocr_crop_all(_upscale(fold_img))
    fold_text = " ".join(fold_lines).lower()
    if "fold" in fold_text:
        actions["fold"] = True

    call_lines = ocr_crop_all(_upscale(call_img))
    call_text = _fix_bb_text(" ".join(call_lines)).lower()
    if "check" in call_text:
        actions["check"] = True
    amount = _extract_amount(call_text)
    if amount is not None and "call" in call_text:
        actions["call"] = amount

    raise_lines = ocr_crop_all(_upscale(raise_img))
    raise_text = _fix_bb_text(" ".join(raise_lines)).lower()
    amount = _extract_amount(raise_text)
    if amount is not None:
        if "raise" in raise_text:
            actions["raise_to"] = amount
        elif "bet" in raise_text:
            actions["bet"] = amount

    return actions


def _extract_amount(text: str) -> Optional[float]:
    """Extract a numeric amount from button text like 'Call 4.5 BB' or 'Raise To 9 BB'."""
    text = re.sub(r'\s*bb\s*', ' ', text, flags=re.IGNORECASE).strip()

    # "4.5" or "9"
    match = re.search(r'(\d+\.\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # "4 5" (space as decimal)
    match = re.search(r'(\d+)\s+(\d)\b', text)
    if match:
        try:
            return float(match.group(1) + "." + match.group(2))
        except ValueError:
            pass

    # Just an integer
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        try:
            return float(matches[-1])  # Take the last number (usually the amount)
        except ValueError:
            pass

    return None
