"""Screen capture module for ACR Poker windows on macOS."""

from typing import Optional, List, Dict
import subprocess
import numpy as np

try:
    import Quartz
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        CGWindowListCreateImage,
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        kCGWindowImageBoundsIgnoreFraming,
    )
    import Quartz.CoreGraphics as CG
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False

import cv2
from PIL import Image


def find_acr_windows() -> List[Dict]:
    """Find all ACR Poker table windows.

    Returns list of dicts with keys: id, title, bounds (x, y, w, h)
    """
    if not HAS_QUARTZ:
        raise RuntimeError("Quartz not available - macOS only")

    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    )

    acr_windows = []
    for window in window_list:
        owner = window.get("kCGWindowOwnerName", "")
        title = window.get("kCGWindowName", "")

        # ACR Poker window identification
        if "ACR" in owner or "Winning Poker" in owner or "Hold'em" in title:
            bounds = window.get("kCGWindowBounds", {})
            acr_windows.append({
                "id": window.get("kCGWindowNumber"),
                "title": title,
                "owner": owner,
                "bounds": {
                    "x": int(bounds.get("X", 0)),
                    "y": int(bounds.get("Y", 0)),
                    "w": int(bounds.get("Width", 0)),
                    "h": int(bounds.get("Height", 0)),
                },
            })

    return acr_windows


def capture_window(window_id: int) -> Optional[np.ndarray]:
    """Capture a specific window by its window ID.

    Returns BGR numpy array (OpenCV format) or None on failure.
    """
    if not HAS_QUARTZ:
        raise RuntimeError("Quartz not available - macOS only")

    image_ref = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )

    if image_ref is None:
        return None

    width = CG.CGImageGetWidth(image_ref)
    height = CG.CGImageGetHeight(image_ref)
    bytes_per_row = CG.CGImageGetBytesPerRow(image_ref)

    pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image_ref))
    img_array = np.frombuffer(pixel_data, dtype=np.uint8)
    img_array = img_array.reshape((height, bytes_per_row // 4, 4))
    img_array = img_array[:height, :width, :]

    # BGRA -> BGR
    bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    return bgr


def capture_from_file(filepath: str) -> np.ndarray:
    """Load a screenshot from file (for testing/development)."""
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img


if __name__ == "__main__":
    # Quick test: find ACR windows and capture the first one
    import sys

    if len(sys.argv) > 1:
        # Load from file
        img = capture_from_file(sys.argv[1])
        print(f"Loaded image from file: {img.shape}")
        cv2.imwrite("test_capture.png", img)
    else:
        windows = find_acr_windows()
        if not windows:
            print("No ACR windows found. Pass a screenshot path to test from file.")
            sys.exit(1)

        print(f"Found {len(windows)} ACR window(s):")
        for w in windows:
            print(f"  [{w['id']}] {w['title']} ({w['bounds']['w']}x{w['bounds']['h']})")

        img = capture_window(windows[0]["id"])
        if img is not None:
            cv2.imwrite("test_capture.png", img)
            print(f"Captured: {img.shape}")
