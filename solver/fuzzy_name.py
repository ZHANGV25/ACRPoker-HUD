"""Fuzzy player name matching for OCR -> hand history name resolution.

OCR frequently garbles player names (e.g. "swarna89" -> "swarna8g",
"RYJITSU" -> "RYJITSL"). This module finds the best match from known
hand history names using edit distance.
"""

from typing import Optional, Set


def _edit_distance(a, b):
    # type: (str, str) -> int
    """Levenshtein edit distance between two strings."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Optimize: use single row DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insert
                prev[j] + 1,          # delete
                prev[j - 1] + cost,   # substitute
            )
        prev = curr
    return prev[lb]


def fuzzy_match(ocr_name, known_names, max_distance=3):
    # type: (str, Set[str], int) -> Optional[str]
    """Find the best matching known name for an OCR-read name.

    Returns the known name if a close enough match is found, or None.

    Rules:
    - Exact match always wins
    - Case-insensitive exact match wins next
    - Edit distance <= max_distance with best score wins
    - For short names (<=4 chars), max_distance is reduced to 1
    - Never matches if the name is too short (<3 chars)
    """
    if not ocr_name or len(ocr_name) < 3:
        return None
    if not known_names:
        return None

    # Exact match
    if ocr_name in known_names:
        return ocr_name

    # Case-insensitive exact match
    ocr_lower = ocr_name.lower()
    for name in known_names:
        if name.lower() == ocr_lower:
            return name

    # Short names need closer matches
    effective_max = max_distance
    if len(ocr_name) <= 4:
        effective_max = 1
    elif len(ocr_name) <= 6:
        effective_max = 2

    # Find closest by edit distance
    best_name = None  # type: Optional[str]
    best_dist = effective_max + 1

    for name in known_names:
        # Quick length check — edit distance is at least abs(len diff)
        if abs(len(name) - len(ocr_name)) > effective_max:
            continue
        dist = _edit_distance(ocr_lower, name.lower())
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name
