"""Preflop range lookup for GTO 6-max cash game."""

import json
import os
from typing import Optional, Tuple

RANGES_PATH = os.path.join(os.path.dirname(__file__), "ranges.json")

# Canonical position order (earliest to latest acting preflop after blinds post)
PREFLOP_ORDER = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

# Postflop order: earlier position = first to act (OOP)
POSTFLOP_ORDER = ["SB", "BB", "UTG", "MP", "CO", "BTN"]


class RangeLookup:
    def __init__(self, path=None):
        # type: (Optional[str]) -> None
        with open(path or RANGES_PATH) as f:
            self._data = json.load(f)

    def rfi(self, position):
        # type: (str) -> Optional[str]
        """RFI range for a position (e.g. 'UTG')."""
        return self._data["rfi"].get(position)

    def vs_rfi(self, defender, opener, action):
        # type: (str, str, str) -> Optional[str]
        """Range for defender facing opener's RFI.

        Args:
            defender: position defending (e.g. 'BB')
            opener: position that opened (e.g. 'BTN')
            action: 'call' or '3bet'
        """
        key = "{}_vs_{}".format(defender, opener)
        entry = self._data["vs_rfi"].get(key)
        if entry is None:
            return None
        return entry.get(action)

    def vs_3bet(self, opener, action):
        # type: (str, str) -> Optional[str]
        """Range for opener facing a 3bet.

        Args:
            opener: position that originally opened (e.g. 'CO')
            action: 'call' or '4bet'
        """
        entry = self._data["vs_3bet"].get(opener)
        if entry is None:
            return None
        return entry.get(action)

    def get(self, scenario_key):
        # type: (str) -> Optional[str]
        """Generic flat lookup. Tries rfi first, then vs_rfi, then vs_3bet.

        Accepts keys like:
            'UTG_rfi', 'BB_vs_BTN_call', 'CO_vs_3bet_call'
        """
        if scenario_key.endswith("_rfi"):
            pos = scenario_key[:-4]
            return self.rfi(pos)

        if "_vs_3bet_" in scenario_key:
            parts = scenario_key.split("_vs_3bet_")
            return self.vs_3bet(parts[0], parts[1])

        # e.g. BB_vs_BTN_call or BB_vs_BTN_3bet
        for action in ("call", "3bet", "4bet"):
            suffix = "_" + action
            if scenario_key.endswith(suffix):
                remainder = scenario_key[:-len(suffix)]
                if "_vs_" in remainder:
                    defender, opener = remainder.split("_vs_")
                    return self.vs_rfi(defender, opener, action)

        return None


def ip_oop_positions(pos_a, pos_b):
    # type: (str, str) -> Tuple[str, str]
    """Given two positions, return (oop, ip) for postflop play.

    The player acting first postflop is OOP.
    """
    idx_a = POSTFLOP_ORDER.index(pos_a)
    idx_b = POSTFLOP_ORDER.index(pos_b)
    if idx_a < idx_b:
        return (pos_a, pos_b)
    return (pos_b, pos_a)
