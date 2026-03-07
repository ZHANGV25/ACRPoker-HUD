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


def _hand_to_combo_key(cards):
    # type: (list) -> Optional[str]
    """Convert hero cards like ['Ah', 'Ks'] to range key like 'AKo'.

    Returns None if cards are invalid.
    """
    if not cards or len(cards) != 2 or not all(cards):
        return None
    RANK_ORDER = "23456789TJQKA"
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    # Higher rank first
    i1 = RANK_ORDER.index(r1) if r1 in RANK_ORDER else -1
    i2 = RANK_ORDER.index(r2) if r2 in RANK_ORDER else -1
    if i1 < 0 or i2 < 0:
        return None
    if i1 < i2:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2  # pair like "AA"
    elif s1 == s2:
        return r1 + r2 + "s"  # suited like "AKs"
    else:
        return r1 + r2 + "o"  # offsuit like "AKo"


def _hand_in_range(combo_key, range_str):
    # type: (str, str) -> bool
    """Check if a combo key (e.g. 'AKo') is in a range string."""
    if not range_str or not combo_key:
        return False
    # Range string is comma-separated: "AA,KK,AKs,AKo,..."
    combos = set(c.strip() for c in range_str.split(","))
    return combo_key in combos


def preflop_advice(hero_cards, hero_pos, game_state, rl):
    # type: (list, str, ..., RangeLookup) -> Optional[str]
    """Return preflop action advice based on position and ranges.

    Returns a string like "RAISE (in UTG open range)" or
    "FOLD (not in range)" or None if can't determine.
    """
    if not hero_cards or len(hero_cards) != 2 or not all(hero_cards):
        return None

    combo = _hand_to_combo_key(hero_cards)
    if combo is None:
        return None

    # Figure out the preflop scenario from visible actions
    # Look at who has raised, called, etc.
    positions = game_state.infer_positions()
    if not positions:
        return None

    # Find first raiser (opener) by checking bets > 1 BB or "R" action label
    opener_pos = None
    opener_bet = 0.0
    three_bettor_pos = None
    for p in game_state.players:
        if p.is_sitting_out or p.is_folded:
            continue
        pos = positions.get(p.seat, "")
        bet = p.current_bet_bb or 0
        action = (p.action_label or "").upper()
        is_raise = action in ("R", "R/B", "RAISE") or (pos not in ("SB", "BB") and bet > 1.0)
        is_blind_raise = (pos == "SB" and bet > 0.5) or (pos == "BB" and bet > 1.0)

        if is_raise or is_blind_raise:
            if opener_pos and bet > opener_bet:
                three_bettor_pos = pos
            elif not opener_pos:
                opener_pos = pos
                opener_bet = bet

    # If we still can't find opener but hero has a Call option > 1 BB, someone raised.
    # Cross-validate: at least one non-blind player must have a bet > 1 BB to confirm.
    # (OCR can misread "0.5" as "5.0", creating phantom raises)
    aa = game_state.available_actions or {}
    call_amt = aa.get("call")
    if not opener_pos and call_amt is not None and call_amt > 1.0:
        # Verify a non-blind player actually has a raised bet
        has_raiser_bet = False
        for p in game_state.players:
            if p.is_sitting_out or p.is_folded or p.is_hero:
                continue
            pos = positions.get(p.seat, "")
            bet = p.current_bet_bb or 0
            if pos not in ("SB", "BB") and bet > 1.0:
                has_raiser_bet = True
                break
            if pos == "BB" and bet > 1.0:
                has_raiser_bet = True
                break
        if has_raiser_bet:
            for pos_name in PREFLOP_ORDER:
                for p in game_state.players:
                    if p.is_sitting_out or p.is_folded or p.is_hero:
                        continue
                    if positions.get(p.seat) == pos_name:
                        opener_pos = pos_name
                        break
                if opener_pos:
                    break

    # Scenario 1: No one has raised yet — hero should RFI?
    if not opener_pos:
        rfi_range = rl.rfi(hero_pos)
        if rfi_range is None:
            if hero_pos == "BB":
                return "CHECK  (BB with no raise)"
            return None
        if _hand_in_range(combo, rfi_range):
            return "RAISE  {} in {}'s open range".format(combo, hero_pos)
        else:
            return "FOLD   {} not in {}'s open range".format(combo, hero_pos)

    # Scenario 2: Someone opened, no 3bet yet — hero faces RFI
    if opener_pos and not three_bettor_pos:
        call_range = rl.vs_rfi(hero_pos, opener_pos, "call")
        three_bet_range = rl.vs_rfi(hero_pos, opener_pos, "3bet")

        in_3bet = _hand_in_range(combo, three_bet_range) if three_bet_range else False
        in_call = _hand_in_range(combo, call_range) if call_range else False

        if in_3bet:
            return "3-BET  {} vs {} open".format(combo, opener_pos)
        elif in_call:
            return "CALL   {} vs {} open".format(combo, opener_pos)
        else:
            return "FOLD   {} not in range vs {} open".format(combo, opener_pos)

    # Scenario 3: There's a 3bet — hero faces 3bet (hero was opener)
    if three_bettor_pos and hero_pos == opener_pos:
        call_range = rl.vs_3bet(hero_pos, "call")
        four_bet_range = rl.vs_3bet(hero_pos, "4bet")

        in_4bet = _hand_in_range(combo, four_bet_range) if four_bet_range else False
        in_call = _hand_in_range(combo, call_range) if call_range else False

        if in_4bet:
            return "4-BET  {} vs 3bet".format(combo)
        elif in_call:
            return "CALL   {} vs 3bet".format(combo)
        else:
            return "FOLD   {} vs 3bet".format(combo)

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
