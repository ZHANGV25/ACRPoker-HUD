"""Reconstruct preflop action from OCR snapshots and determine solver inputs."""

from typing import Dict, List, Optional, Tuple
from solver.range_lookup import RangeLookup, PREFLOP_ORDER, POSTFLOP_ORDER, ip_oop_positions

# Map OCR action labels to semantic actions
ACTION_MAP = {
    "F": "fold",
    "C": "call",
    "C/X": "call",
    "X": "check",
    "R": "raise",
    "R/B": "raise",
    "B": "bet",
}

# Default bet sizing tree for solver (% of pot)
DEFAULT_BET_SIZES = {
    "oop_bet": ["33%", "67%", "150%"],
    "oop_raise": ["60%", "100%", "allin"],
    "ip_bet": ["33%", "67%", "150%"],
    "ip_raise": ["60%", "100%", "allin"],
}


def _normalize_action(label):
    # type: (Optional[str]) -> Optional[str]
    if label is None:
        return None
    label = label.strip().upper()
    return ACTION_MAP.get(label, label.lower())


class PreflopAction:
    """Describes what happened preflop in a hand."""

    def __init__(self):
        self.opener = None          # type: Optional[str]  # position that RFI'd
        self.open_size = None       # type: Optional[float]
        self.callers = []           # type: List[str]  # positions that flatted
        self.three_bettor = None    # type: Optional[str]
        self.three_bet_callers = [] # type: List[str]
        self.four_bettor = None     # type: Optional[str]
        self.is_limped = False
        self.is_walk = False        # BB wins uncontested

    @property
    def scenario_type(self):
        # type: () -> str
        if self.four_bettor:
            return "4bet"
        if self.three_bettor:
            return "3bet"
        if self.opener:
            return "rfi"
        if self.is_limped:
            return "limp"
        return "unknown"

    def __repr__(self):
        parts = ["PreflopAction("]
        if self.opener:
            parts.append("opener={}, callers={}".format(self.opener, self.callers))
        if self.three_bettor:
            parts.append(", 3bettor={}".format(self.three_bettor))
        if self.four_bettor:
            parts.append(", 4bettor={}".format(self.four_bettor))
        parts.append(")")
        return "".join(parts)


def reconstruct_preflop(game_state, positions=None):
    # type: (...) -> PreflopAction
    """Reconstruct preflop action from a game state snapshot.

    Works best when called on a preflop snapshot (action labels reflect preflop).
    On postflop snapshots, infers from who's still in the pot + positions.

    Args:
        game_state: GameState object from pipeline
        positions: dict of seat -> position name (from infer_positions()).
                   If None, calls game_state.infer_positions().

    Returns:
        PreflopAction describing what happened.
    """
    if positions is None:
        positions = game_state.infer_positions()

    result = PreflopAction()
    players = game_state.players

    # Build seat -> player lookup
    seat_to_player = {p.seat: p for p in players}
    seat_to_pos = positions

    # Active seats (not folded, not sitting out) with known positions
    active_seats = []
    for p in players:
        if p.is_sitting_out:
            continue
        if p.seat not in seat_to_pos:
            continue
        active_seats.append(p.seat)

    if not active_seats:
        return result

    # Sort active seats by preflop acting order
    def preflop_sort_key(seat):
        pos = seat_to_pos.get(seat, "")
        if pos in PREFLOP_ORDER:
            return PREFLOP_ORDER.index(pos)
        return 99

    active_seats.sort(key=preflop_sort_key)

    # On preflop, we can read action labels directly
    if game_state.street == "preflop":
        return _reconstruct_from_labels(active_seats, seat_to_player, seat_to_pos)

    # On postflop, infer from who's in the pot
    return _infer_from_postflop(active_seats, seat_to_player, seat_to_pos, game_state)


def _reconstruct_from_labels(active_seats, seat_to_player, seat_to_pos):
    # type: (List[int], Dict, Dict) -> PreflopAction
    """Reconstruct from preflop action labels."""
    result = PreflopAction()

    raisers = []
    callers = []
    folders = []

    for seat in active_seats:
        p = seat_to_player[seat]
        pos = seat_to_pos[seat]
        action = _normalize_action(p.action_label)

        if action == "fold":
            folders.append(pos)
        elif action == "raise":
            raisers.append((pos, p.current_bet_bb))
        elif action == "call":
            callers.append(pos)

    if not raisers:
        if callers:
            result.is_limped = True
        return result

    # First raiser is the opener (earliest in preflop order)
    raisers_sorted = sorted(raisers, key=lambda x: PREFLOP_ORDER.index(x[0])
                            if x[0] in PREFLOP_ORDER else 99)

    result.opener = raisers_sorted[0][0]
    result.open_size = raisers_sorted[0][1]

    if len(raisers_sorted) >= 2:
        result.three_bettor = raisers_sorted[1][0]
        if len(raisers_sorted) >= 3:
            result.four_bettor = raisers_sorted[2][0]

    # Callers who aren't the opener or 3bettor
    raise_positions = {r[0] for r in raisers_sorted}
    for pos in callers:
        if pos not in raise_positions:
            if result.three_bettor:
                result.three_bet_callers.append(pos)
            else:
                result.callers.append(pos)

    return result


def _infer_from_postflop(active_seats, seat_to_player, seat_to_pos, game_state):
    # type: (List[int], Dict, Dict, ...) -> PreflopAction
    """Infer preflop action from postflop state (who's still in)."""
    result = PreflopAction()

    in_pot = []
    for seat in active_seats:
        p = seat_to_player[seat]
        if not p.is_folded:
            pos = seat_to_pos.get(seat)
            if pos:
                in_pot.append(pos)

    if len(in_pot) < 2:
        return result

    # Sort by preflop order
    in_pot.sort(key=lambda p: PREFLOP_ORDER.index(p) if p in PREFLOP_ORDER else 99)

    # Heuristic: the earliest non-blind position in the pot likely opened
    # If only blinds, SB completed or limped
    non_blind = [p for p in in_pot if p not in ("SB", "BB")]
    blinds_in = [p for p in in_pot if p in ("SB", "BB")]

    if non_blind:
        result.opener = non_blind[0]
        result.callers = non_blind[1:] + blinds_in
    elif len(blinds_in) == 2:
        # SB vs BB — SB opened or limped
        result.opener = "SB"
        result.callers = ["BB"]
    elif blinds_in:
        result.opener = blinds_in[0]

    # Use pot size to guess if 3bet happened
    # Single raised pot ~6-7bb, 3bet pot ~20-25bb at flop
    total = game_state.total_bb or game_state.pot_bb or 0
    num_in_pot = len(in_pot)
    if num_in_pot == 2 and total > 15:
        # Likely a 3bet pot
        if result.callers:
            result.three_bettor = result.callers[-1]
            result.callers = []
            # Swap: the 3bettor re-raised, opener called
            result.three_bet_callers = [result.opener]

    return result


def determine_solver_inputs(game_state, range_lookup, hand_tracker=None):
    # type: (...) -> Optional[Dict]
    """Convert game state into solver-ready inputs.

    Returns dict with:
        board: list of card strings
        oop_range: PioSOLVER range string
        ip_range: PioSOLVER range string
        starting_pot: float (pot in BB at start of current street)
        effective_stack: float (min of active stacks)
        hero_position: 'ip' or 'oop'
        hero_hand: string e.g. 'AhKd'
        bet_sizes: dict of bet sizing config
    Returns None if we can't determine inputs (e.g. no board, multiway).
    """
    if not game_state.board or len(game_state.board) < 3:
        return None  # Need at least a flop

    positions = game_state.infer_positions()

    # Get preflop action (from tracker if available, else reconstruct)
    if hand_tracker and hand_tracker.preflop_action:
        preflop = hand_tracker.preflop_action
    else:
        preflop = reconstruct_preflop(game_state, positions)

    # Find active (non-folded) players with positions
    active = []
    for p in game_state.players:
        if p.is_folded or p.is_sitting_out:
            continue
        pos = positions.get(p.seat)
        if pos:
            active.append((p, pos))

    if len(active) < 2:
        return None

    # For now, only handle heads-up postflop
    if len(active) > 2:
        # Multiway: solve hero vs most relevant villain (closest position)
        hero_entry = None
        for p, pos in active:
            if p.is_hero:
                hero_entry = (p, pos)
                break
        if not hero_entry:
            return None

        # Pick the villain who's not hero
        # For multiway, pick the one with the largest stack (most relevant)
        villains = [(p, pos) for p, pos in active if not p.is_hero]
        villains.sort(key=lambda x: x[0].stack_bb or 0, reverse=True)
        active = [hero_entry, villains[0]]

    player_a, pos_a = active[0]
    player_b, pos_b = active[1]

    # Determine IP and OOP
    oop_pos, ip_pos = ip_oop_positions(pos_a, pos_b)

    if pos_a == oop_pos:
        oop_player, ip_player = player_a, player_b
    else:
        oop_player, ip_player = player_b, player_a

    # Look up ranges based on preflop action
    oop_range, ip_range = _lookup_ranges(
        preflop, oop_pos, ip_pos, range_lookup
    )

    if not oop_range or not ip_range:
        return None

    # Effective stack = min of both players' stacks
    stacks = [s for s in [oop_player.stack_bb, ip_player.stack_bb] if s is not None]
    effective_stack = min(stacks) if stacks else 100.0

    # Starting pot (total pot at start of current street)
    # The OCR gives us current total pot, which includes current street bets
    # Approximate: total pot minus current street bets
    current_bets = 0
    for p, pos in active:
        if p.current_bet_bb:
            current_bets += p.current_bet_bb
    starting_pot = (game_state.total_bb or game_state.pot_bb or 0) - current_bets
    if starting_pot < 1:
        starting_pot = game_state.total_bb or game_state.pot_bb or 6.0

    # Hero info
    hero_pos = None
    for p in game_state.players:
        if p.is_hero:
            hero_pos = positions.get(p.seat)
            break

    hero_position = None
    if hero_pos == oop_pos:
        hero_position = "oop"
    elif hero_pos == ip_pos:
        hero_position = "ip"

    hero_hand = ""
    if game_state.hero_cards:
        hero_hand = "".join(game_state.hero_cards)

    return {
        "board": game_state.board,
        "oop_range": oop_range,
        "ip_range": ip_range,
        "oop_position": oop_pos,
        "ip_position": ip_pos,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "hero_position": hero_position,
        "hero_hand": hero_hand,
        "bet_sizes": DEFAULT_BET_SIZES,
    }


def _lookup_ranges(preflop, oop_pos, ip_pos, rl):
    # type: (PreflopAction, str, str, RangeLookup) -> Tuple[Optional[str], Optional[str]]
    """Look up OOP and IP ranges from preflop action and range DB."""

    scenario = preflop.scenario_type

    if scenario == "rfi":
        opener = preflop.opener
        # The non-opener is the caller
        other = oop_pos if ip_pos == opener else ip_pos

        opener_range = rl.rfi(opener)
        caller_range = rl.vs_rfi(other, opener, "call")

        if opener == ip_pos:
            return (caller_range, opener_range)
        else:
            return (opener_range, caller_range)

    elif scenario == "3bet":
        opener = preflop.opener
        three_bettor = preflop.three_bettor

        # Opener called the 3bet
        opener_range = rl.vs_3bet(opener, "call")
        # 3bettor's range
        bettor_range = rl.vs_rfi(three_bettor, opener, "3bet")

        if opener == oop_pos:
            return (opener_range, bettor_range)
        else:
            return (bettor_range, opener_range)

    elif scenario == "4bet":
        opener = preflop.opener
        # Simplify: use opener's 4bet range vs 3bettor's call-4bet range
        opener_range = rl.vs_3bet(opener, "4bet")
        other = oop_pos if ip_pos == opener else ip_pos
        bettor_range = rl.vs_3bet(other, "call")

        if opener == ip_pos:
            return (bettor_range, opener_range)
        else:
            return (opener_range, bettor_range)

    elif scenario == "limp":
        # Limped pot — use wide ranges
        return (rl.rfi("SB"), rl.rfi("BTN"))

    # Fallback: use position-based RFI ranges
    oop_range = rl.rfi(oop_pos) or rl.rfi("CO")
    ip_range = rl.rfi(ip_pos) or rl.rfi("BTN")
    return (oop_range, ip_range)


class HandTracker:
    """Tracks hand state across screenshots for accurate range assignment.

    Use with live.py: call update() on each new GameState.
    The tracker remembers preflop actions even after the board appears.
    """

    def __init__(self):
        self.current_hand_id = None  # type: Optional[str]
        self.preflop_action = None   # type: Optional[PreflopAction]
        self._positions = None       # type: Optional[Dict]
        self._preflop_seen = False

    def update(self, game_state):
        # type: (...) -> None
        """Process a new game state snapshot."""
        hand_id = game_state.hand_id

        # Detect new hand
        if hand_id != self.current_hand_id:
            self._reset()
            self.current_hand_id = hand_id

        # Cache positions
        if self._positions is None or game_state.dealer_seat:
            self._positions = game_state.infer_positions()

        # Capture preflop action when we're on preflop
        if game_state.street == "preflop" and not self._preflop_seen:
            self.preflop_action = reconstruct_preflop(game_state, self._positions)
            self._preflop_seen = True

        # If we missed preflop (joined mid-hand), infer from postflop
        if game_state.street != "preflop" and not self._preflop_seen:
            self.preflop_action = reconstruct_preflop(game_state, self._positions)
            self._preflop_seen = True

    def get_solver_inputs(self, game_state, range_lookup):
        # type: (...) -> Optional[Dict]
        """Get solver inputs using tracked state."""
        self.update(game_state)
        return determine_solver_inputs(game_state, range_lookup, hand_tracker=self)

    def _reset(self):
        self.current_hand_id = None
        self.preflop_action = None
        self._positions = None
        self._preflop_seen = False
