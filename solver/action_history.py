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
    "oop_bet": "33%, 66%, a",
    "oop_raise": "2.5x",
    "ip_bet": "33%, 66%, a",
    "ip_raise": "2.5x",
}

# Standard buckets for observed bet sizes
_SIZE_BUCKETS = [25, 33, 50, 66, 75, 100, 150]


def _snap_to_bucket(pct):
    # type: (float) -> int
    """Snap an observed bet-to-pot percentage to the nearest standard bucket."""
    best = min(_SIZE_BUCKETS, key=lambda b: abs(b - pct))
    return best


def compute_bet_sizes(observed_bets, default=None):
    # type: (List[float], Optional[Dict]) -> Dict[str, str]
    """Convert observed bet-to-pot ratios into solver bet size strings.

    Args:
        observed_bets: list of bet/pot ratios as percentages (e.g. [33.0, 75.0])
        default: fallback sizes dict

    Returns:
        Dict with keys bet_sizes_oop, bet_sizes_ip, raise_sizes_oop, raise_sizes_ip
    """
    if default is None:
        default = DEFAULT_BET_SIZES

    if not observed_bets:
        return {
            "bet_sizes_oop": default["oop_bet"],
            "bet_sizes_ip": default["ip_bet"],
            "raise_sizes_oop": default["oop_raise"],
            "raise_sizes_ip": default["ip_raise"],
        }

    # Snap observed bets to standard buckets, deduplicate
    buckets = sorted(set(_snap_to_bucket(b) for b in observed_bets))
    # Always include all-in
    bet_str = ", ".join("{}%".format(b) for b in buckets) + ", a"

    return {
        "bet_sizes_oop": bet_str,
        "bet_sizes_ip": bet_str,
        "raise_sizes_oop": default["oop_raise"],
        "raise_sizes_ip": default["ip_raise"],
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

    # Ensure hero's hand is in their range so the solver can find it.
    # Combo must be higher-rank first (e.g. "9s2h" not "2h9s").
    if hero_hand and len(hero_hand) == 4 and hero_position:
        RANK_ORDER = "23456789TJQKA"
        r1, s1, r2, s2 = hero_hand[0], hero_hand[1], hero_hand[2], hero_hand[3]
        if RANK_ORDER.index(r1) >= RANK_ORDER.index(r2):
            hero_combo = hero_hand  # already correct order
        else:
            hero_combo = r2 + s2 + r1 + s1  # swap to higher rank first
        target_range = oop_range if hero_position == "oop" else ip_range
        if hero_combo not in target_range:
            if hero_position == "oop":
                oop_range = oop_range + "," + hero_combo
            else:
                ip_range = ip_range + "," + hero_combo

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
    Locks dealer seat and solver matchup for the duration of a hand.
    Tracks postflop actions and observed bet sizes.
    """

    def __init__(self):
        self.current_hand_id = None  # type: Optional[str]
        self.preflop_action = None   # type: Optional[PreflopAction]
        self._positions = None       # type: Optional[Dict]
        self._preflop_seen = False
        self._locked_dealer = None   # type: Optional[int]
        self._locked_solver = None   # type: Optional[Dict]
        # Postflop tracking
        self._last_street = "preflop"
        self._prev_bets = {}  # type: Dict[int, Optional[float]]  # seat -> last bet
        self._observed_bet_pcts = []  # type: List[float]  # bet-to-pot ratios
        self.postflop_actions = {}  # type: Dict[str, List[Tuple[str, str, Optional[float]]]]
        # street -> [(position, action, amount)]

    def update(self, game_state):
        # type: (...) -> None
        """Process a new game state snapshot."""
        hand_id = game_state.hand_id

        # Detect new hand
        if hand_id != self.current_hand_id:
            self._reset()
            self.current_hand_id = hand_id

        # Lock dealer seat from the first valid read
        if self._locked_dealer is None and game_state.dealer_seat:
            self._locked_dealer = game_state.dealer_seat
        # Override game state dealer with locked value
        if self._locked_dealer is not None:
            game_state.dealer_seat = self._locked_dealer

        # Cache positions (uses locked dealer)
        if self._positions is None:
            self._positions = game_state.infer_positions()

        # Capture preflop action when we're on preflop
        if game_state.street == "preflop" and not self._preflop_seen:
            self.preflop_action = reconstruct_preflop(game_state, self._positions)
            self._preflop_seen = True

        # If we missed preflop (joined mid-hand), infer from postflop
        if game_state.street != "preflop" and not self._preflop_seen:
            self.preflop_action = reconstruct_preflop(game_state, self._positions)
            self._preflop_seen = True

        # Track postflop actions and bet sizes
        self._track_postflop(game_state)

    def get_solver_inputs(self, game_state, range_lookup):
        # type: (...) -> Optional[Dict]
        """Get solver inputs using tracked state.

        Locks the IP/OOP matchup and ranges on first computation so that
        villain selection doesn't flip mid-hand (e.g. when a player shoves
        and their stack drops to 0). Pot and effective stack update each frame.
        """
        self.update(game_state)
        result = determine_solver_inputs(game_state, range_lookup, hand_tracker=self)
        if result is None:
            # If fresh computation failed, return locked result with updated pot
            if self._locked_solver is not None:
                return self._locked_solver
            return None
        if self._locked_solver is None:
            # First successful computation — lock matchup and ranges
            self._locked_solver = result
        else:
            # Keep locked matchup/ranges, update dynamic fields
            result["oop_range"] = self._locked_solver["oop_range"]
            result["ip_range"] = self._locked_solver["ip_range"]
            result["oop_position"] = self._locked_solver["oop_position"]
            result["ip_position"] = self._locked_solver["ip_position"]
            result["hero_position"] = self._locked_solver["hero_position"]
        return result

    def _track_postflop(self, game_state):
        # type: (...) -> None
        """Track postflop actions and observed bet sizes."""
        street = game_state.street
        if street == "preflop":
            return

        # Detect street change
        if street != self._last_street:
            self._prev_bets.clear()
            self._last_street = street
            if street not in self.postflop_actions:
                self.postflop_actions[street] = []

        # Check each player for new bets/actions
        pot = game_state.total_bb or game_state.pot_bb or 0
        for p in game_state.players:
            if p.is_sitting_out or p.is_folded:
                continue
            prev_bet = self._prev_bets.get(p.seat)
            curr_bet = p.current_bet_bb or 0

            # Detect new bet/raise
            if curr_bet > 0 and curr_bet != prev_bet:
                pos = self._positions.get(p.seat, "?") if self._positions else "?"
                action = _normalize_action(p.action_label) or "bet"

                # Record action
                if street not in self.postflop_actions:
                    self.postflop_actions[street] = []
                self.postflop_actions[street].append((pos, action, curr_bet))

                # Record observed bet-to-pot ratio
                if pot > 0:
                    pct = (curr_bet / pot) * 100
                    if 5 < pct < 300:  # sane range
                        self._observed_bet_pcts.append(pct)

            self._prev_bets[p.seat] = curr_bet if curr_bet > 0 else prev_bet

    def get_bet_sizes(self):
        # type: () -> Dict[str, str]
        """Return solver bet size strings based on observed play."""
        return compute_bet_sizes(self._observed_bet_pcts)

    def get_street_actions(self, game_state, solver_inputs):
        # type: (...) -> List[Dict[str, object]]
        """Build street_actions for the solver based on current bets.

        Returns list of {"action": str, "amount": float} dicts representing
        actions already played on this street before hero's decision.
        The solver expects OOP to act first.
        """
        if not solver_inputs or game_state.street == "preflop":
            return []

        oop_pos = solver_inputs["oop_position"]
        ip_pos = solver_inputs["ip_position"]
        hero_position = solver_inputs.get("hero_position", "")

        # Find current bets for OOP and IP players
        positions = self._positions or game_state.infer_positions()
        oop_bet = 0.0
        ip_bet = 0.0
        for p in game_state.players:
            if p.is_sitting_out or p.is_folded:
                continue
            pos = positions.get(p.seat, "")
            bet = p.current_bet_bb or 0.0
            if pos == oop_pos:
                oop_bet = bet
            elif pos == ip_pos:
                ip_bet = bet

        # If OCR missed villain's bet but hero has a Call option, infer the bet
        aa = game_state.available_actions or {}
        call_amount = aa.get("call")
        if call_amount is not None and call_amount > 0 and oop_bet == 0 and ip_bet == 0:
            # Someone bet but we didn't detect it — use call amount
            if hero_position == "oop":
                ip_bet = call_amount
            elif hero_position == "ip":
                oop_bet = call_amount

        actions = []  # type: List[Dict[str, object]]

        if oop_bet == 0 and ip_bet == 0:
            # No bets yet — we're at the root
            return []

        if hero_position == "oop":
            # Hero is OOP and it's their turn.
            # If IP has bet, OOP must have checked first, then IP bet.
            if ip_bet > 0:
                actions.append({"action": "check", "amount": 0.0})
                actions.append({"action": "bet", "amount": ip_bet})
            # If only OOP has bet, that means OOP already bet — we're past that
            # (hero wouldn't have action buttons if they already bet)
            return actions

        elif hero_position == "ip":
            # Hero is IP and it's their turn.
            if oop_bet > 0:
                # OOP bet, hero faces it
                actions.append({"action": "bet", "amount": oop_bet})
            # If no OOP bet, hero is acting first as IP (after OOP check)
            # That's the natural root flow: OOP checks, then IP acts
            # But the solver tree starts with OOP, so we need "check"
            elif oop_bet == 0 and ip_bet == 0:
                # OOP checked, IP to act — need to play the check
                actions.append({"action": "check", "amount": 0.0})
            return actions

        return []

    def _reset(self):
        self.current_hand_id = None
        self.preflop_action = None
        self._positions = None
        self._preflop_seen = False
        self._locked_dealer = None
        self._locked_solver = None
        self._last_street = "preflop"
        self._prev_bets = {}
        self._observed_bet_pcts = []
        self.postflop_actions = {}
