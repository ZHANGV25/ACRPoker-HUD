"""Parse ACR hand history files into structured hand data."""

import re
from typing import Optional, List, Dict

# Regex patterns for ACR hand history format
RE_HAND_HEADER = re.compile(
    r"^Hand #(\d+) - Holdem \(No Limit\) - "
    r"\$(\d+\.\d+)/\$(\d+\.\d+) - "
    r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) UTC$"
)
RE_TABLE_BUTTON = re.compile(
    r"^(\S+) 6-max Seat #(\d+) is the button$"
)
RE_SEAT = re.compile(
    r"^Seat (\d+): (\S+) \(\$(\d+\.\d+)\)$"
)
RE_BLIND = re.compile(
    r"^(\S+) posts the (small|big) blind \$(\d+\.\d+)$"
)
RE_DEALT = re.compile(
    r"^Dealt to (\S+) \[(.+)\]$"
)
RE_ACTION = re.compile(
    r"^(\S+) (folds|calls|checks|bets|raises) ?"
    r"(?:\$(\d+\.\d+))?"
    r"(?: to \$(\d+\.\d+))?"
    r"(?: and is all-in)?$"
)
RE_SHOWS = re.compile(
    r"^(\S+) shows \[(.+?)\]"
)
RE_COLLECTED = re.compile(
    r"^(\S+) collected \$(\d+\.\d+) from"
)
RE_STREET = re.compile(
    r"^\*\*\* (HOLE CARDS|FLOP|TURN|RIVER|SHOW DOWN|SUMMARY) \*\*\*"
)
RE_BOARD = re.compile(
    r"^\*\*\* (?:FLOP|TURN|RIVER) \*\*\* \[(.+?)\]"
)
RE_UNCALLED = re.compile(
    r"^Uncalled bet \(\$(\d+\.\d+)\) returned to (\S+)$"
)


class HandAction:
    """A single player action in a hand."""
    __slots__ = ("player", "action", "amount", "street")

    def __init__(self, player, action, amount, street):
        # type: (str, str, float, str) -> None
        self.player = player
        self.action = action  # fold, call, check, bet, raise
        self.amount = amount  # dollar amount (0 for fold/check)
        self.street = street  # preflop, flop, turn, river


class ParsedHand:
    """Structured data from a single parsed hand."""
    __slots__ = (
        "hand_id", "sb", "bb", "timestamp", "table_name",
        "button_seat", "seats", "hero_name", "hero_cards",
        "actions", "board", "showdown_cards", "winners",
        "total_pot", "rake",
    )

    def __init__(self):
        self.hand_id = ""            # type: str
        self.sb = 0.0                # type: float
        self.bb = 0.0                # type: float
        self.timestamp = ""          # type: str
        self.table_name = ""         # type: str
        self.button_seat = 0         # type: int
        self.seats = {}              # type: Dict[int, tuple]  # seat -> (name, stack)
        self.hero_name = ""          # type: str
        self.hero_cards = ""         # type: str
        self.actions = []            # type: List[HandAction]
        self.board = []              # type: List[str]
        self.showdown_cards = {}     # type: Dict[str, str]  # player -> cards
        self.winners = []            # type: List[str]
        self.total_pot = 0.0         # type: float
        self.rake = 0.0              # type: float

    @property
    def num_players(self):
        # type: () -> int
        return len(self.seats)

    def player_seat(self, name):
        # type: (str) -> Optional[int]
        for seat, (n, _) in self.seats.items():
            if n == name:
                return seat
        return None

    def player_position(self, name):
        # type: (str) -> Optional[str]
        """Get position name for a player."""
        seat = self.player_seat(name)
        if seat is None:
            return None
        active_seats = sorted(self.seats.keys())
        n = len(active_seats)
        try:
            btn_idx = active_seats.index(self.button_seat)
        except ValueError:
            return None

        if n == 6:
            pos_names = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        elif n == 5:
            pos_names = ["BTN", "SB", "BB", "UTG", "CO"]
        elif n == 4:
            pos_names = ["BTN", "SB", "BB", "CO"]
        elif n == 3:
            pos_names = ["BTN", "SB", "BB"]
        elif n == 2:
            pos_names = ["BTN", "BB"]
        else:
            return None

        seat_idx = active_seats.index(seat)
        offset = (seat_idx - btn_idx) % n
        if offset < len(pos_names):
            return pos_names[offset]
        return None

    def preflop_actions(self):
        # type: () -> List[HandAction]
        return [a for a in self.actions if a.street == "preflop"]

    def saw_flop(self, player):
        # type: (str) -> bool
        """Did this player see the flop (didn't fold preflop)?"""
        for a in self.actions:
            if a.street != "preflop":
                break
            if a.player == player and a.action == "fold":
                return False
        # If they posted a blind and no preflop action at all (walked), check if flop exists
        if not self.board:
            return False
        # Check if player had any preflop action at all
        pf_players = {a.player for a in self.actions if a.street == "preflop"}
        if player not in pf_players:
            # Player was in blinds but didn't act (e.g. BB in a walk)
            return bool(self.board)
        return True

    def went_to_showdown(self, player):
        # type: (str) -> bool
        return player in self.showdown_cards


def parse_hand(lines):
    # type: (List[str]) -> Optional[ParsedHand]
    """Parse a single hand from a list of text lines."""
    if not lines:
        return None

    hand = ParsedHand()
    current_street = "preflop"
    blind_players = set()  # type: set

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Hand header
        m = RE_HAND_HEADER.match(line)
        if m:
            hand.hand_id = m.group(1)
            hand.sb = float(m.group(2))
            hand.bb = float(m.group(3))
            hand.timestamp = m.group(4)
            continue

        # Table + button
        m = RE_TABLE_BUTTON.match(line)
        if m:
            hand.table_name = m.group(1)
            hand.button_seat = int(m.group(2))
            continue

        # Seat
        m = RE_SEAT.match(line)
        if m:
            seat = int(m.group(1))
            name = m.group(2)
            stack = float(m.group(3))
            hand.seats[seat] = (name, stack)
            continue

        # Blinds
        m = RE_BLIND.match(line)
        if m:
            blind_players.add(m.group(1))
            continue

        # Dealt to hero
        m = RE_DEALT.match(line)
        if m:
            hand.hero_name = m.group(1)
            hand.hero_cards = m.group(2)
            continue

        # Street markers + board cards (same line for FLOP/TURN/RIVER)
        m = RE_STREET.match(line)
        if m:
            street_name = m.group(1)
            if street_name == "FLOP":
                current_street = "flop"
            elif street_name == "TURN":
                current_street = "turn"
            elif street_name == "RIVER":
                current_street = "river"
            elif street_name == "SHOW DOWN":
                current_street = "showdown"
            elif street_name == "SUMMARY":
                current_street = "summary"

            # Also parse board cards from the same line
            full_match = re.findall(r"\[(.+?)\]", line)
            if full_match:
                board = []
                for group in full_match:
                    board.extend(group.split())
                hand.board = board
            continue

        # Showdown cards
        if current_street == "showdown":
            m = RE_SHOWS.match(line)
            if m:
                hand.showdown_cards[m.group(1)] = m.group(2)
                continue
            m = RE_COLLECTED.match(line)
            if m:
                hand.winners.append(m.group(1))
                continue

        # Summary - total pot
        if current_street == "summary":
            if line.startswith("Total pot"):
                pot_match = re.search(r"Total pot \$(\d+\.\d+)", line)
                rake_match = re.search(r"Rake \$(\d+\.\d+)", line)
                if pot_match:
                    hand.total_pot = float(pot_match.group(1))
                if rake_match:
                    hand.rake = float(rake_match.group(1))
            continue

        # Skip non-action lines
        if current_street in ("showdown", "summary"):
            continue
        if line.startswith("Uncalled bet") or line.startswith("Main pot"):
            continue
        if "waits for" in line or "sits out" in line:
            continue
        if "does not show" in line or "mucks" in line:
            continue

        # Player actions
        m = RE_ACTION.match(line)
        if m:
            player = m.group(1)
            action = m.group(2)
            # amount: for raises "to X", use the to-amount; otherwise the direct amount
            amount = 0.0
            if m.group(4):
                amount = float(m.group(4))
            elif m.group(3):
                amount = float(m.group(3))

            # Map action names
            action_map = {
                "folds": "fold",
                "calls": "call",
                "checks": "check",
                "bets": "bet",
                "raises": "raise",
            }
            hand.actions.append(HandAction(
                player=player,
                action=action_map.get(action, action),
                amount=amount,
                street=current_street,
            ))
            continue

    if not hand.hand_id:
        return None
    return hand


def parse_file(filepath):
    # type: (str) -> List[ParsedHand]
    """Parse all hands from an ACR hand history file."""
    hands = []
    current_lines = []  # type: List[str]

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if RE_HAND_HEADER.match(line):
                if current_lines:
                    hand = parse_hand(current_lines)
                    if hand:
                        hands.append(hand)
                current_lines = [line]
            else:
                current_lines.append(line)

    if current_lines:
        hand = parse_hand(current_lines)
        if hand:
            hands.append(hand)

    return hands
