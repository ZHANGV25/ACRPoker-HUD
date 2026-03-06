"""Game state model and assembly from OCR outputs."""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
import json


@dataclass
class PlayerState:
    seat: int
    name: Optional[str] = None
    stack_bb: Optional[float] = None
    action_label: Optional[str] = None  # e.g. "R/B", "C/X", "F"
    current_bet_bb: Optional[float] = None
    is_hero: bool = False
    is_folded: bool = False
    is_sitting_out: bool = False


@dataclass
class GameState:
    hand_id: Optional[str] = None
    stakes: Optional[str] = None  # e.g. "$0.01/$0.02"
    hero_cards: List[str] = field(default_factory=list)  # e.g. ["Jc", "Jh"]
    board: List[str] = field(default_factory=list)  # e.g. ["Qd", "Kc", "Tc"]
    pot_bb: Optional[float] = None
    total_bb: Optional[float] = None
    players: List[PlayerState] = field(default_factory=list)
    street: str = "preflop"  # preflop, flop, turn, river
    available_actions: dict = field(default_factory=dict)
    dealer_seat: Optional[int] = None

    def infer_street(self):
        """Infer current street from board card count."""
        n = len(self.board)
        if n == 0:
            self.street = "preflop"
        elif n == 3:
            self.street = "flop"
        elif n == 4:
            self.street = "turn"
        elif n == 5:
            self.street = "river"

    def infer_positions(self) -> Dict[int, str]:
        """Infer player positions based on dealer button seat.

        Returns mapping of seat number -> position name.
        """
        if self.dealer_seat is None:
            return {}

        active_seats = sorted([p.seat for p in self.players if not p.is_sitting_out])
        if not active_seats:
            return {}

        # Find dealer index in active seats
        try:
            dealer_idx = active_seats.index(self.dealer_seat)
        except ValueError:
            return {}

        n = len(active_seats)
        # Position names for 6-max, clockwise from dealer
        if n == 6:
            pos_names = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        elif n == 5:
            pos_names = ["BTN", "SB", "BB", "UTG", "CO"]
        elif n == 4:
            pos_names = ["BTN", "SB", "BB", "CO"]
        elif n == 3:
            pos_names = ["BTN", "SB", "BB"]
        elif n == 2:
            pos_names = ["BTN", "BB"]  # Heads up: BTN is SB
        else:
            pos_names = [f"S{i}" for i in range(n)]

        positions = {}
        for i, pos_name in enumerate(pos_names):
            seat = active_seats[(dealer_idx + i) % n]
            positions[seat] = pos_name

        return positions

    def to_solver_input(self) -> dict:
        """Convert to JSON-serializable dict for sending to solver."""
        positions = self.infer_positions()

        players_data = []
        for p in self.players:
            if p.is_sitting_out:
                continue
            players_data.append({
                "seat": p.seat,
                "position": positions.get(p.seat, "?"),
                "stack_bb": p.stack_bb,
                "action_label": p.action_label,
                "current_bet_bb": p.current_bet_bb,
                "is_hero": p.is_hero,
                "is_folded": p.is_folded,
            })

        return {
            "hand_id": self.hand_id,
            "stakes": self.stakes,
            "hero_cards": self.hero_cards,
            "board": self.board,
            "pot_bb": self.pot_bb,
            "total_bb": self.total_bb,
            "street": self.street,
            "players": players_data,
            "available_actions": self.available_actions,
            "dealer_seat": self.dealer_seat,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_solver_input(), indent=2)
