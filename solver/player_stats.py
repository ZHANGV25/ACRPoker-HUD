"""Player stat tracking from hand history data.

Stores per-player stats in SQLite and computes HUD stats:
  VPIP, PFR, 3bet%, Fold-to-Cbet%, WTSD%, AF, plus archetype.
"""

import os
import sqlite3
from typing import Optional, Dict, List, Tuple

# Default DB path
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "player_stats.db")


# ─────────────────────────────────────────────────────────────────────────────
# Archetype classification
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPES = {
    "fish":            "Loose passive — plays too many hands, rarely raises",
    "calling_station": "Loose passive — calls everything, hard to bluff",
    "nit":             "Very tight — only plays premium hands",
    "TAG":             "Tight aggressive — solid, selective, raises often",
    "LAG":             "Loose aggressive — plays many hands aggressively",
    "whale":           "Very loose, very passive — ATM machine",
    "maniac":          "Hyper-aggressive — raises almost everything",
    "unknown":         "Not enough data",
}


def classify_archetype(vpip, pfr, three_bet, n_hands):
    # type: (float, float, float, int) -> str
    """Classify player archetype from key stats.

    All stats in percentages (0-100).
    """
    if n_hands < 15:
        return "unknown"

    # Maniac: VPIP>55, PFR>35
    if vpip > 55 and pfr > 35:
        return "maniac"

    # Whale: VPIP>60, PFR<15
    if vpip > 60 and pfr < 15:
        return "whale"

    # Fish: VPIP>40, PFR<15
    if vpip > 40 and pfr < 15:
        return "fish"

    # Calling station: VPIP>35, PFR<12, high call ratio
    if vpip > 35 and pfr < 12:
        return "calling_station"

    # LAG: VPIP>28, PFR>22
    if vpip > 28 and pfr > 22:
        return "LAG"

    # TAG: VPIP 18-28, PFR>16
    if 18 <= vpip <= 28 and pfr > 16:
        return "TAG"

    # Nit: VPIP<16, PFR<12
    if vpip < 16 and pfr < 12:
        return "nit"

    # Default: use VPIP/PFR ratio
    if vpip > 30:
        return "fish"
    return "TAG"


# ─────────────────────────────────────────────────────────────────────────────
# Stats container
# ─────────────────────────────────────────────────────────────────────────────

class PlayerHUDStats:
    """Computed HUD stats for display."""
    __slots__ = (
        "name", "hands", "vpip", "pfr", "three_bet",
        "fold_to_cbet", "wtsd", "af", "archetype",
    )

    def __init__(self, name, hands=0, vpip=0.0, pfr=0.0, three_bet=0.0,
                 fold_to_cbet=0.0, wtsd=0.0, af=0.0, archetype="unknown"):
        self.name = name
        self.hands = hands
        self.vpip = vpip        # % (0-100)
        self.pfr = pfr          # % (0-100)
        self.three_bet = three_bet  # % (0-100)
        self.fold_to_cbet = fold_to_cbet  # % (0-100)
        self.wtsd = wtsd        # % (0-100)
        self.af = af            # ratio (typically 0-5)
        self.archetype = archetype

    def short_label(self):
        # type: () -> str
        """One-line HUD label: VPIP/PFR/3B (n) [type]."""
        if self.hands < 5:
            return ""
        return "{:.0f}/{:.0f}/{:.0f} ({}) {}".format(
            self.vpip, self.pfr, self.three_bet,
            self.hands, self.archetype)

    def __repr__(self):
        return "PlayerHUDStats({}, {})".format(self.name, self.short_label())


# ─────────────────────────────────────────────────────────────────────────────
# SQLite storage
# ─────────────────────────────────────────────────────────────────────────────

class StatsDB:
    """SQLite-backed player stats storage."""

    def __init__(self, db_path=None):
        # type: (Optional[str]) -> None
        self._db_path = db_path or DEFAULT_DB
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS hands_seen (
                hand_id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS player_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id TEXT NOT NULL,
                player TEXT NOT NULL,
                -- preflop flags
                vpip INTEGER DEFAULT 0,       -- 1 if voluntarily put $ in
                pfr INTEGER DEFAULT 0,        -- 1 if raised preflop
                three_bet INTEGER DEFAULT 0,  -- 1 if 3bet preflop
                -- postflop flags
                saw_flop INTEGER DEFAULT 0,
                went_to_showdown INTEGER DEFAULT 0,
                faced_cbet INTEGER DEFAULT 0,
                folded_to_cbet INTEGER DEFAULT 0,
                -- aggregate action counts (all streets)
                n_bets INTEGER DEFAULT 0,
                n_raises INTEGER DEFAULT 0,
                n_calls INTEGER DEFAULT 0,
                n_checks INTEGER DEFAULT 0,
                n_folds INTEGER DEFAULT 0,
                timestamp TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_pa_player ON player_actions(player);
            CREATE INDEX IF NOT EXISTS idx_pa_hand ON player_actions(hand_id);
        """)
        self._conn.commit()

    def has_hand(self, hand_id):
        # type: (str) -> bool
        cur = self._conn.execute(
            "SELECT 1 FROM hands_seen WHERE hand_id = ?", (hand_id,))
        return cur.fetchone() is not None

    def record_hand(self, parsed_hand):
        # type: (...) -> None
        """Record stats from a ParsedHand into the database."""
        from solver.hh_parser import ParsedHand
        h = parsed_hand  # type: ParsedHand

        if self.has_hand(h.hand_id):
            return

        self._conn.execute(
            "INSERT INTO hands_seen (hand_id) VALUES (?)", (h.hand_id,))

        # Determine who is in each seat
        all_players = {name for _, (name, _) in h.seats.items()}

        # Find preflop raiser sequence to detect 3bets
        pf_actions = h.preflop_actions()
        raise_count = 0
        first_raiser = None  # type: Optional[str]
        three_bettor = None  # type: Optional[str]
        vpip_players = set()  # type: set
        pfr_players = set()   # type: set
        three_bet_players = set()  # type: set

        for a in pf_actions:
            if a.action == "raise":
                raise_count += 1
                vpip_players.add(a.player)
                pfr_players.add(a.player)
                if raise_count == 1:
                    first_raiser = a.player
                elif raise_count == 2:
                    three_bettor = a.player
                    three_bet_players.add(a.player)
            elif a.action == "call":
                vpip_players.add(a.player)

        # Detect continuation bets on flop
        # A cbet is: the preflop raiser bets on the flop when checked to
        flop_actions = [a for a in h.actions if a.street == "flop"]
        cbet_player = None  # type: Optional[str]
        faced_cbet = set()  # type: set
        folded_to_cbet = set()  # type: set

        if first_raiser and flop_actions:
            # Check if first aggressive action on flop is by the preflop raiser
            for a in flop_actions:
                if a.action in ("bet", "raise") and a.player == first_raiser:
                    cbet_player = first_raiser
                    break
                elif a.action in ("bet", "raise"):
                    break  # someone else bet first, not a cbet scenario

            if cbet_player:
                # Everyone who acted after the cbet faced it
                saw_cbet = False
                for a in flop_actions:
                    if a.player == cbet_player and a.action in ("bet", "raise"):
                        saw_cbet = True
                        continue
                    if saw_cbet and a.player != cbet_player:
                        faced_cbet.add(a.player)
                        if a.action == "fold":
                            folded_to_cbet.add(a.player)

        # Count actions per player across all streets
        action_counts = {}  # type: Dict[str, Dict[str, int]]
        for a in h.actions:
            if a.player not in action_counts:
                action_counts[a.player] = {
                    "bet": 0, "raise": 0, "call": 0, "check": 0, "fold": 0
                }
            key = a.action
            if key in action_counts[a.player]:
                action_counts[a.player][key] += 1

        # Record per player
        for player in all_players:
            counts = action_counts.get(player, {
                "bet": 0, "raise": 0, "call": 0, "check": 0, "fold": 0
            })

            self._conn.execute("""
                INSERT INTO player_actions (
                    hand_id, player, vpip, pfr, three_bet,
                    saw_flop, went_to_showdown,
                    faced_cbet, folded_to_cbet,
                    n_bets, n_raises, n_calls, n_checks, n_folds,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                h.hand_id,
                player,
                1 if player in vpip_players else 0,
                1 if player in pfr_players else 0,
                1 if player in three_bet_players else 0,
                1 if h.saw_flop(player) else 0,
                1 if h.went_to_showdown(player) else 0,
                1 if player in faced_cbet else 0,
                1 if player in folded_to_cbet else 0,
                counts.get("bet", 0),
                counts.get("raise", 0),
                counts.get("call", 0),
                counts.get("check", 0),
                counts.get("fold", 0),
                h.timestamp,
            ))

        self._conn.commit()

    def get_stats(self, player):
        # type: (str) -> PlayerHUDStats
        """Compute HUD stats for a player."""
        cur = self._conn.execute("""
            SELECT
                COUNT(*) as n,
                SUM(vpip) as total_vpip,
                SUM(pfr) as total_pfr,
                SUM(three_bet) as total_3bet,
                SUM(saw_flop) as total_saw_flop,
                SUM(went_to_showdown) as total_wtsd,
                SUM(faced_cbet) as total_faced_cbet,
                SUM(folded_to_cbet) as total_folded_cbet,
                SUM(n_bets) as total_bets,
                SUM(n_raises) as total_raises,
                SUM(n_calls) as total_calls
            FROM player_actions
            WHERE player = ?
        """, (player,))

        row = cur.fetchone()
        if not row or row[0] == 0:
            return PlayerHUDStats(player)

        n = row[0]
        total_vpip = row[1] or 0
        total_pfr = row[2] or 0
        total_3bet = row[3] or 0
        total_saw_flop = row[4] or 0
        total_wtsd = row[5] or 0
        total_faced_cbet = row[6] or 0
        total_folded_cbet = row[7] or 0
        total_bets = row[8] or 0
        total_raises = row[9] or 0
        total_calls = row[10] or 0

        vpip = (total_vpip / n) * 100 if n else 0
        pfr = (total_pfr / n) * 100 if n else 0
        three_bet = (total_3bet / n) * 100 if n else 0
        wtsd = (total_wtsd / total_saw_flop) * 100 if total_saw_flop else 0
        fold_cbet = (total_folded_cbet / total_faced_cbet) * 100 if total_faced_cbet else 0
        af = (total_bets + total_raises) / total_calls if total_calls else 0.0

        archetype = classify_archetype(vpip, pfr, three_bet, n)

        return PlayerHUDStats(
            name=player,
            hands=n,
            vpip=vpip,
            pfr=pfr,
            three_bet=three_bet,
            fold_to_cbet=fold_cbet,
            wtsd=wtsd,
            af=af,
            archetype=archetype,
        )

    def get_all_stats(self, min_hands=1):
        # type: (int) -> Dict[str, PlayerHUDStats]
        """Get stats for all players with at least min_hands."""
        cur = self._conn.execute(
            "SELECT DISTINCT player FROM player_actions "
            "GROUP BY player HAVING COUNT(*) >= ?", (min_hands,))
        result = {}
        for (name,) in cur.fetchall():
            result[name] = self.get_stats(name)
        return result

    def bulk_import(self, parsed_hands):
        # type: (list) -> int
        """Import a list of ParsedHand objects. Returns number of new hands."""
        count = 0
        for h in parsed_hands:
            if not self.has_hand(h.hand_id):
                self.record_hand(h)
                count += 1
        return count

    def close(self):
        self._conn.close()
