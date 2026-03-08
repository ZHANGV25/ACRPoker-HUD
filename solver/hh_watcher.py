"""Watch ACR hand history directory for new hands and update player stats.

Tails hand history files and feeds new hands to the stats DB as they're
appended. Designed to run in a background thread alongside the main HUD.
"""

import os
import time
import threading
import glob as glob_mod
from typing import Optional, Dict

from solver.hh_parser import parse_hand, RE_HAND_HEADER
from solver.player_stats import StatsDB, PlayerHUDStats
from solver.fuzzy_name import fuzzy_match

# Default ACR hand history directory
DEFAULT_HH_DIR = os.path.expanduser(
    "~/Downloads/AmericasCardroom/handHistory/vortexted/"
)

POLL_INTERVAL = 2.0  # seconds between checks


class HHWatcher:
    """Watches hand history files and maintains live player stats.

    Usage:
        watcher = HHWatcher()
        watcher.start()  # background thread
        stats = watcher.get_player_stats("PlayerName")
    """

    def __init__(self, hh_dir=None, db_path=None):
        # type: (Optional[str], Optional[str]) -> None
        self._hh_dir = hh_dir or DEFAULT_HH_DIR
        self._db = StatsDB(db_path)
        self._file_positions = {}  # type: Dict[str, int]  # filepath -> last read position
        self._running = False
        self._thread = None  # type: Optional[threading.Thread]
        self._lock = threading.Lock()
        self._stats_cache = {}  # type: Dict[str, PlayerHUDStats]
        self._name_resolve_cache = {}  # type: Dict[str, Optional[str]]  # ocr_name -> hh_name
        self._known_names = set()  # type: set  # all player names from hand history
        self._recent_tables = {}  # type: Dict[str, dict]  # table_name -> {seat: name} from last hand
        self._cache_dirty = True
        self._hands_imported = 0

    def start(self):
        # type: () -> None
        """Start watching in a background thread."""
        if self._running:
            return
        self._running = True
        # Load all known names from DB (covers previous sessions)
        self._known_names.update(self._db.all_player_names())
        # Import any new hands from HH files
        self._initial_import()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # type: () -> None
        self._running = False

    def _initial_import(self):
        # type: () -> None
        """Bulk import all existing hand history files."""
        if not os.path.isdir(self._hh_dir):
            return
        files = sorted(glob_mod.glob(os.path.join(self._hh_dir, "HH*.txt")))
        total = 0
        for filepath in files:
            n = self._process_file(filepath)
            total += n
        if total:
            self._hands_imported += total
            self._cache_dirty = True

    def _watch_loop(self):
        # type: () -> None
        """Poll for new hand history data."""
        while self._running:
            try:
                self._poll_files()
            except Exception:
                pass
            time.sleep(POLL_INTERVAL)

    def _poll_files(self):
        # type: () -> None
        """Check all HH files for new data."""
        if not os.path.isdir(self._hh_dir):
            return
        files = glob_mod.glob(os.path.join(self._hh_dir, "HH*.txt"))
        for filepath in files:
            n = self._tail_file(filepath)
            if n > 0:
                self._hands_imported += n
                self._cache_dirty = True
                self._name_resolve_cache.clear()  # new names may have appeared

    def _process_file(self, filepath):
        # type: (str) -> int
        """Process an entire file (for initial import). Returns hands imported."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                self._file_positions[filepath] = len(content.encode("utf-8", errors="replace"))
        except (IOError, OSError):
            return 0

        hands = _parse_content(content)
        count = 0
        for h in hands:
            # Collect known player names and table seat mappings
            for seat, (name, _) in h.seats.items():
                self._known_names.add(name)
            if h.table_name:
                self._recent_tables[h.table_name] = {
                    seat: name for seat, (name, _) in h.seats.items()
                }
            if not self._db.has_hand(h.hand_id):
                self._db.record_hand(h)
                count += 1
        return count

    def _tail_file(self, filepath):
        # type: (str) -> int
        """Read new data appended to a file since last check."""
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            return 0

        last_pos = self._file_positions.get(filepath, 0)
        if file_size <= last_pos:
            return 0

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_data = f.read()
                self._file_positions[filepath] = f.tell()
        except (IOError, OSError):
            return 0

        if not new_data.strip():
            return 0

        hands = _parse_content(new_data)
        count = 0
        for h in hands:
            for seat, (name, _) in h.seats.items():
                self._known_names.add(name)
            if h.table_name:
                self._recent_tables[h.table_name] = {
                    seat: name for seat, (name, _) in h.seats.items()
                }
            if not self._db.has_hand(h.hand_id):
                self._db.record_hand(h)
                count += 1
        return count

    def _resolve_name(self, ocr_name):
        # type: (str) -> Optional[str]
        """Resolve an OCR-read name to a known hand history name."""
        if not ocr_name:
            return None
        # Check cache first
        if ocr_name in self._name_resolve_cache:
            return self._name_resolve_cache[ocr_name]
        resolved = fuzzy_match(ocr_name, self._known_names)
        self._name_resolve_cache[ocr_name] = resolved
        return resolved

    def get_player_stats(self, ocr_name):
        # type: (str) -> PlayerHUDStats
        """Get stats for a player, using fuzzy name matching for OCR errors."""
        resolved = self._resolve_name(ocr_name)
        lookup = resolved or ocr_name
        with self._lock:
            if lookup in self._stats_cache and not self._cache_dirty:
                return self._stats_cache[lookup]
        stats = self._db.get_stats(lookup)
        with self._lock:
            self._stats_cache[lookup] = stats
        return stats

    def get_all_stats(self, min_hands=5):
        # type: (int) -> Dict[str, PlayerHUDStats]
        """Get stats for all players."""
        if self._cache_dirty:
            with self._lock:
                self._stats_cache = self._db.get_all_stats(min_hands)
                self._cache_dirty = False
        return dict(self._stats_cache)

    def get_table_stats(self, player_names):
        # type: (list) -> Dict[str, PlayerHUDStats]
        """Get stats for specific players at current table."""
        result = {}
        for name in player_names:
            if name:
                result[name] = self.get_player_stats(name)
        return result

    def get_table_names(self, table_name):
        # type: (str) -> Dict[int, str]
        """Get seat->name mapping from the most recent hand on this table.

        table_name: extracted from the ACR window title (e.g. "Heyburn").
        Returns {seat_num: player_name} or empty dict.
        """
        return dict(self._recent_tables.get(table_name, {}))

    @property
    def total_hands(self):
        # type: () -> int
        return self._hands_imported

    def close(self):
        self.stop()
        self._db.close()


def _parse_content(content):
    # type: (str) -> list
    """Parse hand history content string into list of ParsedHand."""
    from solver.hh_parser import parse_hand, RE_HAND_HEADER
    lines = content.split("\n")
    hands = []
    current_lines = []

    for line in lines:
        if RE_HAND_HEADER.match(line):
            if current_lines:
                h = parse_hand(current_lines)
                if h:
                    hands.append(h)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        h = parse_hand(current_lines)
        if h:
            hands.append(h)

    return hands
