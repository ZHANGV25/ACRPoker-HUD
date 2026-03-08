"""Pre-solve all strategically unique flop textures.

There are C(52,3) = 22,100 possible flops. Using suit isomorphism
(only the relative suit pattern matters, not the actual suits), this
reduces to ~1,755 strategically unique flop textures.

For each flop, we solve for a standard scenario (e.g., BTN open vs BB call)
and store the full game tree solution. At runtime, we look up the precomputed
solution instead of solving live (which is too slow for flop).

Usage:
    python -m solver.precompute_flops [--workers N] [--output DIR]
    python -m solver.precompute_flops --list-only   # just print flop count

Output:
    One JSON file per flop texture in the output directory.
    Filename: "{texture_key}.json" e.g. "AKQ_r" (rainbow), "AK5_fd" (flush draw)
"""

import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ── Card / rank / suit constants ─────────────────────────────────────────────

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}
SUITS = ['c', 'd', 'h', 's']
DECK = [r + s for r in RANKS for s in SUITS]

# Path to solver binary
SOLVER_BIN = os.path.join(
    os.path.dirname(__file__), "solver-cli", "target", "release",
    "tbl-engine.exe" if sys.platform == "win32" else "tbl-engine"
)


# ── Flop texture classification ──────────────────────────────────────────────

def rank_pattern(cards):
    """Classify the rank pattern of 3 cards.

    Returns tuple of (sorted_ranks_desc, pattern_type) where pattern_type is:
      'trips' — all 3 same rank (e.g. 777)
      'pair'  — exactly 2 same rank (e.g. KK5)
      'unpaired' — all different ranks (e.g. AK7)
    """
    ranks = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    if ranks[0] == ranks[1] == ranks[2]:
        return tuple(ranks), 'trips'
    elif ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        return tuple(ranks), 'pair'
    else:
        return tuple(ranks), 'unpaired'


def suit_pattern(cards):
    """Classify the suit pattern of 3 cards.

    Returns:
      'monotone' — all 3 same suit (e.g. AhKhQh)
      'flush_draw' — exactly 2 same suit (e.g. AhKh7s)
      'rainbow' — all different suits (e.g. AhKdQs)
    """
    suits = [c[1] for c in cards]
    unique = len(set(suits))
    if unique == 1:
        return 'monotone'
    elif unique == 2:
        return 'flush_draw'
    else:
        return 'rainbow'


def texture_key(cards):
    """Generate a canonical texture key for a flop.

    Combines rank pattern + suit pattern. For flush draw boards with
    all different ranks, we also encode WHICH two cards share the suit
    (top-mid, top-low, mid-low) since this is strategically different.

    E.g. "AKQ_m" (monotone), "AK5_fd12" (top two suited), "T72_r" (rainbow)
    """
    ranks = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    rank_str = ''.join(RANKS[r] for r in ranks)
    sp = suit_pattern(cards)

    if sp == 'monotone':
        return "{}_m".format(rank_str)
    elif sp == 'rainbow':
        return "{}_r".format(rank_str)
    else:
        # flush_draw: which two cards share the suit?
        # Sort cards by rank descending to get positional indices
        sorted_cards = sorted(cards, key=lambda c: RANK_VALUE[c[0]], reverse=True)
        suits = [c[1] for c in sorted_cards]

        # All ranks same (trips) — shouldn't be flush_draw, but handle it
        if ranks[0] == ranks[1] == ranks[2]:
            return "{}_fd".format(rank_str)

        # Paired board — only one flush draw sub-type
        if ranks[0] == ranks[1] or ranks[1] == ranks[2]:
            return "{}_fd".format(rank_str)

        # Unpaired: 3 sub-types based on which pair shares suit
        if suits[0] == suits[1]:
            return "{}_fd12".format(rank_str)  # top + mid suited
        elif suits[0] == suits[2]:
            return "{}_fd13".format(rank_str)  # top + low suited
        else:
            return "{}_fd23".format(rank_str)  # mid + low suited


def generate_unique_flop_textures():
    """Generate all strategically unique flop textures.

    Uses suit isomorphism: only the rank pattern + suit pattern matters.
    Returns dict: texture_key -> representative board (3 cards).
    """
    seen = set()
    textures = {}

    for combo in itertools.combinations(range(52), 3):
        cards = [DECK[i] for i in combo]
        key = texture_key(cards)
        if key not in seen:
            seen.add(key)
            textures[key] = cards

    return textures


# ── Representative board cards for each texture ──────────────────────────────

def canonical_board(key, cards):
    """Create canonical board cards for a texture.

    Assigns specific suits based on the texture sub-type:
    - rainbow (_r): spade, heart, diamond
    - monotone (_m): all spades
    - flush_draw (_fd): paired boards → pair card + kicker suited
    - flush_draw (_fd12): top + mid suited (spade, spade, heart)
    - flush_draw (_fd13): top + low suited (spade, heart, spade)
    - flush_draw (_fd23): mid + low suited (heart, spade, spade)
    """
    ranks = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    rank_chars = [RANKS[r] for r in ranks]

    suffix = key.split('_', 1)[1]
    if suffix == 'r':
        return [rank_chars[0] + 's', rank_chars[1] + 'h', rank_chars[2] + 'd']
    elif suffix == 'm':
        return [rank_chars[0] + 's', rank_chars[1] + 's', rank_chars[2] + 's']
    elif suffix == 'fd12':
        return [rank_chars[0] + 's', rank_chars[1] + 's', rank_chars[2] + 'h']
    elif suffix == 'fd13':
        return [rank_chars[0] + 's', rank_chars[1] + 'h', rank_chars[2] + 's']
    elif suffix == 'fd23':
        return [rank_chars[0] + 'h', rank_chars[1] + 's', rank_chars[2] + 's']
    elif suffix == 'fd':
        # Paired board flush draw — one pair card + kicker suited
        return [rank_chars[0] + 's', rank_chars[1] + 'h', rank_chars[2] + 's']
    else:
        return cards


# ── Default scenario for precomputation ──────────────────────────────────────

def load_ranges():
    """Load default ranges from ranges.json."""
    ranges_path = os.path.join(os.path.dirname(__file__), "ranges.json")
    with open(ranges_path) as f:
        return json.load(f)


def get_default_scenario(ranges_data):
    """Get the most common postflop scenario: BTN open vs BB call.

    Returns (oop_range, ip_range) strings.
    """
    # BB is OOP, BTN is IP
    oop_range = ranges_data.get("vs_rfi", {}).get("BB_vs_BTN", {}).get("call", "")
    ip_range = ranges_data.get("rfi", {}).get("BTN", "")

    if not oop_range or not ip_range:
        raise ValueError("Could not load BTN vs BB ranges from ranges.json")

    return oop_range, ip_range


# Standard postflop pot and stacks (BTN open 2.5BB, BB calls → 5BB pot)
DEFAULT_STARTING_POT = 5.0
DEFAULT_EFFECTIVE_STACK = 97.5  # 100BB - 2.5BB open


# ── Solver invocation ────────────────────────────────────────────────────────

def solve_flop(board, oop_range, ip_range, starting_pot=DEFAULT_STARTING_POT,
               effective_stack=DEFAULT_EFFECTIVE_STACK,
               max_iterations=500, target_exploitability=0.01):
    """Run the solver for a single flop texture.

    We use a dummy hero hand (doesn't affect the game tree solution)
    since we're precomputing the full strategy, not a specific hand.

    Returns the solver output dict or None on error.
    """
    # Pick a hero hand not on the board
    board_set = set(board)
    hero_hand = None
    for combo in itertools.combinations(DECK, 2):
        if combo[0] not in board_set and combo[1] not in board_set:
            hero_hand = list(combo)
            break

    if hero_hand is None:
        return None

    solver_input = {
        "board": board,
        "oop_range": oop_range,
        "ip_range": ip_range,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "hero_hand": hero_hand,
        "hero_position": "oop",
        "max_iterations": max_iterations,
        "target_exploitability": target_exploitability,
    }

    try:
        proc = subprocess.run(
            [SOLVER_BIN],
            input=json.dumps(solver_input),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per flop
        )
        if proc.returncode != 0:
            sys.stderr.write("[precompute] Solver error for {}: {}\n".format(
                board, proc.stderr[:200]))
            return None

        result = json.loads(proc.stdout)
        result["board"] = board
        result["solver_input"] = solver_input
        return result

    except subprocess.TimeoutExpired:
        sys.stderr.write("[precompute] Timeout for {}\n".format(board))
        return None
    except Exception as e:
        sys.stderr.write("[precompute] Exception for {}: {}\n".format(board, e))
        return None


def solve_one(args):
    """Worker function for parallel execution."""
    key, board, oop_range, ip_range = args
    result = solve_flop(board, oop_range, ip_range)
    return key, result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-solve all unique flop textures")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel solver processes (default: 2)")
    parser.add_argument("--output", type=str, default="solver/flop_solutions",
                        help="Output directory for solution files")
    parser.add_argument("--list-only", action="store_true",
                        help="Just list unique textures without solving")
    parser.add_argument("--max-iterations", type=int, default=500,
                        help="Solver iterations per flop (default: 500)")
    args = parser.parse_args()

    # Generate textures
    textures = generate_unique_flop_textures()
    print("Unique flop textures: {}".format(len(textures)))

    if args.list_only:
        # Print summary
        by_suit = defaultdict(int)
        for key in textures:
            sp = key.split('_', 1)[1]
            by_suit[sp] += 1
        for sp, count in sorted(by_suit.items()):
            print("  {}: {}".format(sp, count))
        return

    # Check solver binary
    if not os.path.exists(SOLVER_BIN):
        print("ERROR: Solver binary not found at {}".format(SOLVER_BIN))
        print("Run: cd solver/solver-cli && cargo build --release")
        sys.exit(1)

    # Load ranges
    ranges_data = load_ranges()
    oop_range, ip_range = get_default_scenario(ranges_data)
    print("Scenario: BTN open vs BB call")
    print("OOP range: {} combos".format(len(oop_range.split(','))))
    print("IP range: {} combos".format(len(ip_range.split(','))))

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Check which textures are already solved
    already_solved = set()
    for fname in os.listdir(args.output):
        if fname.endswith('.json'):
            already_solved.add(fname[:-5])

    to_solve = [(k, v) for k, v in textures.items() if k not in already_solved]
    print("Already solved: {}".format(len(already_solved)))
    print("Remaining: {}".format(len(to_solve)))

    if not to_solve:
        print("All flops already solved!")
        return

    # Prepare work items
    work = []
    for key, cards in to_solve:
        board = canonical_board(key, cards)
        work.append((key, board, oop_range, ip_range))

    # Solve
    solved = 0
    failed = 0
    start_time = time.time()

    print("\nSolving {} flops with {} workers...\n".format(len(work), args.workers))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve_one, w): w[0] for w in work}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result_key, result = future.result()
                if result:
                    out_path = os.path.join(args.output, "{}.json".format(result_key))
                    with open(out_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    solved += 1
                else:
                    failed += 1
            except Exception as e:
                sys.stderr.write("[precompute] Failed {}: {}\n".format(key, e))
                failed += 1

            total = solved + failed
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            remaining = (len(work) - total) / rate if rate > 0 else 0
            print("\r  [{}/{}] solved={} failed={} rate={:.1f}/min ETA={:.0f}min".format(
                total, len(work), solved, failed, rate * 60, remaining / 60),
                end="", flush=True)

    elapsed = time.time() - start_time
    print("\n\nDone! Solved {} flops in {:.1f} minutes ({} failed)".format(
        solved, elapsed / 60, failed))


if __name__ == "__main__":
    main()
