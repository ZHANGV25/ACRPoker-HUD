# ACR Poker OCR + GTO Solver Pipeline

## Architecture
```
ACR Poker window
    |
    v
capture.py ---- screenshot ----+
    |                           |
    v                           v
pipeline.py (orchestrator)    card_id.py (board + hero cards)
    |                           |
    v                           v
vision_ocr.py (stacks,        regions.py (ratio-based coords)
 pot, actions, bets)
    |
    v
game_state.py -----> watch.py (live loop + smoothing)
                        |
                        v
                    action_history.py (preflop reconstruction)
                    range_lookup.py (IP/OOP ranges)
                        |
                        v
                    solver-cli (Rust, wraps postflop-solver)
                        |
                        v
                    ui.py (macOS overlay window)
```

## Phase 0: Mac OCR Pipeline -- COMPLETE
**36 tests, all passing.**

| Component | File | Status |
|-----------|------|--------|
| Window capture (Quartz API) | `src/capture.py` | Done |
| Vision OCR (stacks, pot, actions, bets) | `src/vision_ocr.py` | Done |
| Card ID (board + hero, all 13 ranks) | `src/card_id.py` | Done |
| 4-color deck support | `src/card_id.py` | Done |
| Dealer button detection | `src/card_id.py` | Done |
| Game state model + position inference | `src/game_state.py` | Done |
| Full pipeline (screenshot -> GameState) | `src/pipeline.py` | Done |
| Live watcher with temporal smoothing | `src/watch.py` | Done |

## Phase 1: Preflop Range Lookup -- COMPLETE
- ~35 GTO 6-max cash ranges encoded in JSON
- Scenarios: RFI per position, call/3bet vs each position, vs-3bet call/4bet
- Files: `solver/ranges.json`, `solver/range_lookup.py`

## Phase 2: Action Sequence Reconstruction -- COMPLETE
- Maps dealer_seat + action labels -> preflop history
- Determines IP vs OOP players for solver
- HandTracker locks dealer seat and solver inputs per hand
- File: `solver/action_history.py`

## Phase 3: Rust Solver CLI -- COMPLETE
- Wraps b-inary/postflop-solver in a Rust CLI binary
- Input: JSON (board, ranges, pot, stacks, bet sizes, street actions)
- Output: JSON (action frequencies + EV for hero's hand)
- Runs locally on Mac, called from watch.py in a background thread
- Solves turn/river (flop disabled — too slow for live play)
- File: `solver/solver-cli/src/main.rs`

## Phase 4: Integration -- COMPLETE
- watch.py calls solver-cli when new street detected + hero has action
- Display strategy (fold/call/raise %, EV) in overlay or terminal
- Solver runs in background thread, result shown when ready
- EngineRunner handles key dedup, street invalidation, timeout

## Phase 5: Multi-Table -- COMPLETE
- Parallel pipeline instances per window (watch.py TableState)
- Per-table HandTracker, ReadingSmoother, EngineRunner

## Phase 6: Overlay UI -- COMPLETE
- Native macOS AppKit overlay window (ui.py)
- Clean layout: hero info, strategy (front and center), players, debug
- Preflop advice displayed prominently with visual borders
- Background capture thread with timer-based UI updates

## OCR Reliability (live-tested)
- Hero card OCR gated on action buttons (skips small/minimized view)
- First 2 frames after action appears are skipped (transition settling)
- Forced 4-color mode for hero cards; green contamination guard
- Board pip width ratio threshold 2.0 for spade/club disambiguation
- Hero card 2: reject chars <14px wide to avoid partial-occlusion misreads
- Dealer button: color neutrality filter, brightness threshold 175
- Temporal smoothing with lock-in + duplicate hero card rejection
- HandTracker locks dealer seat, IP/OOP, ranges per hand
- Expanded action word filtering for OCR garble
- Preflop advice cross-validates call amounts against actual player bets

## TODO

### Phase 7: Hand History Parser + Player Stats
Parse ACR hand history files for accurate player stat tracking. Much more reliable than OCR for building player profiles.

**Hand history location:**
```
~/Downloads/AmericasCardroom/handHistory/vortexted/
```
- One file per table session, e.g. `HH20260307 CASHID-G34634290T254 TN-Ideal GAMETYPE-Hold'em ...`
- Plain text, appended after each hand completes (not during)
- Contains: exact player names, seats, stacks, all actions with dollar amounts, hole cards (hero only), board, pot, rake

**Format example:**
```
Hand #2686716500 - Holdem (No Limit) - $0.01/$0.02 - 2026/03/07 06:44:11 UTC
Ideal 6-max Seat #6 is the button
Seat 1: 6o6linKin6 ($3.91)
Seat 2: EZcomeEZgo18 ($1.68)
...
*** HOLE CARDS ***
Dealt to vortexted [3h 2d]
Heebert raises $0.05 to $0.05
SVSPNYK calls $0.05
vortexted folds
...
*** SUMMARY ***
```

**Plan:**
1. **File watcher** — tail hand history files, detect new hands as they're appended
2. **Parser** — extract structured data: players, actions, amounts, showdowns
3. **Stat tracker** — per-player stats stored in local SQLite DB:
   - VPIP (voluntarily put $ in pot) — loose vs tight
   - PFR (preflop raise %) — passive vs aggressive
   - 3bet % — how often they 3bet
   - Fold to Cbet % — can we bluff them?
   - WTSD (went to showdown %) — calling station detection
   - AF (aggression factor) — bet+raise / call ratio
4. **Display in overlay** — show key stats next to player names
5. **Archetype classification** — bucket players: fish, calling station, nit, LAG, TAG
   - Simple thresholds (e.g. VPIP>40 + PFR<10 = calling station)

**Hybrid approach:** OCR for live in-hand info (cards, board, pot, action buttons). Hand history for player stats (updated after each hand completes).

### Phase 8: Exploitative Adjustments
- Adjust solver input ranges based on opponent archetype
- e.g. vs calling station: widen value range, remove bluffs
- e.g. vs nit: narrow their range, bluff more
- Could use LLM for nuanced multi-street pattern interpretation (Phase 8b)

## Known Issues
- Dollar mode tables: stacks/pot read as garbage (only BB mode works)
- OCR occasionally misreads decimals (e.g. "0.5" -> "5.0")
- Board card turn/river order occasionally swaps (rare)

## File Structure
```
poker/
  ui.py                          - macOS overlay UI (AppKit)
  monitor.py                     - Simple monitor launcher
  src/
    capture.py                   - Window capture (macOS Quartz API)
    regions.py                   - Table region coordinates (ratio-based)
    vision_ocr.py                - macOS Vision OCR
    card_id.py                   - Card identification (rank + suit)
    game_state.py                - Game state model
    pipeline.py                  - Main OCR pipeline
    watch.py                     - Live watcher + solver integration
  solver/
    ranges.json                  - Preflop range lookup table
    range_lookup.py              - Range lookup by position + scenario
    action_history.py            - Action reconstruction + HandTracker
    solver-cli/                  - Rust CLI wrapping postflop-solver
  templates/
    card_ranks/                  - Card rank templates (all 13 ranks)
  tests/
  reference_screenshots/
```
