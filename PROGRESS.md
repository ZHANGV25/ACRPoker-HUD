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

## Phase 7: Hand History Parser + Player Stats -- COMPLETE
Parse ACR hand history files for accurate player stat tracking.

- **File watcher** (`solver/hh_watcher.py`) — tails HH directory, detects new hands as appended
- **Parser** (`solver/hh_parser.py`) — extracts all structured data: players, actions, amounts, boards, showdowns
- **Stat tracker** (`solver/player_stats.py`) — SQLite DB with per-player stats:
  - VPIP, PFR, 3bet%, Fold-to-Cbet%, WTSD%, AF
- **Archetype classification** — fish, calling_station, nit, TAG, LAG, whale, maniac
- **Overlay integration** — stats shown next to player names (VPIP/PFR archetype)
- HH location: `~/Downloads/AmericasCardroom/handHistory/vortexted/`
- 498 hands parsed from existing files, 61 players tracked

## Phase 8: Exploitative Adjustments -- COMPLETE
- `solver/exploitative.py` adjusts solver inputs based on opponent archetype
- **Range expansion/contraction**: fish/whale assumed wider range, nit assumed tighter
- **Preflop advice tips**: e.g. "CALL AQo vs UTG open [nit: wider steal]"
- **Solver range adjustment**: villain's range auto-widened/narrowed before solving
- Full 169-combo hand strength ordering for range manipulation

## Known Issues
- Dollar mode tables: stacks/pot read as garbage (only BB mode works)
- OCR occasionally misreads decimals (e.g. "0.5" -> "5.0")
- Board card turn/river order occasionally swaps (rare)

## TODO
- **Fix card OCR reliability** — still makes costly misreads in live play:
  - 8 misread as Q (round shape confusion with Q template)
  - 6 misread as 8 or 5
  - 10 misread as 8 or J (the "0" in "10" matches 8 template)
  - Card 2 misreads (T→8, 8→T) via `_identify_card_right` path
  - Root cause: template matching IoU scoring isn't discriminative enough; Q is a "catch-all" for round shapes. Consider CNN-based or feature-based approach for more reliable rank classification at small card sizes (~467px wide tables).

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
    hh_parser.py                 - Hand history file parser
    player_stats.py              - SQLite stat tracker + archetype classification
    hh_watcher.py                - Live file watcher for hand histories
    exploitative.py              - Exploitative range adjustments
    solver-cli/                  - Rust CLI wrapping postflop-solver
  templates/
    card_ranks/                  - Card rank templates (all 13 ranks)
  tests/
  reference_screenshots/
```
