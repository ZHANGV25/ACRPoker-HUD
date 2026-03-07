# ACR Poker OCR + GTO Solver Pipeline

## Architecture
```
All on Mac — no network, no second machine.

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
game_state.py -----> watch.py (live loop)
                        |
                        v
                    action_history.py (preflop reconstruction)
                    range_lookup.py (IP/OOP ranges)
                        |
                        v
                    solver-cli (Rust, wraps postflop-solver)
                        |
                        v
                    Print strategy in separate terminal window
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
- Scenarios: RFI per position, call/3bet vs each position
- Files: `solver/ranges.json`, `solver/range_lookup.py`

## Phase 2: Action Sequence Reconstruction -- COMPLETE
- Maps dealer_seat + action labels -> preflop history
- Determines IP vs OOP players for solver
- HandTracker locks dealer seat and solver inputs per hand
- File: `solver/action_history.py`

## Phase 3: Rust Solver CLI -- NOT STARTED
- Wrap b-inary/postflop-solver in a Rust CLI binary
- Input: JSON (board, ranges, pot, stacks, bet sizes)
- Output: JSON (action frequencies + EV for hero's hand)
- Runs locally on Mac, called from watch.py
- File: `solver/solver-cli/` (Rust crate)

## Phase 4: Integration -- NOT STARTED
- watch.py calls solver-cli when new street detected
- Display strategy (fold/call/raise %, EV) in terminal output
- Solver runs in background, result shown when ready

## Phase 5: Multi-Table -- NOT STARTED
- Parallel pipeline instances per window

## OCR Reliability (live-tested)
- Forced 4-color mode for hero cards; green contamination guard
- Board pip width ratio threshold 2.0 for spade/club disambiguation
- Hero card 2: reject chars <14px wide to avoid partial-occlusion misreads
- Dealer button: color neutrality filter, brightness threshold 175
- Temporal smoothing with lock-in + duplicate hero card rejection
- HandTracker locks dealer seat, IP/OOP, ranges per hand
- Expanded action word filtering for OCR garble

## Known Issues
- Dollar mode tables: stacks/pot read as garbage (only BB mode works)
- Board card turn/river order occasionally swaps (rare)

## File Structure
```
poker/
  src/
    capture.py               - Window capture (macOS Quartz API)
    regions.py               - Table region coordinates (ratio-based)
    vision_ocr.py            - macOS Vision OCR
    card_id.py               - Card identification (rank + suit)
    game_state.py            - Game state model
    pipeline.py              - Main OCR pipeline
    watch.py                 - Live watcher + solver integration
  solver/
    ranges.json              - Preflop range lookup table
    range_lookup.py          - Range lookup by position + scenario
    action_history.py        - Action reconstruction + HandTracker
    solver-cli/              - Rust CLI wrapping postflop-solver (TODO)
  templates/
    card_ranks/              - Card rank templates (all 13 ranks)
  tests/
  reference_screenshots/
```
