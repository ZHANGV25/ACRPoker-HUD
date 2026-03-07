# ACR Poker OCR + GTO Solver Pipeline

## Architecture
```
Mac (this machine)                    Windows PC
==================                    ==========
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
game_state.py --- GameState JSON ---> ZeroMQ ---> solver_bridge.py
                                                      |
                                                      v
                                                  range_lookup.py (preflop ranges)
                                                  action_history.py (reconstruct sequence)
                                                      |
                                                      v
                                                  solver-cli (Rust, wraps postflop-solver)
                                                      |
                                                      v
                                              strategy JSON <--- ZeroMQ <--- back to Mac
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
| ZeroMQ publisher (Mac -> PC) | `src/network.py` | Done |
| Live capture loop (detect hero's turn) | `src/live.py` | Done |

## Phase 1: Preflop Range Lookup Table -- COMPLETE
- ~35 GTO 6-max cash ranges from FreeBetRange encoded in JSON
- PioSOLVER-compatible format: "AA,KK,QQ,AKs,AKo,AQs:0.5,..."
- Scenarios: RFI per position, call/3bet vs each position
- Files: `solver/ranges.json`, `solver/range_lookup.py`

## Phase 2: Action Sequence Reconstruction -- COMPLETE
- Maps dealer_seat + action labels -> ordered preflop history
- Determines IP vs OOP players for solver
- HandTracker locks dealer seat and solver inputs per hand
- Files: `solver/action_history.py`

## Phase 2.5: Live Watcher -- COMPLETE
- Continuously captures ACR window, runs OCR pipeline on state changes
- Temporal smoothing (ReadingSmoother) with lock-in + majority vote
- Settled-frame check (double capture, pixel diff < 5.0)
- Skips small tiled multi-table windows (< 750pt width)
- Displays formatted game state with solver inputs (IP/OOP/ranges/pot/stacks)
- File: `src/watch.py`

## OCR Reliability Improvements (live-tested)
- **Suit detection**: Forced 4-color mode for hero cards; green contamination guard (`green > black * 3`) to reject action label bleed-through
- **Spade vs club**: Board pip width ratio threshold raised 1.8 -> 2.0 (4-color spade pips have wider bases)
- **Hero card 2 rank**: Reject chars < 14px wide in `_identify_card_right` to avoid partial-occlusion misreads (9->J)
- **Dealer button**: Color neutrality filter eliminates colored action circles (R/C/F); brightness threshold lowered 190 -> 175
- **Temporal smoothing**: Lock-in after 2 consistent reads; duplicate hero card rejection unlocks both positions
- **HandTracker locking**: Dealer seat, IP/OOP positions, and ranges locked per hand; pot/stack update dynamically
- **Name filtering**: Expanded action word list (TOLD, SITFING, etc.) and collapsed-string matching for OCR garble
- **Seat 6 stack region**: Widened to capture full "73 BB" text

## Phase 3: Rust Solver CLI -- NOT STARTED
- Wrap b-inary/postflop-solver in a CLI binary
- Input: JSON (board, ranges, pot, stacks, bet sizes)
- Output: JSON (action frequencies + EV for hero's hand)
- File: `solver/solver-cli/` (Rust crate)

## Phase 4: Bidirectional ZeroMQ -- HALF DONE
- Mac -> PC: game state (done, PUB/SUB in network.py)
- PC -> Mac: solver strategy (not done)
- Switch to REQ/REP pattern for request-response flow

## Phase 5: PC-Side Orchestrator -- NOT STARTED
- Receives GameState, runs range lookup + action reconstruction
- Calls solver-cli, returns strategy to Mac
- File: `solver/solver_bridge.py`

## Phase 6: Mac Display -- NOT STARTED
- Show solver recommendation (action frequencies, EV)
- Terminal overlay or simple GUI

## Phase 7: Multi-Table -- NOT STARTED
- Parallel pipeline instances per window
- Per-table ZeroMQ topics or separate ports

## Known Issues
- **Dollar mode**: Stacks show `?` and pot reads are garbage when table uses $ instead of BB
- **Board card order**: Turn/river occasionally swap (needs more investigation)
- **Dealer button**: Detection works on all reference screenshots but may fail on edge cases during live play (HandTracker locking mitigates this)

## File Structure
```
poker/
  src/                       # Mac-side OCR (complete)
    capture.py               - Window capture (macOS Quartz API)
    regions.py               - Table region coordinates (ratio-based)
    vision_ocr.py            - macOS Vision OCR (primary)
    card_id.py               - Card identification (rank + suit)
    game_state.py            - Game state model + serialization
    pipeline.py              - Main OCR pipeline
    network.py               - ZeroMQ network layer (Mac -> PC)
    live.py                  - Live window capture loop
    watch.py                 - Live watcher with temporal smoothing
  solver/                    # Solver integration
    ranges.json              - Preflop range lookup table
    range_lookup.py          - Range lookup by position + scenario
    action_history.py        - Action sequence reconstruction + HandTracker
    solver_bridge.py         - Orchestrator (receive -> solve -> respond)
    solver-cli/              - Rust CLI wrapping postflop-solver (TODO)
  templates/
    card_ranks/              - Card rank templates (all 13 ranks)
  tests/
  reference_screenshots/
  debug_zones.py             - Visual region overlay tool
  calibrate.py               - Visual region calibration tool
  requirements.txt
```
