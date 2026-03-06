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
**79 tests, all passing.**

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

## Phase 1: Preflop Range Lookup Table -- NOT STARTED
- Encode ~35 GTO 6-max cash ranges from FreeBetRange into JSON
- PioSOLVER-compatible format: "AA,KK,QQ,AKs,AKo,AQs:0.5,..."
- Scenarios: RFI per position, call/3bet vs each position
- File: `solver/ranges.json`

## Phase 2: Action Sequence Reconstruction -- NOT STARTED
- Map dealer_seat + action labels -> ordered preflop/postflop history
- Determine IP vs OOP players for solver
- Infer bet sizes from current_bet_bb per player
- File: `solver/action_history.py`

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

## File Structure
```
poker/
  src/                       # Mac-side OCR (complete)
    capture.py               - Window capture (macOS Quartz API)
    regions.py               - Table region coordinates (ratio-based)
    ocr.py                   - Tesseract OCR (deprecated fallback)
    vision_ocr.py            - macOS Vision OCR (primary)
    digit_ocr.py             - Template-based digit matching (unused fallback)
    card_id.py               - Card identification (rank + suit)
    game_state.py            - Game state model + serialization
    pipeline.py              - Main OCR pipeline
    network.py               - ZeroMQ network layer (Mac -> PC)
    live.py                  - Live window capture loop
  solver/                    # PC-side solver (TODO)
    ranges.json              - Preflop range lookup table
    action_history.py        - Action sequence reconstruction
    solver_bridge.py         - Orchestrator (receive -> solve -> respond)
    solver-cli/              - Rust CLI wrapping postflop-solver
  templates/
    card_ranks/              - Card rank templates (all 13 ranks)
  tests/
  reference_screenshots/
  calibrate.py               - Visual region calibration tool
  requirements.txt
```
