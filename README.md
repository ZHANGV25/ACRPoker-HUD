# ACR Poker HUD

Real-time poker HUD for ACR Poker on macOS. Captures the poker table via screen capture, runs OCR to extract game state, looks up preflop GTO ranges, and solves postflop spots using a local Rust solver. Everything runs locally — no network, no second machine.

## What It Does

- **Live OCR**: Reads board cards, hero cards, player stacks, bets, pot size, dealer button, and action labels from the ACR Poker window
- **Preflop Ranges**: Looks up GTO open/call/3-bet/4-bet ranges by position (6-max cash)
- **Postflop Solver**: Runs a local Rust solver (wrapping [postflop-solver](https://github.com/b-inary/postflop-solver)) for turn and river spots
- **Overlay UI**: Compact always-on-top window showing strategy advice, powered by native macOS AppKit
- **Multi-table**: Tracks multiple expanded table windows independently

## Requirements

- **macOS** (uses Quartz screen capture + Vision OCR framework)
- **Python 3.9** (ships with Xcode Command Line Tools)
- **Rust** (for building the solver CLI)
- **ACR Poker** in BB display mode (dollar mode not supported)

## Setup

### 1. Python dependencies

```bash
pip3 install pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa opencv-python numpy
```

### 2. Screen recording permission

Go to **System Settings > Privacy & Security > Screen Recording** and enable your terminal app (Terminal.app, iTerm2, etc).

### 3. Build the Rust solver (optional, for postflop)

```bash
cd solver/solver-cli
cargo build --release
```

This produces `solver/solver-cli/target/release/tbl-engine`.

### 4. ACR Poker settings

- Use **BB display mode** (not dollar amounts)
- Enable **4-color deck** for best card recognition
- Keep the table window expanded (not minimized/tiled)

## Running

### Overlay UI (recommended)

```bash
python3 ui.py
```

Opens a compact overlay window that polls the poker table and displays:
- Your cards and position
- Preflop advice (RAISE / CALL / FOLD with range info)
- Postflop solver output (EV, equity, action frequencies)
- Player table with stacks, bets, and actions
- Debug info at the bottom

### Terminal watcher

```bash
python3 -m src.watch           # live capture from all table windows
python3 -m src.watch --all     # show every poll, not just changes
python3 -m src.watch -f img.png  # process a single screenshot
```

### Single screenshot pipeline

```bash
python3 -m src.pipeline screenshot.png
```

Outputs the full game state as JSON.

## Architecture

```
ACR Poker window
    |
    v
capture.py ---- screenshot (Quartz) ----+
    |                                     |
    v                                     v
pipeline.py (orchestrator)          card_id.py (board + hero cards)
    |                                     |
    v                                     v
vision_ocr.py (stacks,             regions.py (ratio-based coords)
 pot, actions, bets)
    |
    v
game_state.py -----> watch.py (live loop + smoothing)
                        |
                        v
                    action_history.py (preflop reconstruction)
                    range_lookup.py (preflop GTO ranges)
                        |
                        v
                    solver-cli (Rust, wraps postflop-solver)
                        |
                        v
                    ui.py (macOS overlay) or terminal output
```

## Project Structure

```
poker/
  ui.py                        - macOS overlay UI (AppKit)
  src/
    capture.py                 - Window capture (macOS Quartz API)
    regions.py                 - Table region coordinates (ratio-based)
    vision_ocr.py              - macOS Vision OCR for text
    card_id.py                 - Card identification (rank templates + suit analysis)
    game_state.py              - Game state model + position inference
    pipeline.py                - Main OCR pipeline (screenshot -> GameState)
    watch.py                   - Live watcher with temporal smoothing + solver integration
  solver/
    ranges.json                - Preflop GTO range tables (6-max cash)
    range_lookup.py            - Range lookup by position + scenario
    action_history.py          - Preflop action reconstruction + HandTracker
    solver-cli/                - Rust CLI wrapping postflop-solver
      src/main.rs              - Solver binary (JSON in -> JSON out)
  templates/
    card_ranks/                - Card rank templates for template matching (13 ranks)
  tests/                       - Test suite (card ID, pipeline, ranges, etc.)
  reference_screenshots/       - Test screenshots
```

## How It Works

### Card Recognition
- **Rank**: Template matching against reference images for all 13 ranks (2-A)
- **Suit**: HSV color analysis + shape features (pip width ratios for club/spade, top/mid ratios for heart/diamond)
- **Board cards**: Per-slot calibrated regions with top-right pip detection
- **Hero cards**: Separate detection with fallback strategies for occluded cards

### Temporal Smoothing
- Cards are stabilized using a lock-in + majority vote system
- Hero cards only read when action buttons are visible (view is expanded)
- First 2 frames after action appears are skipped to let the view settle
- Board card locks reset on street changes

### Preflop Strategy
- ~35 GTO ranges for 6-max cash: RFI, vs-RFI (call/3bet), vs-3bet (call/4bet)
- Position-aware: adjusts advice based on opener position and hero position
- Cross-validates OCR reads to avoid phantom raise detection

### Postflop Solver
- Rust binary wrapping the postflop-solver library
- Solves turn and river spots (flop skipped — too slow for live play)
- Runs in a background thread, results displayed when ready
- Inputs: board, ranges (from preflop reconstruction), pot, effective stacks

## Known Limitations

- Only works with ACR Poker on macOS
- Requires BB display mode (dollar amounts not supported)
- Flop solving disabled (too slow for live play)
- OCR occasionally misreads small text (decimals like "0.5" -> "5.0")
- 6-max only (no support for other table sizes yet)

## Tests

```bash
python3 -m pytest tests/ -v
```
