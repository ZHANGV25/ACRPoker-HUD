# ACR Poker OCR Pipeline - Progress

## Architecture
- Mac captures screenshots of ACR Poker tables
- OCR pipeline extracts game state (stacks, pot, actions, cards)
- Game state sent as JSON over ZeroMQ to PC for GTO solving

## Current Status
Pipeline reads screenshots and outputs structured game state JSON with card identification.

### Working
- macOS Vision OCR for text recognition (replaced Tesseract - much better)
- Pot total reading (BB mode)
- Stack reading for most seats
- Action label reading (F, R, C, R/B, etc.)
- Action button parsing (Fold/Call/Check/Raise + amounts)
- Hand strength text
- **Board card identification (rank + suit)** via template matching + color/shape analysis
- **Hero card identification (rank + suit)** — same approach
- Street inference from board card count
- Game state model with position inference
- ZeroMQ network layer (PUB/SUB)

### Card Identification System
- Template matching for rank (templates in `templates/card_ranks/`)
  - Current templates: 2, 3, 7, 9, J (need 4, 5, 6, 8, T, Q, K, A)
  - HSV-based text extraction handles both red and black rank text
- Color analysis for red/black suit distinction
- Shape analysis for specific suit:
  - Hearts vs Diamonds: width ratio at top vs middle
  - Clubs vs Spades: pixel group count at 35-45% height (clubs have 3 separate lobes)

### Known Issues
- Decimal points lost in small text (4.5 BB -> 45 in action buttons)
- Font "3" sometimes misread as "2" by Vision OCR (330.5 -> 2230.5) — stacks affected
- Some seat regions don't align across different window sizes
- Action labels only detected for some seats (region alignment)
- Missing card rank templates (4, 5, 6, 8, T, Q, K, A) — will be added from more screenshots

### Next Steps
1. Dealer button detection (agent working on this)
2. Live capture integration (agent working on this)
3. Region auto-calibration (detect table layout dynamically)
4. Multi-table support
5. Solver integration on PC side (TexasSolver)
6. Preflop lookup tables

## File Structure
```
poker/
  src/
    capture.py       - Window capture (macOS Quartz API)
    regions.py       - Table region coordinates (ratio-based)
    ocr.py           - Tesseract OCR (deprecated, kept as fallback)
    vision_ocr.py    - macOS Vision OCR (primary)
    digit_ocr.py     - Template-based digit matching (for problematic reads)
    card_id.py       - Card identification (rank + suit from board/hero cards)
    game_state.py    - Game state model
    pipeline.py      - Main OCR pipeline
    network.py       - ZeroMQ network layer (Mac -> PC)
    live.py          - Live window capture loop (in progress)
  templates/
    card_ranks/      - Card rank templates for template matching
  reference_screenshots/
  calibrate.py       - Visual region calibration tool
  requirements.txt
```
