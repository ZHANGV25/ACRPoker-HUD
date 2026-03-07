"""Main OCR pipeline: screenshot -> game state JSON."""

import cv2
import numpy as np

from src.regions import (
    SEATS, HERO_SEAT, POT_TOTAL, POT_COMMITTED,
    BOARD_CARDS,
    FOLD_BUTTON, CALL_CHECK_BUTTON, RAISE_BET_BUTTON,
    HAND_STRENGTH, HERO_CARDS, TITLE_BAR, HAND_ID,
    extract_table_area,
)
from src.vision_ocr import (
    read_bb_amount, read_pot, read_action_label,
    read_action_buttons, ocr_crop, ocr_crop_all,
)
from src.card_id import detect_and_identify_board, detect_and_identify_hero, detect_dealer_button
from src.game_state import GameState, PlayerState

# Words that UI overlays on top of or near player names (results, actions)
_ACTION_WORDS = frozenset({
    "FOLD", "CALL", "CHECK", "RAISE", "BET", "ALLIN", "ALL-IN",
    "MUCK", "SHOW", "DON'T SHOW", "DONT SHOW", "DON'T", "DONT",
    "POST SB", "POST BB", "POST", "POST B8",
    "CHFCK", "CHEC", "RAISF", "RAIS",
    "SITTING OUT", "SITTING", "SIT OUT", "DO GAILUC",
})


def _clean_name(raw):
    # type: (str) -> str
    """Filter out action/overlay words that OCR picks up in the name region."""
    if not raw:
        return ""
    upper = raw.strip().upper()
    # Exact match
    if upper in _ACTION_WORDS:
        return ""
    # Starts with action word (e.g. "FOLD " or "CALL 2.5")
    for word in _ACTION_WORDS:
        if upper.startswith(word) and (len(upper) == len(word) or not upper[len(word)].isalpha()):
            return ""
    # Fuzzy: collapse spaces and recheck (OCR reads "FOLD" as "FOI D", "FOL D", etc.)
    collapsed = upper.replace(" ", "")
    for word in _ACTION_WORDS:
        if collapsed == word.replace(" ", ""):
            return ""
    # Common OCR misreads of action words
    if collapsed in ("FOID", "FOLD", "FOIS", "FOLO", "TOLD", "CALI", "CHFCK", "RAISF"):
        return ""
    # Catch "SIT TING OUT", "SITTING O", etc.
    if "SITTING" in collapsed or "SITOUT" in collapsed or "SITFING" in collapsed:
        return ""
    return raw.strip()



def _infer_dealer_from_bets(state):
    # type: (GameState) -> int | None
    """Infer dealer seat from blind bets on preflop.

    The 0.5 BB bet uniquely identifies the SB. Dealer is the seat before SB.
    Only works preflop before blinds are consumed.
    """
    if state.street != "preflop":
        return None

    active = [p for p in state.players if not p.is_sitting_out]
    active_seats = sorted([p.seat for p in active])
    if len(active_seats) < 2:
        return None

    # Find the unique 0.5 BB bettor (SB)
    sb_candidates = [p for p in active
                     if p.current_bet_bb is not None
                     and 0.4 <= p.current_bet_bb <= 0.6]
    if len(sb_candidates) != 1:
        return None

    sb_seat = sb_candidates[0].seat
    try:
        sb_idx = active_seats.index(sb_seat)
    except ValueError:
        return None

    dealer_idx = (sb_idx - 1) % len(active_seats)
    return active_seats[dealer_idx]


def process_screenshot(img: np.ndarray) -> GameState:
    """Process a single screenshot into a GameState.

    This is the main entry point for the OCR pipeline.
    """
    state = GameState()

    # Auto-crop to game window if screenshot has black padding
    img = extract_table_area(img)

    # Title bar info
    title_text = ocr_crop(TITLE_BAR.crop(img))
    state.stakes = title_text

    # Hand ID
    hand_id_text = ocr_crop(HAND_ID.crop(img))
    if ":" in hand_id_text:
        state.hand_id = hand_id_text.split(":")[-1].strip()

    # Pot
    state.total_bb = read_pot(POT_TOTAL.crop(img))
    state.pot_bb = read_pot(POT_COMMITTED.crop(img))

    # Board cards
    board_crop = BOARD_CARDS.crop(img)
    state.board = detect_and_identify_board(board_crop, full_img=img)
    state.infer_street()

    # Hero cards
    hero_crop = HERO_CARDS.crop(img)
    state.hero_cards = detect_and_identify_hero(hero_crop)

    # Dealer button (visual detection first, validated by bets after players are read)
    state.dealer_seat = detect_dealer_button(img)

    # Players
    for seat_num, regions in SEATS.items():
        player = PlayerState(seat=seat_num)

        # Name
        name = _clean_name(ocr_crop(regions["name"].crop(img)))
        player.name = name if name else None

        # Stack
        stack = read_bb_amount(regions["stack"].crop(img))
        player.stack_bb = stack

        # Action label
        action = read_action_label(regions["action"].crop(img))
        player.action_label = action

        # Current bet
        bet = read_bb_amount(regions["bet"].crop(img))
        player.current_bet_bb = bet

        # Check if folded
        if action and "F" in action.split("/")[-1]:
            player.is_folded = True

        # Check if sitting out
        if name and "SITTING" in name.upper():
            player.is_sitting_out = True

        # Hero
        if seat_num == HERO_SEAT:
            player.is_hero = True

        state.players.append(player)

    # Validate/fix dealer seat using blind bets (more reliable than visual on preflop)
    bet_dealer = _infer_dealer_from_bets(state)
    if bet_dealer is not None:
        state.dealer_seat = bet_dealer

    # Action buttons
    state.available_actions = read_action_buttons(
        FOLD_BUTTON.crop(img),
        CALL_CHECK_BUTTON.crop(img),
        RAISE_BET_BUTTON.crop(img),
    )

    # Hand strength
    hs = ocr_crop(HAND_STRENGTH.crop(img))
    if hs:
        state.available_actions["hand_strength"] = hs

    return state


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <screenshot_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Error: could not load %s" % sys.argv[1])
        sys.exit(1)

    state = process_screenshot(img)
    print(state.to_json())
