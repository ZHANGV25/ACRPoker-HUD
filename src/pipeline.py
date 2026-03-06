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

    # Dealer button
    state.dealer_seat = detect_dealer_button(img)

    # Players
    for seat_num, regions in SEATS.items():
        player = PlayerState(seat=seat_num)

        # Name
        name = ocr_crop(regions["name"].crop(img))
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
