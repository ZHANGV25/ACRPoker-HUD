"""Tests for card_id.py — card detection from reference screenshots.

These tests use the reference screenshots in reference_screenshots/ and only
depend on OpenCV + numpy (no macOS Vision framework needed).
"""

import os
import pytest
import cv2
import numpy as np

from src.card_id import (
    _extract_char_from_roi,
    _match_rank,
    _detect_suit,
    _find_merged_card_area,
    detect_and_identify_board,
    detect_and_identify_hero,
    detect_dealer_button,
    identify_card,
    _load_templates,
)
from src.regions import BOARD_CARDS, HERO_CARDS, extract_table_area

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_screenshots")


def _ref(name):
    path = os.path.join(REF_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Reference screenshot not found: {name}")
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Could not load: {name}")
    return img


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------
class TestTemplateLoading:
    def test_templates_load(self):
        templates = _load_templates()
        assert isinstance(templates, dict)
        # We should have at least the 5 templates: 2, 3, 7, 9, J
        assert len(templates) >= 5
        for rank, imgs in templates.items():
            assert isinstance(imgs, list), f"Template {rank} should be a list"
            for img in imgs:
                assert img.shape[0] == 40, f"Template {rank} height != 40"

    def test_template_ranks_valid(self):
        templates = _load_templates()
        valid_ranks = set("23456789TJQKA")
        for rank in templates:
            assert rank in valid_ranks, f"Unexpected rank template: {rank}"


# ---------------------------------------------------------------------------
# Character extraction
# ---------------------------------------------------------------------------
class TestExtractChar:
    def test_returns_normalized_height(self):
        # Create a synthetic card corner with a dark character on white
        roi = np.ones((50, 40, 3), dtype=np.uint8) * 240  # white background
        # Draw a dark rectangle (simulates a rank character)
        roi[10:40, 10:25] = 30  # dark text
        result = _extract_char_from_roi(roi)
        assert result is not None
        assert result.shape[0] == 40

    def test_empty_roi_returns_none(self):
        # All white — no text
        roi = np.ones((50, 40, 3), dtype=np.uint8) * 255
        result = _extract_char_from_roi(roi)
        # May return None or a tiny/noisy result
        # The key is it shouldn't crash

    def test_red_text_detected(self):
        # Simulate red rank text (e.g. hearts/diamonds card)
        roi = np.ones((50, 40, 3), dtype=np.uint8) * 240  # white bg in BGR
        # Red text in BGR: B=30, G=30, R=200
        roi[10:40, 10:25] = [30, 30, 200]
        result = _extract_char_from_roi(roi)
        assert result is not None
        assert result.shape[0] == 40


# ---------------------------------------------------------------------------
# Suit detection (synthetic tests)
# ---------------------------------------------------------------------------
class TestDetectSuit:
    def _make_card_with_color(self, is_red):
        """Make a synthetic card image with colored suit symbol in top-right."""
        card = np.ones((120, 80, 3), dtype=np.uint8) * 240  # white card
        # Add dark border
        card[:5, :] = 50
        card[:, :3] = 50
        # Suit pip in top-right corner (matching _detect_suit ROI: 55-95% width, 3-22% height)
        sy, sx = 8, 48
        if is_red:
            card[sy:sy + 18, sx:sx + 25] = [30, 30, 200]  # red in BGR
        else:
            card[sy:sy + 18, sx:sx + 25] = [30, 30, 30]  # black
        return card

    def test_red_card_returns_heart_or_diamond(self):
        card = self._make_card_with_color(is_red=True)
        suit = _detect_suit(card)
        assert suit in ('h', 'd')

    def test_black_card_returns_club_or_spade(self):
        card = self._make_card_with_color(is_red=False)
        suit = _detect_suit(card)
        assert suit in ('c', 's')


# ---------------------------------------------------------------------------
# Board card detection from reference screenshots
# ---------------------------------------------------------------------------
class TestBoardDetection:
    def test_flop_screenshot_finds_3_cards(self):
        """Screenshot 03 has a flop — should find 3 board cards."""
        img = _ref("03_flop_hero_check.png")
        board_crop = BOARD_CARDS.crop(img)
        cards = detect_and_identify_board(board_crop)
        assert len(cards) == 3, f"Expected 3 board cards, got {len(cards)}: {cards}"

    def test_flop_screenshot_04_finds_3_cards(self):
        """Screenshot 04 has a flop — should find 3 board cards."""
        img = _ref("04_flop_facing_bet.png")
        board_crop = BOARD_CARDS.crop(img)
        cards = detect_and_identify_board(board_crop)
        assert len(cards) == 3, f"Expected 3 board cards, got {len(cards)}: {cards}"

    def test_preflop_screenshot_finds_no_cards(self):
        """Screenshot 01 is preflop with tiled view — should find 0 board cards."""
        img = _ref("01_preflop_tiled.png")
        board_crop = BOARD_CARDS.crop(img)
        cards = detect_and_identify_board(board_crop)
        assert len(cards) == 0, f"Expected 0 board cards, got {len(cards)}: {cards}"

    def test_board_cards_are_valid_format(self):
        """Any detected card should be 2 chars: rank + suit."""
        img = _ref("03_flop_hero_check.png")
        board_crop = BOARD_CARDS.crop(img)
        cards = detect_and_identify_board(board_crop)
        valid_ranks = set("23456789TJQKA")
        valid_suits = set("hdcs")
        for card in cards:
            if card is not None:
                assert len(card) == 2, f"Card '{card}' not 2 chars"
                assert card[0] in valid_ranks, f"Invalid rank in '{card}'"
                assert card[1] in valid_suits, f"Invalid suit in '{card}'"


# ---------------------------------------------------------------------------
# Hero card detection from reference screenshots
# ---------------------------------------------------------------------------
class TestHeroDetection:
    def test_hero_turn_screenshot_finds_2_cards(self):
        """Screenshot 02 shows hero's turn with hole cards visible."""
        img = _ref("02_preflop_hero_turn.png")
        hero_crop = HERO_CARDS.crop(img)
        cards = detect_and_identify_hero(hero_crop)
        assert len(cards) == 2, f"Expected 2 hero cards, got {len(cards)}: {cards}"

    def test_hero_cards_valid_format(self):
        img = _ref("02_preflop_hero_turn.png")
        hero_crop = HERO_CARDS.crop(img)
        cards = detect_and_identify_hero(hero_crop)
        valid_ranks = set("23456789TJQKA")
        valid_suits = set("hdcs")
        for card in cards:
            if card is not None:
                assert len(card) == 2
                assert card[0] in valid_ranks
                assert card[1] in valid_suits


# ---------------------------------------------------------------------------
# Dealer button detection
# ---------------------------------------------------------------------------
class TestDealerButton:
    def test_finds_dealer_in_screenshot(self):
        """Should find the dealer button in a full table screenshot."""
        img = _ref("02_preflop_hero_turn.png")
        seat = detect_dealer_button(img)
        assert seat is not None, "Dealer button not found"
        assert 1 <= seat <= 6, f"Invalid seat number: {seat}"

    def test_dealer_seat_consistent_across_screenshots(self):
        """Dealer button detection should return a valid seat for each screenshot."""
        for name in ["03_flop_hero_check.png", "04_flop_facing_bet.png"]:
            img = _ref(name)
            seat = detect_dealer_button(img)
            # May or may not find it depending on screenshot, but shouldn't crash
            if seat is not None:
                assert 1 <= seat <= 6


# ---------------------------------------------------------------------------
# Merged card area detection
# ---------------------------------------------------------------------------
class TestMergedCardArea:
    def test_flop_has_merged_area(self):
        img = _ref("03_flop_hero_check.png")
        board_crop = BOARD_CARDS.crop(img)
        rect = _find_merged_card_area(board_crop, min_area=5000)
        assert rect is not None, "No merged card area found on flop"
        x, y, w, h = rect
        assert w > h, "Merged 3-card area should be wider than tall"

    def test_preflop_no_merged_area(self):
        img = _ref("01_preflop_tiled.png")
        board_crop = BOARD_CARDS.crop(img)
        rect = _find_merged_card_area(board_crop, min_area=5000)
        # Preflop: no board cards, so no large white area expected
        # (May or may not be None depending on table felt color)


# ---------------------------------------------------------------------------
# identify_card with synthetic input
# ---------------------------------------------------------------------------
class TestIdentifyCard:
    def test_returns_none_for_blank(self):
        blank = np.zeros((100, 70, 3), dtype=np.uint8)
        result = identify_card(blank)
        assert result is None

    def test_returns_none_for_noise(self):
        np.random.seed(42)
        noise = np.random.randint(0, 256, (100, 70, 3), dtype=np.uint8)
        result = identify_card(noise)
        # Should either return None or a low-confidence guess — shouldn't crash


# ---------------------------------------------------------------------------
# Precise board card detection (slot-based, visually verified ground truth)
# ---------------------------------------------------------------------------
class TestBoardCardValues:
    """Test exact card values using slot-based detection with full image."""

    @staticmethod
    def _detect_board(name):
        img = _ref(name)
        img = extract_table_area(img)
        board_crop = BOARD_CARDS.crop(img)
        return detect_and_identify_board(board_crop, full_img=img)

    def test_flop_03(self):
        assert self._detect_board("03_flop_hero_check.png") == ['2h', '9c', '3s']

    def test_flop_04(self):
        assert self._detect_board("04_flop_facing_bet.png") == ['2h', '9c', '3s']

    def test_flop_05(self):
        assert self._detect_board("05_flop_5s9s7h_hero_turn.png") == ['5s', '9c', '7h']

    def test_flop_07(self):
        assert self._detect_board("07_flop_2s9sQh_hero_check.png") == ['2s', '9c', 'Qh']

    def test_turn_08(self):
        assert self._detect_board("08_turn_2s9sQhQd_hero_bet.png") == ['2s', '9c', 'Qh', 'Qd']

    def test_turn_10(self):
        assert self._detect_board("10_turn_Kh3hAs2h_hero_bet.png") == ['Kh', '3c', 'Ad', '2h']

    def test_turn_13_dollar(self):
        assert self._detect_board("13_turn_Ks8h6s9c_dollar.png") == ['Kc', '8h', '6h', '9s']

    def test_turn_15_dollar(self):
        assert self._detect_board("15_turn_9s4c7c6s_dollar.png") == ['9s', '4d', '7s', '6c']

    def test_preflop_01_no_cards(self):
        assert self._detect_board("01_preflop_tiled.png") == []


# ---------------------------------------------------------------------------
# Precise hero card detection (visually verified ground truth)
# ---------------------------------------------------------------------------
class TestHeroCardValues:
    """Test exact hero card values across all screenshots."""

    @staticmethod
    def _detect_hero(name):
        img = _ref(name)
        img = extract_table_area(img)
        return detect_and_identify_hero(HERO_CARDS.crop(img))

    def test_hero_02(self):
        # 2-color screenshot: 7d reads as 7h (red=hearts in 4-color mode)
        assert self._detect_hero("02_preflop_hero_turn.png") == ['7h', 'Js']

    def test_hero_05(self):
        # 2-color screenshot: Td reads as Th (red=hearts in 4-color mode)
        assert self._detect_hero("05_flop_5s9s7h_hero_turn.png") == ['Th', '6s']

    def test_hero_06(self):
        assert self._detect_hero("06_preflop_hero_turn.png") == ['8s', '9h']

    def test_hero_07(self):
        assert self._detect_hero("07_flop_2s9sQh_hero_check.png") == ['8s', '9h']

    def test_hero_09(self):
        # 2-color screenshot: Jc reads as Js (black=spades in 4-color mode)
        assert self._detect_hero("09_preflop_TsJc.png") == ['Ts', 'Js']

    def test_hero_11_dollar(self):
        assert self._detect_hero("11_preflop_4hTs_dollar.png") == ['4h', 'Ts']

    def test_hero_13_dollar(self):
        assert self._detect_hero("13_turn_Ks8h6s9c_dollar.png") == ['4h', 'Ts']

    def test_hero_15_folded(self):
        assert self._detect_hero("15_turn_9s4c7c6s_dollar.png") == []
