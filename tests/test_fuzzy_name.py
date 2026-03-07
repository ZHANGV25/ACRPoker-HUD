"""Tests for fuzzy player name matching."""

from solver.fuzzy_name import fuzzy_match, _edit_distance


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_sub(self):
        assert _edit_distance("hello", "hallo") == 1

    def test_one_insert(self):
        assert _edit_distance("helo", "hello") == 1

    def test_one_delete(self):
        assert _edit_distance("hello", "helo") == 1

    def test_empty(self):
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3

    def test_completely_different(self):
        assert _edit_distance("abc", "xyz") == 3


class TestFuzzyMatch:
    KNOWN = {"swarna89", "Pavelik", "WoWiWoN", "dahui36", "RYJITSU",
             "vortexted", "Heebert", "vla01092019", "6o6linKin6", "EZcomeEZgo18"}

    def test_exact_match(self):
        assert fuzzy_match("swarna89", self.KNOWN) == "swarna89"

    def test_case_insensitive(self):
        assert fuzzy_match("PAVELIK", self.KNOWN) == "Pavelik"
        assert fuzzy_match("pavelik", self.KNOWN) == "Pavelik"

    def test_one_char_ocr_error(self):
        # OCR reads '9' as 'g'
        assert fuzzy_match("swarna8g", self.KNOWN) == "swarna89"

    def test_two_char_ocr_error(self):
        # OCR garbles last two chars
        assert fuzzy_match("swarna8O", self.KNOWN) == "swarna89"

    def test_missing_char(self):
        # OCR drops a character
        assert fuzzy_match("Paveli", self.KNOWN) == "Pavelik"

    def test_extra_char(self):
        # OCR adds a character
        assert fuzzy_match("Pavellik", self.KNOWN) == "Pavelik"

    def test_ryjitsu_garble(self):
        assert fuzzy_match("RYJITSL", self.KNOWN) == "RYJITSU"

    def test_heebert_garble(self):
        assert fuzzy_match("Heehert", self.KNOWN) == "Heebert"

    def test_long_name_garble(self):
        assert fuzzy_match("vla0l092019", self.KNOWN) == "vla01092019"

    def test_no_match_too_different(self):
        assert fuzzy_match("xyzabc123", self.KNOWN) is None

    def test_no_match_too_short(self):
        assert fuzzy_match("ab", self.KNOWN) is None

    def test_empty_input(self):
        assert fuzzy_match("", self.KNOWN) is None

    def test_empty_known(self):
        assert fuzzy_match("swarna89", set()) is None

    def test_short_name_strict(self):
        # Short names (<=4 chars) require edit distance 1
        known = {"JxR", "ABCD", "test"}
        assert fuzzy_match("JxR", known) == "JxR"
        assert fuzzy_match("JxS", known) == "JxR"  # 1 edit
        assert fuzzy_match("JyS", known) is None   # 2 edits, too many for short name

    def test_vortexted_garble(self):
        assert fuzzy_match("vortexled", self.KNOWN) == "vortexted"

    def test_number_prefix_name(self):
        assert fuzzy_match("606linKin6", self.KNOWN) == "6o6linKin6"
