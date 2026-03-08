from semcache.utils.normalize import normalize_prompt
from semcache.utils.hashing import hash_prompt


class TestNormalizePrompt:
    def test_removes_punctuation(self):
        assert normalize_prompt("Explain CNN!!!") == "explain cnn"

    def test_strips_whitespace(self):
        assert normalize_prompt("  What is CNN? ") == "what is cnn"

    def test_lowercases(self):
        assert normalize_prompt("HELLO WORLD") == "hello world"

    def test_collapses_spaces(self):
        assert normalize_prompt("what   is   a   CNN") == "what is a cnn"

    def test_mixed(self):
        assert normalize_prompt("Explain CNN please!!!") == "explain cnn please"

    def test_already_normalized(self):
        assert normalize_prompt("explain cnn") == "explain cnn"

    def test_empty_string(self):
        assert normalize_prompt("") == ""

    def test_only_punctuation(self):
        assert normalize_prompt("!!!???...") == ""


class TestHashPrompt:
    def test_deterministic(self):
        assert hash_prompt("explain cnn") == hash_prompt("explain cnn")

    def test_returns_hex_string(self):
        result = hash_prompt("explain cnn")
        assert isinstance(result, str)
        assert len(result) == 64
        int(result, 16)  # raises ValueError if not valid hex

    def test_different_inputs_produce_different_hashes(self):
        assert hash_prompt("explain cnn") != hash_prompt("what is cnn")

    def test_empty_string(self):
        result = hash_prompt("")
        assert isinstance(result, str)
        assert len(result) == 64
