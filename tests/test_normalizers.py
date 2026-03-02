from __future__ import annotations

import pytest

from runpodx.training.normalizers import get_normalizer


class TestAlpacaNormalizer:
    def test_normalize(self):
        normalizer = get_normalizer("sft", "alpaca")
        records = [
            {"instruction": "Summarize this", "input": "Long text...", "output": "Short summary"},
            {"instruction": "Translate", "output": "Translated text"},
        ]
        result = normalizer.normalize(records)
        assert len(result) == 2
        assert result[0]["instruction"] == "Summarize this"
        assert result[0]["input"] == "Long text..."
        assert result[1]["input"] == ""

    def test_validate(self):
        normalizer = get_normalizer("sft", "alpaca")
        assert normalizer.validate_record({"instruction": "x", "output": "y"}) is True
        assert normalizer.validate_record({"instruction": "x"}) is False


class TestShareGPTNormalizer:
    def test_normalize(self):
        normalizer = get_normalizer("sft", "sharegpt")
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "What is Python?"},
                    {"from": "gpt", "value": "A programming language."},
                ]
            }
        ]
        result = normalizer.normalize(records)
        assert result[0]["instruction"] == "What is Python?"
        assert result[0]["output"] == "A programming language."

    def test_role_format(self):
        normalizer = get_normalizer("sft", "sharegpt")
        records = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]
        result = normalizer.normalize(records)
        assert result[0]["instruction"] == "Hello"
        assert result[0]["output"] == "Hi there"

    def test_validate(self):
        normalizer = get_normalizer("sft", "sharegpt")
        assert normalizer.validate_record({"conversations": [{"a": 1}, {"b": 2}]}) is True
        assert normalizer.validate_record({"conversations": [{"a": 1}]}) is False
        assert normalizer.validate_record({}) is False


class TestCompletionNormalizer:
    def test_normalize(self):
        normalizer = get_normalizer("sft", "completion")
        records = [{"text": "Some raw text for pre-training."}]
        result = normalizer.normalize(records)
        assert result[0]["output"] == "Some raw text for pre-training."
        assert result[0]["instruction"] == ""

    def test_validate(self):
        normalizer = get_normalizer("sft", "completion")
        assert normalizer.validate_record({"text": "hello"}) is True
        assert normalizer.validate_record({"content": "hello"}) is False


class TestCustomNormalizer:
    def test_normalize_with_mapping(self):
        normalizer = get_normalizer(
            "sft",
            "custom",
            field_mapping={"instruction": "prompt", "output": "response"},
        )
        records = [{"prompt": "Do something", "response": "Done!"}]
        result = normalizer.normalize(records)
        assert result[0]["instruction"] == "Do something"
        assert result[0]["output"] == "Done!"

    def test_validate_with_mapping(self):
        normalizer = get_normalizer(
            "sft",
            "custom",
            field_mapping={"output": "answer"},
        )
        assert normalizer.validate_record({"answer": "yes"}) is True
        assert normalizer.validate_record({"output": "yes"}) is False


class TestNormalizerRegistry:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="No normalizer"):
            get_normalizer("nonexistent", "alpaca")

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="No normalizer"):
            get_normalizer("sft", "nonexistent")
