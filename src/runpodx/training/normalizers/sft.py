from __future__ import annotations

from typing import Any

from runpodx.training.normalizers import BaseNormalizer, register_normalizer


@register_normalizer("sft", "alpaca")
class AlpacaNormalizer(BaseNormalizer):
    """Alpaca format: instruction/input/output fields."""

    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "instruction": r.get("instruction", ""),
                "input": r.get("input", ""),
                "output": r.get("output", ""),
            }
            for r in records
        ]

    def validate_record(self, record: dict[str, Any]) -> bool:
        return "instruction" in record and "output" in record


@register_normalizer("sft", "sharegpt")
class ShareGPTNormalizer(BaseNormalizer):
    """ShareGPT format: conversations array with from/value pairs."""

    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for r in records:
            conversations = r.get("conversations", [])
            instruction = ""
            output = ""
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("human", "user") and not instruction:
                    instruction = value
                elif role in ("gpt", "assistant") and not output:
                    output = value
            normalized.append({"instruction": instruction, "input": "", "output": output})
        return normalized

    def validate_record(self, record: dict[str, Any]) -> bool:
        return "conversations" in record and len(record["conversations"]) >= 2


@register_normalizer("sft", "completion")
class CompletionNormalizer(BaseNormalizer):
    """Completion format: just a text field for continued pre-training."""

    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"instruction": "", "input": "", "output": r.get("text", "")}
            for r in records
        ]

    def validate_record(self, record: dict[str, Any]) -> bool:
        return "text" in record


@register_normalizer("sft", "custom")
class CustomNormalizer(BaseNormalizer):
    """Custom format with user-provided field mapping."""

    def __init__(self, field_mapping: dict[str, str] | None = None):
        self._mapping = field_mapping or {}

    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        instr_key = self._mapping.get("instruction", "instruction")
        input_key = self._mapping.get("input", "input")
        output_key = self._mapping.get("output", "output")
        return [
            {
                "instruction": r.get(instr_key, ""),
                "input": r.get(input_key, ""),
                "output": r.get(output_key, ""),
            }
            for r in records
        ]

    def validate_record(self, record: dict[str, Any]) -> bool:
        output_key = self._mapping.get("output", "output")
        return output_key in record
