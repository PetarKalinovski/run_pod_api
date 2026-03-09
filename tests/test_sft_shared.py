from __future__ import annotations

from pathlib import Path

from runpodx.training.scripts.sft_shared import (
    analyze_ascii_response,
    build_prompt_completion_record,
    build_generation_prompt,
    extract_messages,
    format_sharegpt_example,
    resolve_wandb_run_id,
)


def test_extract_messages_supports_sharegpt_aliases():
    example = {
        "conversations": [
            {"from": "human", "value": "draw a sword"},
            {"from": "gpt", "value": " /\\\n ||\n ||\n ||"},
        ]
    }

    messages = extract_messages(example)

    assert messages == [
        {"role": "User", "content": "draw a sword"},
        {"role": "Assistant", "content": " /\\\n ||\n ||\n ||"},
    ]


def test_format_sharegpt_appends_eos_and_sanitizes_tail_repeats():
    example = {
        "messages": [
            {"role": "user", "content": "draw a sword"},
            {"role": "assistant", "content": " /\\\n ||\n ||\n ||\n ||\n ||"},
        ]
    }

    formatted = format_sharegpt_example(
        example,
        eos_token="<eos>",
        append_eos=True,
        sanitize_ascii=True,
        max_tail_line_repeats=2,
    )

    assert "### User:\ndraw a sword" in formatted
    assert formatted.endswith(" ||\n ||<eos>")


def test_analyze_ascii_response_detects_line_loops():
    metrics = analyze_ascii_response("|\n|\n|\n|\n+")

    assert metrics["line_count"] == 5
    assert metrics["max_consecutive_repeat"] == 4
    assert metrics["tail_repeat_count"] == 1
    assert metrics["repeat_line_fraction"] > 0.0


def test_build_generation_prompt_for_sharegpt():
    prompt = build_generation_prompt("Draw a castle.", "sharegpt")
    assert prompt == "### User:\nDraw a castle.\n\n### Assistant:\n"


def test_build_generation_prompt_for_prompt_completion():
    prompt = build_generation_prompt("Draw a castle.", "prompt_completion")
    assert prompt == "### Instruction:\nDraw a castle.\n\n### Response:\n"


def test_build_prompt_completion_record_extracts_art_block():
    example = {
        "messages": [
            {"role": "user", "content": "Draw an ASCII art cat."},
            {
                "role": "assistant",
                "content": (
                    "Here is an ASCII art cat:\n\n"
                    "```text\n"
                    " /\\_/\\\\\n"
                    "( o.o )\n"
                    " > ^ <\n"
                    "```\n\n"
                    "### User:\nWrite code for it"
                ),
            },
        ]
    }

    record = build_prompt_completion_record(
        example,
        source_format="sharegpt",
        eos_token="<eos>",
        append_eos=True,
        sanitize_ascii=True,
        max_tail_line_repeats=4,
    )

    assert record == {
        "prompt": "### Instruction:\nDraw an ASCII art cat.\n\n### Response:\n",
        "completion": "/\\_/\\\\\n( o.o )\n > ^ <<eos>",
    }


def test_build_prompt_completion_record_keeps_non_empty_completion():
    example = {
        "messages": [
            {"role": "user", "content": "Draw an ASCII art cat."},
            {
                "role": "assistant",
                "content": "Here is a Python program that prints a cat.",
            },
        ]
    }

    record = build_prompt_completion_record(
        example,
        source_format="sharegpt",
        eos_token="<eos>",
        append_eos=True,
    )

    assert record == {
        "prompt": "### Instruction:\nDraw an ASCII art cat.\n\n### Response:\n",
        "completion": "Here is a Python program that prints a cat.<eos>",
    }


def test_build_prompt_completion_record_strips_common_ascii_preamble():
    example = {
        "messages": [
            {"role": "user", "content": "Draw an ASCII art cat."},
            {
                "role": "assistant",
                "content": "Here is an ASCII art cat:\n /\\_/\\\\\n( o.o )\n > ^ <",
            },
        ]
    }

    record = build_prompt_completion_record(
        example,
        source_format="sharegpt",
        eos_token="<eos>",
        append_eos=True,
    )

    assert record == {
        "prompt": "### Instruction:\nDraw an ASCII art cat.\n\n### Response:\n",
        "completion": "/\\_/\\\\\n( o.o )\n > ^ <<eos>",
    }


def test_resolve_wandb_run_id_reuses_existing_id_when_resuming(tmp_path: Path):
    run_id_path = tmp_path / ".wandb_run_id"
    run_id_path.write_text("existing-run-id")

    run_id, resume_mode = resolve_wandb_run_id(
        str(tmp_path),
        resume_from_checkpoint="latest",
        generate_id=lambda: "new-run-id",
    )

    assert run_id == "existing-run-id"
    assert resume_mode == "allow"
    assert run_id_path.read_text() == "existing-run-id"


def test_resolve_wandb_run_id_overwrites_for_fresh_run(tmp_path: Path):
    run_id_path = tmp_path / ".wandb_run_id"
    run_id_path.write_text("old-run-id")

    run_id, resume_mode = resolve_wandb_run_id(
        str(tmp_path),
        resume_from_checkpoint=None,
        generate_id=lambda: "fresh-run-id",
    )

    assert run_id == "fresh-run-id"
    assert resume_mode is None
    assert run_id_path.read_text() == "fresh-run-id"
