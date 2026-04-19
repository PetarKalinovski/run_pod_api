from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

DEFAULT_ASCII_BENCHMARK_PROMPT_GROUPS = {
    "core_objects": [
        "Draw an ASCII art sword.",
        "Draw an ASCII art lantern.",
        "Draw an ASCII art chair.",
        "Draw an ASCII art key.",
        "Draw an ASCII art skull.",
    ],
    "animals_creatures": [
        "Draw an ASCII art cat.",
        "Draw an ASCII art owl.",
        "Draw an ASCII art fish.",
        "Draw an ASCII art dragon.",
        "Draw an ASCII art horse.",
    ],
    "scenes_structures": [
        "Draw an ASCII art castle.",
        "Draw an ASCII art pine tree.",
        "Draw an ASCII art house.",
        "Draw an ASCII art mountain.",
        "Draw an ASCII art ship.",
    ],
    "compositions": [
        "Draw an ASCII art knight holding a sword.",
        "Draw an ASCII art robot.",
        "Draw an ASCII art treasure chest.",
        "Draw an ASCII art campfire.",
        "Draw an ASCII art bird on a branch.",
    ],
}

DEFAULT_ASCII_BENCHMARK_PROMPTS = [
    prompt
    for group_prompts in DEFAULT_ASCII_BENCHMARK_PROMPT_GROUPS.values()
    for prompt in group_prompts
]

WANDB_RUN_ID_FILENAME = ".wandb_run_id"

_ROLE_ALIASES = {
    "assistant": "Assistant",
    "gpt": "Assistant",
    "human": "User",
    "system": "System",
    "user": "User",
}

_ROLE_STOP_MARKERS = (
    "\n### User:",
    "\n### Assistant:",
    "\n### Instruction:",
    "\n### Response:",
)

_CODE_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n(.*?)```", re.DOTALL)
_LEADING_PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is)|below is|this is|the following is)?\s*"
    r"(?:an?\s+)?ascii(?:\s+art)?(?:\s+of|\s+for)?[^:\n]*:\s*$",
    re.IGNORECASE,
)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def canonicalize_role(role: Any) -> str:
    normalized = str(role or "user").strip().lower()
    return _ROLE_ALIASES.get(normalized, normalized.capitalize() or "User")


def extract_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = example.get("messages") or example.get("conversations") or []
    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = canonicalize_role(msg.get("role", msg.get("from", "user")))
        content = msg.get("content", msg.get("value", ""))
        normalized.append({"role": role, "content": str(content or "")})
    return normalized


def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction or "").strip()
    input_text = str(input_text or "").strip()
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def truncate_at_role_markers(text: str) -> str:
    normalized = _normalize_newlines(text)
    cut_idx = None
    for marker in _ROLE_STOP_MARKERS:
        idx = normalized.find(marker)
        if idx != -1 and (cut_idx is None or idx < cut_idx):
            cut_idx = idx
    if cut_idx is not None:
        normalized = normalized[:cut_idx]
    return normalized


def extract_first_code_block(text: str) -> str | None:
    match = _CODE_FENCE_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip("\n")


def is_likely_ascii_art_line(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped.strip():
        return False

    letters = sum(ch.isalpha() for ch in stripped)
    symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in stripped)
    return symbols >= 1 and letters <= max(2, len(stripped.strip()) // 4)


def extract_ascii_art_block(text: str) -> str | None:
    lines = _normalize_newlines(text).split("\n")
    best_block: list[str] | None = None
    block_start = None
    ascii_like_count = 0

    for idx, line in enumerate(lines + ["__RUNPODX_SENTINEL__"]):
        is_blank = line == ""
        is_ascii = is_likely_ascii_art_line(line)
        if line != "__RUNPODX_SENTINEL__" and (is_blank or is_ascii):
            if block_start is None:
                block_start = idx
                ascii_like_count = 0
            if is_ascii:
                ascii_like_count += 1
            continue

        if block_start is not None and ascii_like_count >= 2:
            block = lines[block_start:idx]
            if best_block is None or sum(len(item) for item in block) > sum(
                len(item) for item in best_block
            ):
                best_block = block
        block_start = None
        ascii_like_count = 0

    if best_block is None:
        return None
    return "\n".join(best_block).strip("\n")


def count_ascii_art_lines(text: str) -> int:
    return sum(1 for line in _normalize_newlines(text).split("\n") if is_likely_ascii_art_line(line))


def resolve_wandb_run_id(
    output_dir: str,
    *,
    resume_from_checkpoint: str | None,
    generate_id,
) -> tuple[str, str | None]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id_path = output_path / WANDB_RUN_ID_FILENAME

    existing_run_id = None
    if run_id_path.exists():
        existing_run_id = run_id_path.read_text().strip() or None

    if resume_from_checkpoint and existing_run_id:
        return existing_run_id, "allow"

    run_id = generate_id()
    run_id_path.write_text(run_id)
    return run_id, None


def strip_ascii_preamble(text: str) -> str:
    lines = _normalize_newlines(text).split("\n")
    first_ascii_idx = next(
        (idx for idx, line in enumerate(lines) if is_likely_ascii_art_line(line)),
        None,
    )
    if first_ascii_idx is None or first_ascii_idx == 0 or first_ascii_idx > 3:
        return text.strip()

    leading_lines = lines[:first_ascii_idx]
    trailing_lines = lines[first_ascii_idx:]
    if count_ascii_art_lines("\n".join(trailing_lines)) < 2:
        return text.strip()

    stripped_leading = [line.strip() for line in leading_lines if line.strip()]
    if not stripped_leading:
        return "\n".join(trailing_lines).strip()
    if all(_LEADING_PREAMBLE_RE.match(line) for line in stripped_leading):
        return "\n".join(trailing_lines).strip()
    return text.strip()


def trim_repeating_tail_lines(text: str, max_repeats: int | None) -> str:
    if max_repeats is None:
        return _normalize_newlines(text).rstrip("\n")

    lines = _normalize_newlines(text).split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return ""

    tail_line = lines[-1]
    repeat_start = len(lines) - 1
    while repeat_start > 0 and lines[repeat_start - 1] == tail_line:
        repeat_start -= 1

    repeat_count = len(lines) - repeat_start
    if tail_line and repeat_count > max_repeats:
        lines = lines[: repeat_start + max_repeats]
    return "\n".join(lines)


def sanitize_ascii_output(text: str, max_tail_line_repeats: int | None) -> str:
    return trim_repeating_tail_lines(text, max_tail_line_repeats)


def clean_ascii_completion_text(
    text: str,
    *,
    sanitize_ascii: bool = False,
    max_tail_line_repeats: int | None = None,
) -> str:
    cleaned = truncate_at_role_markers(text).strip()

    fenced = extract_first_code_block(cleaned)
    if fenced:
        fenced_block = extract_ascii_art_block(fenced)
        if fenced_block is not None:
            cleaned = fenced_block
        elif any(is_likely_ascii_art_line(line) for line in fenced.split("\n")):
            cleaned = fenced.strip()
    else:
        cleaned = strip_ascii_preamble(cleaned)
        ascii_block = extract_ascii_art_block(cleaned)
        if ascii_block is not None and count_ascii_art_lines(cleaned) >= 2:
            cleaned = ascii_block

    if sanitize_ascii:
        cleaned = sanitize_ascii_output(cleaned, max_tail_line_repeats)
    return cleaned.strip()


def build_prompt_completion_record(
    example: dict[str, Any],
    *,
    source_format: str,
    eos_token: str = "",
    append_eos: bool = True,
    sanitize_ascii: bool = False,
    max_tail_line_repeats: int | None = None,
) -> dict[str, str] | None:
    prompt_text = ""
    prompt_input = ""
    completion_text = ""

    if "instruction" in example and "output" in example:
        prompt_text = str(example.get("instruction", "") or "")
        prompt_input = str(example.get("input", "") or "")
        completion_text = str(example.get("output", "") or "")
    elif "prompt" in example and "completion" in example:
        prompt_text = str(example.get("prompt", "") or "")
        completion_text = str(example.get("completion", "") or "")
    elif source_format == "sharegpt" or example.get("messages") or example.get("conversations"):
        system_parts: list[str] = []
        saw_user = False
        for msg in extract_messages(example):
            role = msg["role"]
            content = msg["content"].strip()
            if role == "System" and not saw_user and content:
                system_parts.append(content)
            elif role == "User" and not prompt_text and content:
                prompt_text = content
                saw_user = True
            elif role == "Assistant" and prompt_text and content:
                completion_text = content
                break
        if system_parts:
            prompt_text = "\n\n".join([*system_parts, prompt_text]).strip()
    elif source_format == "alpaca":
        prompt_text = str(example.get("instruction", "") or "")
        prompt_input = str(example.get("input", "") or "")
        completion_text = str(example.get("output", "") or "")
    elif source_format == "completion":
        completion_text = str(example.get("text", "") or "")
    else:
        return None

    prompt_text = prompt_text.strip()
    if not completion_text.strip():
        return None

    completion = clean_ascii_completion_text(
        completion_text,
        sanitize_ascii=sanitize_ascii,
        max_tail_line_repeats=max_tail_line_repeats,
    )
    if not completion:
        return None
    if append_eos and eos_token:
        completion = f"{completion}{eos_token}"

    prompt = prompt_text
    if "prompt" not in example or "completion" not in example:
        prompt = build_instruction_prompt(prompt_text, prompt_input)
    return {"prompt": prompt, "completion": completion}


def convert_dataset_to_prompt_completion(
    dataset: Dataset,
    *,
    source_format: str,
    eos_token: str = "",
    append_eos: bool = True,
    sanitize_ascii: bool = False,
    max_tail_line_repeats: int | None = None,
) -> Dataset:
    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        record = build_prompt_completion_record(
            example,
            source_format=source_format,
            eos_token=eos_token,
            append_eos=append_eos,
            sanitize_ascii=sanitize_ascii,
            max_tail_line_repeats=max_tail_line_repeats,
        )
        if record is None:
            return {"prompt": "", "completion": "", "__keep__": False}
        return {**record, "__keep__": True}

    converted = dataset.map(_convert, remove_columns=dataset.column_names)
    converted = converted.filter(lambda row: row["__keep__"])
    return converted.remove_columns("__keep__")


def format_alpaca_example(
    example: dict[str, Any],
    *,
    eos_token: str = "",
    append_eos: bool = True,
    sanitize_ascii: bool = False,
    max_tail_line_repeats: int | None = None,
) -> str:
    instruction = str(example.get("instruction", ""))
    inp = str(example.get("input", ""))
    output = str(example.get("output", ""))

    if sanitize_ascii:
        output = sanitize_ascii_output(output, max_tail_line_repeats)
    if append_eos and eos_token:
        output = f"{output}{eos_token}"

    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def format_sharegpt_example(
    example: dict[str, Any],
    *,
    eos_token: str = "",
    append_eos: bool = True,
    sanitize_ascii: bool = False,
    max_tail_line_repeats: int | None = None,
) -> str:
    parts: list[str] = []
    for msg in extract_messages(example):
        content = msg["content"]
        if msg["role"] == "Assistant":
            if sanitize_ascii:
                content = sanitize_ascii_output(content, max_tail_line_repeats)
            if append_eos and eos_token:
                content = f"{content}{eos_token}"
        parts.append(f"### {msg['role']}:\n{content}")
    return "\n\n".join(parts)


def build_generation_prompt(prompt: str, dataset_format: str) -> str:
    if dataset_format == "prompt_completion":
        return build_instruction_prompt(prompt)
    if dataset_format == "sharegpt":
        return f"### User:\n{prompt}\n\n### Assistant:\n"
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def _safe_std(values: list[float | int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(float(v) for v in values) / len(values)
    return (sum((float(v) - mean) ** 2 for v in values) / len(values)) ** 0.5


def analyze_ascii_response(text: str) -> dict[str, float | int | str]:
    lines = _normalize_newlines(text).split("\n")
    if lines == [""]:
        lines = []

    non_empty_lines = [line for line in lines if line.strip()]
    unique_non_empty = len(set(non_empty_lines))
    max_consecutive_repeat = 0
    tail_repeat_count = 0
    current_repeat = 0
    previous = None

    for line in lines:
        if line == previous:
            current_repeat += 1
        else:
            current_repeat = 1
            previous = line
        max_consecutive_repeat = max(max_consecutive_repeat, current_repeat)

    if lines:
        tail_line = lines[-1]
        idx = len(lines) - 1
        while idx >= 0 and lines[idx] == tail_line:
            tail_repeat_count += 1
            idx -= 1

    repeat_fraction = 0.0
    if non_empty_lines:
        repeat_fraction = 1.0 - (unique_non_empty / len(non_empty_lines))

    widths = [len(line.rstrip()) for line in non_empty_lines]
    occupancies = [sum(1 for ch in line if not ch.isspace()) for line in non_empty_lines]
    non_space_chars = [ch for ch in text if not ch.isspace()]
    alnum_fraction = (
        sum(ch.isalnum() for ch in non_space_chars) / len(non_space_chars) if non_space_chars else 0.0
    )
    ascii_art_line_count = count_ascii_art_lines(text)
    contains_code_fence_or_preamble = int("```" in text or bool(_LEADING_PREAMBLE_RE.search(text.split("\n")[0].strip())))

    is_too_short = int(len(non_empty_lines) < 4)
    is_looping = int(max_consecutive_repeat >= 4 or repeat_fraction >= 0.5)
    is_punctuation_blob = int(
        ascii_art_line_count < 2
        and len(non_space_chars) > 0
        and (sum((not ch.isalnum()) for ch in non_space_chars) / len(non_space_chars)) >= 0.8
    )
    contains_non_art_text = int(any(sum(ch.isalpha() for ch in line) >= max(4, len(line.strip()) // 3) for line in non_empty_lines))

    failure_tags = [
        tag
        for tag, active in (
            ("too_short", is_too_short),
            ("looping", is_looping),
            ("punctuation_blob", is_punctuation_blob),
            ("contains_non_art_text", contains_non_art_text),
            ("has_preamble", contains_code_fence_or_preamble),
        )
        if active
    ]

    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty_lines),
        "unique_non_empty_line_count": unique_non_empty,
        "max_consecutive_repeat": max_consecutive_repeat,
        "tail_repeat_count": tail_repeat_count,
        "repeat_line_fraction": repeat_fraction,
        "char_count": sum(len(line) for line in lines),
        "width_std": _safe_std(widths),
        "occupancy_std": _safe_std(occupancies),
        "alnum_fraction": alnum_fraction,
        "ascii_art_line_count": ascii_art_line_count,
        "contains_code_fence_or_preamble": contains_code_fence_or_preamble,
        "is_too_short": is_too_short,
        "is_looping": is_looping,
        "is_punctuation_blob": is_punctuation_blob,
        "contains_non_art_text": contains_non_art_text,
        "failure_tags": ",".join(failure_tags),
    }


def summarize_generation_benchmark(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_line_count": 0.0,
            "mean_non_empty_line_count": 0.0,
            "mean_repeat_line_fraction": 0.0,
            "mean_max_consecutive_repeat": 0.0,
            "mean_char_count": 0.0,
            "mean_width_std": 0.0,
            "mean_occupancy_std": 0.0,
            "mean_alnum_fraction": 0.0,
            "eos_termination_rate": 0.0,
            "loop_rate": 0.0,
            "too_short_rate": 0.0,
            "punctuation_blob_rate": 0.0,
            "non_art_text_rate": 0.0,
        }

    count = len(rows)
    return {
        "mean_line_count": sum(float(row["line_count"]) for row in rows) / count,
        "mean_non_empty_line_count": sum(float(row["non_empty_line_count"]) for row in rows) / count,
        "mean_repeat_line_fraction": sum(float(row["repeat_line_fraction"]) for row in rows) / count,
        "mean_max_consecutive_repeat": sum(float(row["max_consecutive_repeat"]) for row in rows) / count,
        "mean_char_count": sum(float(row["char_count"]) for row in rows) / count,
        "mean_width_std": sum(float(row["width_std"]) for row in rows) / count,
        "mean_occupancy_std": sum(float(row["occupancy_std"]) for row in rows) / count,
        "mean_alnum_fraction": sum(float(row["alnum_fraction"]) for row in rows) / count,
        "eos_termination_rate": sum(float(row["terminated_on_eos"]) for row in rows) / count,
        "loop_rate": sum(float(row["is_looping"]) for row in rows) / count,
        "too_short_rate": sum(float(row["is_too_short"]) for row in rows) / count,
        "punctuation_blob_rate": sum(float(row["is_punctuation_blob"]) for row in rows) / count,
        "non_art_text_rate": sum(float(row["contains_non_art_text"]) for row in rows) / count,
    }


def write_generation_benchmark_report(output_dir: str, aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, str]:
    benchmark_dir = Path(output_dir) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    json_path = benchmark_dir / "generation_benchmark.json"
    md_path = benchmark_dir / "generation_benchmark.md"
    json_payload = {"aggregate": aggregate, "rows": rows}
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    lines = [
        "# Generation Benchmark",
        "",
        "## Aggregate",
    ]
    for key, value in aggregate.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Samples"])
    for row in rows:
        lines.append(f"### {row['prompt']}")
        lines.append(f"- terminated_on_eos: {row['terminated_on_eos']}")
        lines.append(f"- line_count: {row['line_count']}")
        lines.append(f"- non_empty_line_count: {row['non_empty_line_count']}")
        lines.append(f"- repeat_line_fraction: {row['repeat_line_fraction']}")
        lines.append(f"- max_consecutive_repeat: {row['max_consecutive_repeat']}")
        lines.append(f"- width_std: {row['width_std']}")
        lines.append(f"- occupancy_std: {row['occupancy_std']}")
        lines.append(f"- failure_tags: {row['failure_tags']}")
        lines.append("```")
        lines.append(str(row["response"]))
        lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return str(json_path), str(md_path)
