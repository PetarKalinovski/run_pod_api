from __future__ import annotations

from collections import Counter
import difflib
import json
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

try:
    from sft_shared import (
        DEFAULT_ASCII_BENCHMARK_PROMPTS,
        analyze_ascii_response,
        build_generation_prompt,
        clean_ascii_completion_text,
        convert_dataset_to_prompt_completion,
        count_ascii_art_lines,
        summarize_generation_benchmark,
    )
except ImportError:
    from runpodx.training.scripts.sft_shared import (
        DEFAULT_ASCII_BENCHMARK_PROMPTS,
        analyze_ascii_response,
        build_generation_prompt,
        clean_ascii_completion_text,
        convert_dataset_to_prompt_completion,
        count_ascii_art_lines,
        summarize_generation_benchmark,
    )


PROSE_MARKERS = ("###", "```", "here is", "response:", "assistant:", "user:", "ascii art")
PROFILE_VECTOR_BINS = 16
OCCUPANCY_GRID_SIZE = 24


def prepare_datasets(
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    *,
    dataset_format: str,
    tokenizer,
    config: dict,
) -> tuple[Dataset, Dataset | None]:
    converter_kwargs = {
        "source_format": dataset_format,
        "eos_token": tokenizer.eos_token or "",
        "append_eos": False,
        "sanitize_ascii": config.get("sanitize_ascii_outputs", False),
        "max_tail_line_repeats": config.get("ascii_max_tail_line_repeats"),
    }

    original_train_size = len(train_dataset)
    train_dataset = convert_dataset_to_prompt_completion(train_dataset, **converter_kwargs)
    train_dataset = train_dataset.rename_column("completion", "target")
    print(
        "[runpodx] Converted train dataset to GRPO prompt/target pairs: "
        f"{original_train_size} -> {len(train_dataset)}"
    )

    if eval_dataset is not None:
        original_eval_size = len(eval_dataset)
        eval_dataset = convert_dataset_to_prompt_completion(eval_dataset, **converter_kwargs)
        eval_dataset = eval_dataset.rename_column("completion", "target")
        print(
            "[runpodx] Converted validation dataset to GRPO prompt/target pairs: "
            f"{original_eval_size} -> {len(eval_dataset)}"
        )

    return train_dataset, eval_dataset


def normalize_reward_text(text: str) -> str:
    return clean_ascii_completion_text(
        str(text or ""),
        sanitize_ascii=False,
        max_tail_line_repeats=None,
    ).strip()


def non_empty_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.split("\n") if line.strip()]


def visible_char_count(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def trimmed_canvas_lines(text: str) -> list[str]:
    raw_lines = [line.rstrip("\n") for line in text.split("\n")]
    non_empty_indices = [idx for idx, line in enumerate(raw_lines) if line.strip()]
    if not non_empty_indices:
        return []

    top = non_empty_indices[0]
    bottom = non_empty_indices[-1]
    lines = [line.rstrip() for line in raw_lines[top : bottom + 1]]

    left_offsets = [
        next((idx for idx, ch in enumerate(line) if not ch.isspace()), 0)
        for line in lines
        if line.strip()
    ]
    min_left = min(left_offsets, default=0)
    return [line[min_left:] if len(line) > min_left else "" for line in lines]


def resample_series(values: list[float], bins: int = PROFILE_VECTOR_BINS) -> list[float]:
    if bins <= 0:
        return []
    if not values:
        return [0.0] * bins
    if len(values) == 1:
        return [float(values[0])] * bins

    result: list[float] = []
    source_last = len(values) - 1
    target_last = bins - 1
    for idx in range(bins):
        position = 0.0 if target_last == 0 else idx * source_last / target_last
        lower = int(position)
        upper = min(source_last, lower + 1)
        fraction = position - lower
        interpolated = values[lower] * (1.0 - fraction) + values[upper] * fraction
        result.append(float(interpolated))
    return result


def line_profile(text: str) -> tuple[list[float], list[float]]:
    lines = trimmed_canvas_lines(text)
    if not lines:
        return [0.0] * PROFILE_VECTOR_BINS, [0.0] * PROFILE_VECTOR_BINS

    widths: list[float] = []
    left_offsets: list[float] = []
    canvas_width = max((len(line) for line in lines), default=1)

    for line in lines:
        if not line.strip():
            widths.append(0.0)
            left_offsets.append(0.0)
            continue
        first = next((idx for idx, ch in enumerate(line) if not ch.isspace()), 0)
        last = max(idx for idx, ch in enumerate(line) if not ch.isspace())
        widths.append(float(last - first + 1))
        left_offsets.append(float(first))

    widths = [value / max(1.0, float(canvas_width)) for value in widths]
    left_offsets = [value / max(1.0, float(canvas_width)) for value in left_offsets]
    return resample_series(widths), resample_series(left_offsets)


def mean_absolute_distance(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    size = min(len(lhs), len(rhs))
    return sum(abs(lhs[idx] - rhs[idx]) for idx in range(size)) / size


def normalized_occupancy_points(text: str, size: int = OCCUPANCY_GRID_SIZE) -> set[tuple[int, int]]:
    lines = trimmed_canvas_lines(text)
    if not lines:
        return set()

    height = len(lines)
    width = max((len(line) for line in lines), default=1)
    points: set[tuple[int, int]] = set()
    for row_idx, line in enumerate(lines):
        for col_idx, ch in enumerate(line):
            if ch.isspace():
                continue
            norm_row = 0 if height == 1 else round(row_idx * (size - 1) / (height - 1))
            norm_col = 0 if width == 1 else round(col_idx * (size - 1) / (width - 1))
            points.add((norm_row, norm_col))
    return points


def occupancy_iou(lhs: str, rhs: str) -> float:
    lhs_points = normalized_occupancy_points(lhs)
    rhs_points = normalized_occupancy_points(rhs)
    if not lhs_points and not rhs_points:
        return 1.0
    if not lhs_points or not rhs_points:
        return 0.0
    return len(lhs_points & rhs_points) / len(lhs_points | rhs_points)


def char_ngram_jaccard(lhs: str, rhs: str, n: int = 3) -> float:
    lhs_text = "".join(ch for ch in lhs if not ch.isspace())
    rhs_text = "".join(ch for ch in rhs if not ch.isspace())
    if len(lhs_text) < n or len(rhs_text) < n:
        return difflib.SequenceMatcher(None, lhs_text, rhs_text).ratio()

    lhs_ngrams = {lhs_text[idx : idx + n] for idx in range(len(lhs_text) - n + 1)}
    rhs_ngrams = {rhs_text[idx : idx + n] for idx in range(len(rhs_text) - n + 1)}
    if not lhs_ngrams and not rhs_ngrams:
        return 1.0
    if not lhs_ngrams or not rhs_ngrams:
        return 0.0
    return len(lhs_ngrams & rhs_ngrams) / len(lhs_ngrams | rhs_ngrams)


def max_identical_char_run(text: str) -> int:
    best = 0
    current = 0
    previous = None
    for ch in text:
        if ch.isspace():
            continue
        if ch == previous:
            current += 1
        else:
            previous = ch
            current = 1
        best = max(best, current)
    return best


def max_ngram_frequency_ratio(text: str, n: int = 3) -> float:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < n:
        return 0.0
    ngrams = [compact[idx : idx + n] for idx in range(len(compact) - n + 1)]
    counts = Counter(ngrams)
    return max(counts.values()) / max(1, len(ngrams))


def completion_token_length(completion_ids) -> int:
    if completion_ids is None:
        return 0
    if hasattr(completion_ids, "shape") and completion_ids.shape:
        return int(completion_ids.shape[-1])
    return len(completion_ids)


def soft_overlong_penalty(
    completion_ids,
    *,
    max_completion_length: int,
    soft_overlong_cache: int,
) -> float:
    length = completion_token_length(completion_ids)
    if length <= 0 or soft_overlong_cache <= 0 or max_completion_length <= 0:
        return 0.0
    if length <= max_completion_length - soft_overlong_cache:
        return 0.0

    cap_start = max_completion_length - soft_overlong_cache
    penalty = (length - cap_start) / max(1, soft_overlong_cache)
    return -float(max(0.0, min(1.0, penalty)))


def ascii_style_reward(completions, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        raw = str(completion or "")
        cleaned = normalize_reward_text(raw)
        metrics = analyze_ascii_response(cleaned)
        non_empty = max(1, int(metrics["non_empty_line_count"]))
        ascii_lines = count_ascii_art_lines(cleaned)
        ascii_ratio = ascii_lines / non_empty

        visible_chars = sum(1 for ch in cleaned if not ch.isspace()) or 1
        alpha_ratio = sum(1 for ch in cleaned if ch.isalpha()) / visible_chars
        marker_penalty = sum(0.35 for marker in PROSE_MARKERS if marker in raw.lower())
        unique_symbols = len({ch for ch in cleaned if not ch.isalnum() and not ch.isspace()})
        symbol_bonus = min(unique_symbols / 6.0, 1.0) * 0.12

        reward = 0.75 * ascii_ratio + 0.25 * min(ascii_lines / 8.0, 1.0)
        reward += symbol_bonus
        reward -= marker_penalty
        reward -= max(0.0, alpha_ratio - 0.18) * 2.5
        rewards.append(float(reward))
    return rewards


def anti_loop_reward(completions, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        cleaned = normalize_reward_text(str(completion or ""))
        metrics = analyze_ascii_response(cleaned)
        repeat_fraction = float(metrics["repeat_line_fraction"])
        max_repeat = max(0, int(metrics["max_consecutive_repeat"]) - 1)
        trigram_ratio = max_ngram_frequency_ratio(cleaned, n=3)
        char_run = max_identical_char_run(cleaned)
        loop_penalty = repeat_fraction * 1.75 + max_repeat / 16.0
        loop_penalty += max(0.0, trigram_ratio - 0.08) * 2.5
        loop_penalty += max(0.0, char_run - 12) / 18.0
        rewards.append(float(max(-1.0, 1.0 - min(2.5, loop_penalty))))
    return rewards


def target_similarity_reward(completions, target, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion, expected in zip(completions, target, strict=True):
        cleaned_completion = normalize_reward_text(str(completion or ""))
        cleaned_target = normalize_reward_text(str(expected or ""))
        char_ratio = difflib.SequenceMatcher(None, cleaned_completion, cleaned_target).ratio()
        ngram_overlap = char_ngram_jaccard(cleaned_completion, cleaned_target, n=3)
        generated_lines = set(non_empty_lines(cleaned_completion))
        target_lines = set(non_empty_lines(cleaned_target))
        if target_lines:
            line_overlap = len(generated_lines & target_lines) / len(target_lines)
        else:
            line_overlap = 0.0
        occupancy_score = occupancy_iou(cleaned_completion, cleaned_target)
        generated_widths, generated_offsets = line_profile(cleaned_completion)
        target_widths, target_offsets = line_profile(cleaned_target)
        width_similarity = 1.0 - mean_absolute_distance(generated_widths, target_widths)
        offset_similarity = 1.0 - mean_absolute_distance(generated_offsets, target_offsets)
        reward = (
            0.2 * char_ratio
            + 0.2 * ngram_overlap
            + 0.15 * line_overlap
            + 0.3 * occupancy_score
            + 0.1 * width_similarity
            + 0.05 * offset_similarity
        )
        rewards.append(float(reward))
    return rewards


def target_length_reward(
    completions,
    target,
    *,
    completion_ids=None,
    max_completion_length: int = 192,
    soft_overlong_cache: int = 32,
    **kwargs,
) -> list[float]:
    rewards: list[float] = []
    completion_id_rows = completion_ids or [None] * len(completions)
    for completion, expected, completion_id_row in zip(
        completions,
        target,
        completion_id_rows,
        strict=True,
    ):
        cleaned_completion = normalize_reward_text(str(completion or ""))
        cleaned_target = normalize_reward_text(str(expected or ""))
        generated_lines = non_empty_lines(cleaned_completion)
        target_lines = non_empty_lines(cleaned_target)
        target_count = max(4, len(target_lines))
        line_diff = abs(len(generated_lines) - target_count) / target_count

        generated_visible = visible_char_count(cleaned_completion)
        target_visible = max(12, visible_char_count(cleaned_target))
        char_diff = abs(generated_visible - target_visible) / target_visible

        reward = 1.0 - 0.55 * line_diff - 0.45 * min(char_diff, 1.5)
        if len(generated_lines) < max(3, int(target_count * 0.4)):
            reward -= 0.7
        if generated_visible < max(12, int(target_visible * 0.3)):
            reward -= 0.7
        reward += soft_overlong_penalty(
            completion_id_row,
            max_completion_length=max_completion_length,
            soft_overlong_cache=soft_overlong_cache,
        )
        rewards.append(float(max(-1.0, reward)))
    return rewards


def build_reward_functions(config: dict):
    target_length = partial(
        target_length_reward,
        max_completion_length=int(config.get("grpo_max_completion_length", 192)),
        soft_overlong_cache=int(config.get("grpo_soft_overlong_cache", 32)),
    )
    target_length.__name__ = "target_length_reward"
    return [
        ascii_style_reward,
        anti_loop_reward,
        target_similarity_reward,
        target_length,
    ]


def run_generation_benchmark(model, tokenizer, config: dict, report_to: str):
    prompts = config.get("generation_eval_prompts") or DEFAULT_ASCII_BENCHMARK_PROMPTS
    if not config.get("generation_eval_enabled") and not config.get("generation_eval_prompts"):
        return None

    print("[runpodx] Running generation benchmark...")
    benchmark_rows = []
    prompt_device = next(model.parameters()).device
    max_new_tokens = config.get("generation_eval_max_new_tokens", 256)

    for prompt in prompts:
        rendered_prompt = build_generation_prompt(prompt, "prompt_completion")
        encoded = tokenizer(rendered_prompt, return_tensors="pt").to(prompt_device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = generated[0][encoded["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        metrics = analyze_ascii_response(response)
        benchmark_rows.append(
            {
                "prompt": prompt,
                "response": response,
                **metrics,
                "terminated_on_eos": int(tokenizer.eos_token_id in generated_ids.tolist()),
            }
        )

    aggregate = summarize_generation_benchmark(benchmark_rows)
    print(f"[runpodx] Generation benchmark aggregate: {json.dumps(aggregate, indent=2)}")

    if report_to == "wandb":
        try:
            import wandb

            wandb.log({f"benchmark/{key}": value for key, value in aggregate.items()})
            table = wandb.Table(
                columns=[
                    "prompt",
                    "response",
                    "line_count",
                    "non_empty_line_count",
                    "unique_non_empty_line_count",
                    "max_consecutive_repeat",
                    "tail_repeat_count",
                    "repeat_line_fraction",
                    "terminated_on_eos",
                ]
            )
            for row in benchmark_rows:
                table.add_data(
                    row["prompt"],
                    row["response"],
                    row["line_count"],
                    row["non_empty_line_count"],
                    row["unique_non_empty_line_count"],
                    row["max_consecutive_repeat"],
                    row["tail_repeat_count"],
                    row["repeat_line_fraction"],
                    row["terminated_on_eos"],
                )
            wandb.log({"benchmark/samples": table})
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[runpodx] WARN: failed to log benchmark to W&B: {exc}")

    return aggregate
