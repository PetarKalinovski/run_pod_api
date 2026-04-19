from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runpodx.training.synthetic_ascii.storage import append_jsonl, write_summary


@dataclass(frozen=True)
class MixedDatasetConfig:
    synthetic_paths: tuple[str, ...]
    real_paths: tuple[str, ...]
    output_dir: str
    target_total: int | None = None
    validation_fraction: float = 0.05
    seed: int = 42
    max_per_source: int | None = None
    max_per_prompt_family: int | None = None
    real_fraction_min: float = 0.2
    real_fraction_max: float = 0.3


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _prompt_family(row: dict[str, Any]) -> str:
    family = str(row.get("prompt_family") or row.get("family_id") or "").strip().lower()
    if family:
        return family
    messages = row.get("messages") or []
    if len(messages) >= 2:
        prompt = str(messages[1].get("content") or "")
    else:
        prompt = ""
    prompt = prompt.lower()
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt[:120]


def _assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if len(messages) >= 3:
        return str(messages[2].get("content") or "")
    return ""


def _source_name(row: dict[str, Any]) -> str:
    for key in ("source_dataset", "source", "category"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or []
    prompt = str(messages[1].get("content") or "") if len(messages) >= 2 else ""
    answer = _assistant_text(row)
    return (prompt.strip(), answer.strip())


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _limit_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_source: int | None,
    max_per_prompt_family: int | None,
) -> list[dict[str, Any]]:
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        source = _source_name(row)
        family = _prompt_family(row)
        if max_per_source is not None and source_counts[source] >= max_per_source:
            continue
        if max_per_prompt_family is not None and family_counts[family] >= max_per_prompt_family:
            continue
        source_counts[source] += 1
        family_counts[family] += 1
        selected.append(row)
    return selected


def _sample_rows(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    if target >= len(rows):
        return list(rows)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = set(indices[:target])
    return [row for i, row in enumerate(rows) if i in keep]


def _split_train_val(rows: list[dict[str, Any]], validation_fraction: float, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    rng.shuffle(rows)
    val_count = max(1, int(round(len(rows) * validation_fraction))) if rows else 0
    validation = rows[:val_count]
    train = rows[val_count:]
    return train, validation


def build_mixed_ascii_dataset(config: MixedDatasetConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    synthetic_rows = []
    for path in config.synthetic_paths:
        synthetic_rows.extend(_load_jsonl(path))
    real_rows = []
    for path in config.real_paths:
        real_rows.extend(_load_jsonl(path))

    synthetic_rows = _dedupe_rows(synthetic_rows)
    real_rows = _dedupe_rows(real_rows)

    synthetic_rows = _limit_rows(
        synthetic_rows,
        max_per_source=config.max_per_source,
        max_per_prompt_family=config.max_per_prompt_family,
    )
    real_rows = _limit_rows(
        real_rows,
        max_per_source=config.max_per_source,
        max_per_prompt_family=config.max_per_prompt_family,
    )

    if config.target_total is None:
        target_real = len(real_rows)
        target_synth = len(synthetic_rows)
    else:
        target_real = min(len(real_rows), max(1, int(round(config.target_total * config.real_fraction_min))))
        target_real_cap = min(len(real_rows), max(target_real, int(round(config.target_total * config.real_fraction_max))))
        target_real = min(target_real_cap, target_real)
        remaining = max(0, config.target_total - target_real)
        target_synth = min(len(synthetic_rows), remaining)
        if target_synth < remaining:
            extra_real = min(len(real_rows) - target_real, remaining - target_synth)
            target_real += max(0, extra_real)

    selected_real = _sample_rows(real_rows, target_real, rng)
    selected_synth = _sample_rows(synthetic_rows, target_synth, rng)
    combined = selected_real + selected_synth
    combined = _dedupe_rows(combined)
    train_rows, validation_rows = _split_train_val(combined, config.validation_fraction, rng)

    output_dir = Path(config.output_dir)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    summary_path = output_dir / "summary.json"
    if train_path.exists():
        train_path.unlink()
    if validation_path.exists():
        validation_path.unlink()

    for row in train_rows:
        append_jsonl(train_path, row)
    for row in validation_rows:
        append_jsonl(validation_path, row)

    def by_source(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in rows:
            counts[_source_name(row)] += 1
        return dict(counts)

    summary = {
        'target_total': config.target_total,
        'seed': config.seed,
        'selected_total': len(combined),
        'train_count': len(train_rows),
        'validation_count': len(validation_rows),
        'selected_real': len(selected_real),
        'selected_synthetic': len(selected_synth),
        'by_source': by_source(combined),
        'top_prompt_families': Counter(_prompt_family(row) for row in combined).most_common(20),
        'synthetic_paths': list(config.synthetic_paths),
        'real_paths': list(config.real_paths),
    }
    write_summary(summary_path, summary)
    return summary
