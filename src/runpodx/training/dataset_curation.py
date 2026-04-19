from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SIGNATURE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\- ]{1,11}$")
_LOGO_RE = re.compile(r"\blogo\b|\bemblem\b|\bbadge\b|\bbrand\b", re.IGNORECASE)
_CORE_OBJECT_RE = re.compile(
    r"\bsword\b|\bweapons?\b|\bcastle\b|\bcat\b|\bdragon\b|\btree\b|\bplant\b|\banimals?\b|\bcreatures?\b|\bbuildings?\b|\bhouse\b|\bmonuments?\b|\bvehicles?\b|\bcar\b|\bship\b|\bairplanes?\b|\bowl\b|\bhorse\b|\bchair\b|\blantern\b|\bkey\b|\bskull\b|\brobot\b|\bknight\b|\bpeople\b|\bmythology\b|\bnature\b|\breptiles?\b|\bbirds?\b|\bfish\b|\bdeer\b|\bunicorns?\b",
    re.IGNORECASE,
)
_LATTICE_LINE_RE = re.compile(r"^[ .+#@*=:_|`'\-]{6,}$")
_PROMPT_FAMILY_CLEANUP_RE = re.compile(
    r"\b(?:draw|create|generate|make|an|a|the|in ascii art|ascii art|only output the ascii art|do not output any other text|width|height|medium size|small size|large size|color|colou?r|following category|category|please|shown|with|and|of|in)\b",
    re.IGNORECASE,
)
_PROMPT_DIMENSION_RE = re.compile(r"\b(?:width|height)\s*[:=]?\s*\d+\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_PRODUCT_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\boffice\s+chair\b", re.IGNORECASE), "office chair"),
    (re.compile(r"\brecliner\s+chair\b", re.IGNORECASE), "recliner chair"),
    (re.compile(r"\bgaming\s+chair\b", re.IGNORECASE), "gaming chair"),
    (re.compile(r"\bhanging\s+chair\b", re.IGNORECASE), "hanging chair"),
    (re.compile(r"\bchair\b", re.IGNORECASE), "chair"),
    (re.compile(r"\bsunglasses\b", re.IGNORECASE), "sunglasses"),
    (re.compile(r"\bglasses\b", re.IGNORECASE), "glasses"),
    (re.compile(r"\bsneakers\b|\bconverse\b|\bnike\b", re.IGNORECASE), "sneakers"),
    (re.compile(r"\bshoes\b|\bpumps\b|\bfootwear\b|\bboots\b", re.IGNORECASE), "shoes"),
    (re.compile(r"\bbracelet\b", re.IGNORECASE), "bracelet"),
    (re.compile(r"\bnecklace\b|\bpendant\b", re.IGNORECASE), "necklace"),
    (re.compile(r"\bbag\b|\btote\b|\bpurse\b|\bhandbag\b", re.IGNORECASE), "bag"),
    (re.compile(r"\bdress\b|\bjumpsuit\b|\bcostume\b", re.IGNORECASE), "dress"),
    (re.compile(r"\bmotorcycle\b", re.IGNORECASE), "motorcycle"),
    (re.compile(r"\bsnowmobile\b", re.IGNORECASE), "snowmobile"),
    (re.compile(r"\bsaw\b", re.IGNORECASE), "saw"),
    (re.compile(r"\btelevision\b|\btv\b", re.IGNORECASE), "television"),
]
_GENERIC_DESCRIPTOR_RE = re.compile(
    r"\b(?:black|white|blue|green|red|yellow|brown|grey|gray|gold|silver|pink|purple|orange|wooden|wood|metal|plastic|leather|chrome|mesh|fabric|bamboo|made|from|has|have|is|are|on|top|front|background|standing|shown|worn|comfortable|elegant|perfect|clear|lenses|frames|wheels|arm|rests|seat|upholstered|body|blade|pair|small|large|open|closed|holding|pointing|various|types)\b",
    re.IGNORECASE,
)


def normalize_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def clean_current_ascii_art(text: str) -> str:
    lines = normalize_newlines(text).split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        tail = lines[-1].strip()
        if _SIGNATURE_RE.fullmatch(tail):
            lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _safe_stddev(values: list[float | int]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _repeated_ngram_ratio(text: str, n: int = 2) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if len(visible) < n:
        return 0.0
    ngrams = ["".join(visible[idx : idx + n]) for idx in range(len(visible) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def analyze_ascii_art(text: str) -> dict[str, float | int]:
    lines = normalize_newlines(text).split("\n")
    if lines == [""]:
        lines = []

    non_empty = [line for line in lines if line.strip()]
    unique_non_empty = len(set(non_empty))
    visible_chars = sum(1 for ch in text if not ch.isspace())
    total_chars = max(1, sum(len(line) for line in lines))
    density = visible_chars / total_chars

    max_repeat = 0
    current = 0
    prev = None
    for line in non_empty:
        if line == prev:
            current += 1
        else:
            current = 1
            prev = line
        max_repeat = max(max_repeat, current)

    repeat_fraction = 0.0
    if non_empty:
        repeat_fraction = 1.0 - (unique_non_empty / len(non_empty))

    non_space_chars = [ch for ch in text if not ch.isspace()]
    char_counter = Counter(non_space_chars)
    dominant_char_ratio = (max(char_counter.values()) / len(non_space_chars)) if non_space_chars else 0.0
    unique_symbol_count = len({ch for ch in non_space_chars if not ch.isalnum()})
    top3_char_ratio = 0.0
    if non_space_chars:
        top3 = sum(v for _, v in char_counter.most_common(3))
        top3_char_ratio = top3 / len(non_space_chars)

    lattice_like_lines = sum(1 for line in non_empty if _LATTICE_LINE_RE.fullmatch(line.strip()))
    lattice_line_fraction = (lattice_like_lines / len(non_empty)) if non_empty else 0.0

    widths = [len(line.rstrip()) for line in lines]
    non_empty_widths = [len(line.rstrip()) for line in non_empty]
    occupancies = [sum(1 for ch in line if not ch.isspace()) for line in non_empty]
    occupancy_deltas = [abs(curr - prev) for prev, curr in zip(occupancies, occupancies[1:])]

    left_edges = []
    right_edges = []
    centers = []
    for line in non_empty:
        visible_positions = [idx for idx, ch in enumerate(line) if not ch.isspace()]
        if not visible_positions:
            continue
        left = visible_positions[0]
        right = visible_positions[-1]
        left_edges.append(left)
        right_edges.append(right)
        centers.append((left + right) / 2.0)

    line_transition_scores = []
    for prev_line, curr_line in zip(non_empty, non_empty[1:]):
        prev_chars = set(prev_line.rstrip())
        curr_chars = set(curr_line.rstrip())
        union = prev_chars | curr_chars
        if not union:
            continue
        overlap = len(prev_chars & curr_chars) / len(union)
        line_transition_scores.append(1.0 - overlap)

    alnum_count = sum(ch.isalnum() for ch in non_space_chars)
    blank_line_fraction = ((len(lines) - len(non_empty)) / len(lines)) if lines else 0.0
    alnum_fraction = (alnum_count / len(non_space_chars)) if non_space_chars else 0.0

    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
        "char_count": sum(len(line) for line in lines),
        "density": density,
        "repeat_fraction": repeat_fraction,
        "max_consecutive_repeat": max_repeat,
        "dominant_char_ratio": dominant_char_ratio,
        "unique_symbol_count": unique_symbol_count,
        "top3_char_ratio": top3_char_ratio,
        "lattice_line_fraction": lattice_line_fraction,
        "max_line_width": max(widths) if widths else 0,
        "mean_line_width": (sum(widths) / len(widths)) if widths else 0.0,
        "width_std": _safe_stddev(widths),
        "non_empty_width_std": _safe_stddev(non_empty_widths),
        "occupancy_std": _safe_stddev(occupancies),
        "occupancy_range": (max(occupancies) - min(occupancies)) if occupancies else 0,
        "adjacent_occupancy_delta_mean": (
            sum(occupancy_deltas) / len(occupancy_deltas) if occupancy_deltas else 0.0
        ),
        "left_edge_std": _safe_stddev(left_edges),
        "right_edge_std": _safe_stddev(right_edges),
        "center_drift": _safe_stddev(centers),
        "line_transition_ratio": (
            sum(line_transition_scores) / len(line_transition_scores) if line_transition_scores else 0.0
        ),
        "alnum_fraction": alnum_fraction,
        "blank_line_fraction": blank_line_fraction,
        "repeated_bigram_ratio": _repeated_ngram_ratio(text, n=2),
        "repeated_trigram_ratio": _repeated_ngram_ratio(text, n=3),
    }


def passes_quality_filters(
    text: str,
    *,
    min_lines: int,
    max_lines: int,
    min_chars: int = 10,
    max_chars: int = 4000,
    max_repeat_fraction: float = 0.9,
    max_consecutive_repeat: int = 40,
    max_dominant_char_ratio: float = 0.85,
    min_unique_symbol_count: int = 2,
    max_top3_char_ratio: float = 0.95,
    max_lattice_line_fraction: float = 0.7,
    min_width_std: float = 0.0,
    min_occupancy_std: float = 0.0,
    min_line_transition_ratio: float = 0.0,
    max_alnum_fraction: float = 1.0,
    max_blank_line_fraction: float = 1.0,
    max_repeated_bigram_ratio: float = 1.0,
) -> bool:
    metrics = analyze_ascii_art(text)
    if int(metrics["non_empty_line_count"]) < min_lines:
        return False
    if int(metrics["line_count"]) > max_lines:
        return False
    if int(metrics["char_count"]) < min_chars or int(metrics["char_count"]) > max_chars:
        return False
    if float(metrics["repeat_fraction"]) > max_repeat_fraction:
        return False
    if int(metrics["max_consecutive_repeat"]) > max_consecutive_repeat:
        return False
    if float(metrics["dominant_char_ratio"]) > max_dominant_char_ratio:
        return False
    if int(metrics["unique_symbol_count"]) < min_unique_symbol_count:
        return False
    if float(metrics["top3_char_ratio"]) > max_top3_char_ratio:
        return False
    if float(metrics["lattice_line_fraction"]) > max_lattice_line_fraction:
        return False
    if float(metrics["width_std"]) < min_width_std:
        return False
    if float(metrics["occupancy_std"]) < min_occupancy_std:
        return False
    if float(metrics["line_transition_ratio"]) < min_line_transition_ratio:
        return False
    if float(metrics["alnum_fraction"]) > max_alnum_fraction:
        return False
    if float(metrics["blank_line_fraction"]) > max_blank_line_fraction:
        return False
    if float(metrics["repeated_bigram_ratio"]) > max_repeated_bigram_ratio:
        return False
    return True


def score_ascii_art_quality(text: str) -> float:
    metrics = analyze_ascii_art(text)
    shape_variation_score = min(1.0, float(metrics["occupancy_range"]) / 18.0) + min(
        1.0, float(metrics["width_std"]) / 12.0
    )
    occupancy_profile_score = min(1.0, float(metrics["occupancy_std"]) / 8.0) + min(
        1.0, float(metrics["adjacent_occupancy_delta_mean"]) / 6.0
    )
    edge_profile_score = min(1.0, float(metrics["left_edge_std"]) / 6.0) + min(
        1.0, float(metrics["right_edge_std"]) / 6.0
    )
    symbol_diversity_score = min(1.0, int(metrics["unique_symbol_count"]) / 8.0)

    repetition_penalty = (
        float(metrics["repeat_fraction"])
        + min(1.0, int(metrics["max_consecutive_repeat"]) / 8.0)
        + min(1.0, float(metrics["repeated_bigram_ratio"]))
    )
    lattice_penalty = float(metrics["lattice_line_fraction"])
    logo_text_penalty = min(1.0, float(metrics["alnum_fraction"]) * 3.0)
    ultra_dense_texture_penalty = max(0.0, float(metrics["top3_char_ratio"]) - 0.55) * 2.0
    trivial_length_penalty = 1.0 if int(metrics["non_empty_line_count"]) < 6 else 0.0

    return (
        1.2 * shape_variation_score
        + 1.0 * occupancy_profile_score
        + 0.8 * edge_profile_score
        + 0.7 * symbol_diversity_score
        - 1.2 * repetition_penalty
        - 1.0 * lattice_penalty
        - 0.8 * logo_text_penalty
        - 0.8 * ultra_dense_texture_penalty
        - 0.7 * trivial_length_penalty
    )


def _normalize_art_for_dedupe(text: str) -> str:
    lines = [ln.rstrip() for ln in normalize_newlines(text).split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _prompt_key(normalized_record: dict[str, Any]) -> str:
    messages = normalized_record.get("messages") or []
    if len(messages) < 2:
        return ""
    return str(messages[1].get("content", "") or "").strip().lower()


def prompt_family_key(prompt: str) -> str:
    text = str(prompt or "").strip().lower()

    for pattern, canonical in _PRODUCT_FAMILY_PATTERNS:
        if pattern.search(text):
            return canonical

    if "create an ascii drawing in the following category:" in text:
        category = text.split("create an ascii drawing in the following category:", 1)[1]
        category = category.split(". only output the ascii art", 1)[0]
        category = category.replace("/", " ").strip()
        category = _GENERIC_DESCRIPTOR_RE.sub(" ", category)
        category = _NON_WORD_RE.sub(" ", category)
        category_tokens = [token for token in category.split() if token]
        if category_tokens:
            return " ".join(category_tokens[:4])

    text = _PROMPT_DIMENSION_RE.sub(" ", text)
    text = _PROMPT_FAMILY_CLEANUP_RE.sub(" ", text)
    text = _GENERIC_DESCRIPTOR_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    tokens = [token for token in text.split() if token]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " ".join(tokens[:4])


def bucket_prompt(prompt: str) -> str:
    text = str(prompt or "")
    if _LOGO_RE.search(text):
        return "logo"
    if _CORE_OBJECT_RE.search(text):
        return "core"
    return "general"


def normalize_140k_example(example: dict[str, Any]) -> dict[str, Any] | None:
    conversations = example.get("conversations") or []
    if len(conversations) < 2:
        return None
    prompt = str(conversations[0].get("content", "") or "").strip()
    assistant = str(conversations[1].get("content", "") or "").strip()
    if not prompt or not assistant:
        return None
    art = assistant.removeprefix("```").removesuffix("```").strip()
    if not art:
        return None
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": normalize_newlines(art).strip()},
        ],
        "source": "ascii140k",
        "bucket": bucket_prompt(prompt),
        "category": "",
    }


def normalize_current_example(example: dict[str, Any]) -> dict[str, Any] | None:
    messages = example.get("messages") or []
    if len(messages) < 3:
        return None
    art = clean_current_ascii_art(example.get("ascii_art", ""))
    if not art:
        return None
    category = str(example.get("category", "") or "")
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Create an ASCII drawing in the following category: {category}. Only output the ASCII art. DO NOT output any other text.",
            },
            {"role": "assistant", "content": art},
        ],
        "source": "avalovelace_ascii_art",
        "bucket": "core" if bucket_prompt(category) == "core" else "general",
        "category": category,
    }


def normalize_apehex_example(example: dict[str, Any]) -> dict[str, Any] | None:
    art = normalize_newlines(str(example.get("content", "") or "")).strip()
    if not art:
        return None

    lines = art.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        tail = lines[-1].strip()
        if _SIGNATURE_RE.fullmatch(tail):
            lines.pop()
    art = "\n".join(lines).strip()
    if not art:
        return None

    labels = str(example.get("labels", "") or "").strip()
    caption = str(example.get("caption", "") or "").strip()
    category = labels.replace(",", "/") if labels else "Miscellaneous/Other"
    prompt_subject = caption or category or "ASCII art"
    prompt = f"Create an ASCII drawing in the following category: {prompt_subject}. Only output the ASCII art. DO NOT output any other text."
    bucket_source = labels or caption or prompt_subject
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": art},
        ],
        "source": "apehex_asciiart",
        "bucket": "core" if bucket_prompt(bucket_source) == "core" else "general",
        "category": category,
    }


def _attach_quality_metadata(record: dict[str, Any], prompt_family: str, quality_score: float) -> dict[str, Any]:
    enriched = dict(record)
    enriched["prompt_family"] = prompt_family
    enriched["quality_score"] = round(quality_score, 6)
    return enriched


def _filter_and_score_record(
    normalized: dict[str, Any],
    *,
    source_kind: str,
    min_quality_score: float | None,
) -> tuple[dict[str, Any], float] | None:
    art = normalized["messages"][2]["content"]
    if source_kind == "current":
        passes = passes_quality_filters(
            art,
            min_lines=3,
            max_lines=90,
            min_chars=30,
            max_chars=6000,
            max_repeat_fraction=0.68,
            max_consecutive_repeat=10,
            max_dominant_char_ratio=0.68,
            min_unique_symbol_count=3,
            max_top3_char_ratio=0.84,
            max_lattice_line_fraction=0.45,
            min_width_std=0.5,
            min_occupancy_std=0.5,
            min_line_transition_ratio=0.03,
            max_alnum_fraction=0.24,
            max_blank_line_fraction=0.35,
            max_repeated_bigram_ratio=0.97,
        )
    else:
        passes = passes_quality_filters(
            art,
            min_lines=4,
            max_lines=70,
            min_chars=50,
            max_chars=3600,
            max_repeat_fraction=0.68,
            max_consecutive_repeat=10,
            max_dominant_char_ratio=0.68,
            min_unique_symbol_count=3,
            max_top3_char_ratio=0.84,
            max_lattice_line_fraction=0.45,
            min_width_std=0.5,
            min_occupancy_std=0.5,
            min_line_transition_ratio=0.03,
            max_alnum_fraction=0.24,
            max_blank_line_fraction=0.35,
            max_repeated_bigram_ratio=0.97,
        )
    if not passes:
        return None

    quality_score = score_ascii_art_quality(art)
    if min_quality_score is not None and quality_score < min_quality_score:
        return None
    return normalized, quality_score


def curate_mixed_records(
    current_rows: list[dict[str, Any]],
    rows_140k: list[dict[str, Any]],
    apehex_rows: list[dict[str, Any]] | None = None,
    *,
    seed: int = 42,
    target_140k_core: int = 9000,
    target_140k_logo: int = 600,
    target_140k_general: int = 4000,
    target_apehex_core: int = 2500,
    target_apehex_general: int = 2500,
    max_per_prompt_current: int = 3,
    max_per_prompt_140k: int = 2,
    max_per_prompt_family_current: int = 6,
    max_per_prompt_family_140k: int = 4,
    include_logos: bool = False,
    max_logo_ratio: float = 0.05,
    min_quality_score: float | None = None,
    manual_anchor_rows: list[dict[str, Any]] | None = None,
    target_total: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    curated_current: list[tuple[float, dict[str, Any]]] = []
    seen_art = set()
    current_prompt_counts = Counter()
    current_family_counts = Counter()

    def maybe_add_record(
        normalized: dict[str, Any],
        *,
        source_kind: str,
        prompt_cap: int,
        family_cap: int,
        prompt_counts: Counter,
        family_counts: Counter,
        target_list: list[tuple[float, dict[str, Any]]],
    ) -> None:
        prompt_key = _prompt_key(normalized)
        prompt_family = prompt_family_key(prompt_key)
        if prompt_counts[prompt_key] >= prompt_cap:
            return
        if family_counts[prompt_family] >= family_cap:
            return
        scored = _filter_and_score_record(normalized, source_kind=source_kind, min_quality_score=min_quality_score)
        if scored is None:
            return
        _, quality_score = scored
        art = normalized["messages"][2]["content"]
        art_key = _normalize_art_for_dedupe(art)
        if art_key in seen_art:
            return
        seen_art.add(art_key)
        prompt_counts[prompt_key] += 1
        family_counts[prompt_family] += 1
        target_list.append((quality_score, _attach_quality_metadata(normalized, prompt_family, quality_score)))

    for row in current_rows:
        normalized = normalize_current_example(row)
        if not normalized:
            continue
        category = normalized.get("category", "")
        if category in {"Miscellaneous/One line", "Art and design/Dividers"}:
            continue
        maybe_add_record(
            normalized,
            source_kind="current",
            prompt_cap=max_per_prompt_current,
            family_cap=max_per_prompt_family_current,
            prompt_counts=current_prompt_counts,
            family_counts=current_family_counts,
            target_list=curated_current,
        )

    anchor_records: list[tuple[float, dict[str, Any]]] = []
    if manual_anchor_rows:
        anchor_prompt_counts = Counter()
        anchor_family_counts = Counter()
        for row in manual_anchor_rows:
            normalized = normalize_current_example(row) if row.get("ascii_art") else row
            if not normalized or not normalized.get("messages"):
                continue
            normalized = dict(normalized)
            normalized["source"] = normalized.get("source", "manual_anchor")
            normalized["bucket"] = normalized.get("bucket") or bucket_prompt(_prompt_key(normalized))
            maybe_add_record(
                normalized,
                source_kind="current",
                prompt_cap=max_per_prompt_current,
                family_cap=max_per_prompt_family_current,
                prompt_counts=anchor_prompt_counts,
                family_counts=anchor_family_counts,
                target_list=anchor_records,
            )

    buckets: dict[str, list[tuple[float, dict[str, Any]]]] = {"core": [], "logo": [], "general": []}
    bucket_prompt_counts = {"core": Counter(), "logo": Counter(), "general": Counter()}
    bucket_family_counts = {"core": Counter(), "logo": Counter(), "general": Counter()}
    for row in rows_140k:
        normalized = normalize_140k_example(row)
        if not normalized:
            continue
        bucket = normalized["bucket"]
        if bucket == "logo" and not include_logos:
            continue
        maybe_add_record(
            normalized,
            source_kind="140k",
            prompt_cap=max_per_prompt_140k,
            family_cap=max_per_prompt_family_140k,
            prompt_counts=bucket_prompt_counts[bucket],
            family_counts=bucket_family_counts[bucket],
            target_list=buckets[bucket],
        )

    apehex_buckets: dict[str, list[tuple[float, dict[str, Any]]]] = {"core": [], "general": []}
    apehex_prompt_counts = {"core": Counter(), "general": Counter()}
    apehex_family_counts = {"core": Counter(), "general": Counter()}
    for row in apehex_rows or []:
        normalized = normalize_apehex_example(row)
        if not normalized:
            continue
        bucket = normalized["bucket"]
        if bucket == "logo":
            continue
        target_bucket = bucket if bucket in apehex_buckets else "general"
        maybe_add_record(
            normalized,
            source_kind="current",
            prompt_cap=max_per_prompt_140k,
            family_cap=max_per_prompt_family_140k,
            prompt_counts=apehex_prompt_counts[target_bucket],
            family_counts=apehex_family_counts[target_bucket],
            target_list=apehex_buckets[target_bucket],
        )

    curated_current.sort(key=lambda item: item[0], reverse=True)
    anchor_records.sort(key=lambda item: item[0], reverse=True)
    for values in buckets.values():
        values.sort(key=lambda item: item[0], reverse=True)
    for values in apehex_buckets.values():
        values.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict[str, Any]] = [record for _, record in anchor_records]
    selected.extend(record for _, record in curated_current)
    selected.extend(record for _, record in apehex_buckets["core"][:target_apehex_core])
    selected.extend(record for _, record in buckets["core"][:target_140k_core])
    if include_logos and target_140k_logo > 0:
        selected.extend(record for _, record in buckets["logo"][:target_140k_logo])
    selected.extend(record for _, record in apehex_buckets["general"][:target_apehex_general])
    selected.extend(record for _, record in buckets["general"][:target_140k_general])

    if include_logos and max_logo_ratio >= 0.0:
        max_logo_count = math.floor(len(selected) * max_logo_ratio)
        logos = [row for row in selected if row.get("bucket") == "logo"]
        if len(logos) > max_logo_count:
            kept_logos = logos[:max_logo_count]
            non_logos = [row for row in selected if row.get("bucket") != "logo"]
            selected = non_logos + kept_logos

    if target_total is not None and len(selected) > target_total:
        selected.sort(key=lambda row: float(row.get("quality_score", 0.0)), reverse=True)
        selected = selected[:target_total]

    rng.shuffle(selected)
    return selected


def summarize_curated_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter()
    by_bucket = Counter()
    by_source_bucket = Counter()
    by_prompt_family = Counter()
    quality_scores = []
    for row in records:
        source = str(row.get("source", "unknown"))
        bucket = str(row.get("bucket", "unknown"))
        by_source[source] += 1
        by_bucket[bucket] += 1
        by_source_bucket[f"{source}:{bucket}"] += 1
        prompt_family = str(row.get("prompt_family", "") or "")
        if prompt_family:
            by_prompt_family[prompt_family] += 1
        try:
            quality_scores.append(float(row.get("quality_score", 0.0)))
        except (TypeError, ValueError):
            pass

    quality_scores_sorted = sorted(quality_scores)
    def percentile(p: float) -> float:
        if not quality_scores_sorted:
            return 0.0
        idx = min(len(quality_scores_sorted) - 1, max(0, int(round((len(quality_scores_sorted) - 1) * p))))
        return quality_scores_sorted[idx]

    return {
        "total_rows": len(records),
        "by_source": dict(by_source),
        "by_bucket": dict(by_bucket),
        "by_source_bucket": dict(by_source_bucket),
        "top_prompt_families": by_prompt_family.most_common(20),
        "quality_score": {
            "min": min(quality_scores_sorted) if quality_scores_sorted else 0.0,
            "p25": percentile(0.25),
            "median": percentile(0.5),
            "p75": percentile(0.75),
            "max": max(quality_scores_sorted) if quality_scores_sorted else 0.0,
            "mean": (sum(quality_scores_sorted) / len(quality_scores_sorted)) if quality_scores_sorted else 0.0,
        },
    }


def split_train_validation(records: list[dict[str, Any]], validation_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    rows = list(records)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * validation_ratio))
    return rows[cut:], rows[:cut]
