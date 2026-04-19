from __future__ import annotations

from dataclasses import dataclass

from runpodx.training.dataset_curation import analyze_ascii_art, score_ascii_art_quality


@dataclass(frozen=True)
class SyntheticAsciiEvaluation:
    family_id: str
    bucket: str
    prompt: str
    generator_prompt: str
    ascii_text: str
    trimmed_ascii_text: str
    accepted: bool
    quality_score: float
    rejection_reason: str | None
    removed_top_lines: int
    removed_bottom_lines: int
    failed_checks: list[str]


@dataclass(frozen=True)
class RenderChoice:
    variant_name: str
    evaluation: SyntheticAsciiEvaluation


def _filled_style_quality_score(text: str) -> float:
    metrics = analyze_ascii_art(text)
    base_score = score_ascii_art_quality(text)

    score = 1.035231428125891 * base_score
    score += 1.1082119569590774 * min(1.5, float(metrics["occupancy_range"]) / 18.0)
    score += 0.7110487690411399 * min(1.5, float(metrics["center_drift"]) / 10.0)
    score += 0.6 * min(1.5, float(metrics["width_std"]) / 14.0)
    score += 0.7 * min(1.5, float(metrics["occupancy_std"]) / 10.0)
    score += 0.6106407025490448 * min(1.5, float(metrics["line_transition_ratio"]) / 0.22)
    score += 0.31020513524886284 * min(1.2, float(metrics["unique_symbol_count"]) / 8.0)

    fill_pref = max(0.0, 1.0 - abs(float(metrics["top3_char_ratio"]) - 0.6804577964038849) / 0.2505527283012033)
    dom_pref = max(0.0, 1.0 - abs(float(metrics["dominant_char_ratio"]) - 0.23000175206679196) / 0.18)
    score += 0.8784643447538785 * fill_pref
    score += 0.5 * dom_pref

    score -= 1.258244497529739 * float(metrics["repeat_fraction"])
    score -= 0.6 * min(1.5, int(metrics["max_consecutive_repeat"]) / 12.0)
    score -= 0.617388740687673 * float(metrics["blank_line_fraction"])
    score -= 1.0593660115085362 * float(metrics["alnum_fraction"])
    score -= 0.6697508007324786 * float(metrics["lattice_line_fraction"])
    score -= 0.07548944266747842 * float(metrics["repeated_bigram_ratio"])

    if int(metrics["non_empty_line_count"]) < 5:
        score -= 0.8
    return score


def _failed_checks(text: str, *, quality_score: float) -> list[str]:
    metrics = analyze_ascii_art(text)
    failures: list[str] = []
    if int(metrics["non_empty_line_count"]) < 4:
        failures.append("too_few_non_empty_lines")
    if int(metrics["line_count"]) > 90:
        failures.append("too_many_lines")
    if int(metrics["char_count"]) < 30:
        failures.append("too_few_characters")
    if int(metrics["char_count"]) > 6000:
        failures.append("too_many_characters")
    if float(metrics["repeat_fraction"]) > 0.86:
        failures.append("repeat_fraction_too_high")
    if int(metrics["max_consecutive_repeat"]) > 20:
        failures.append("max_consecutive_repeat_too_high")
    if float(metrics["dominant_char_ratio"]) > 0.8055641154196528:
        failures.append("dominant_char_ratio_too_high")
    if int(metrics["unique_symbol_count"]) < 2:
        failures.append("too_few_unique_symbols")
    if float(metrics["top3_char_ratio"]) > 0.962163498655398:
        failures.append("top3_char_ratio_too_high")
    if float(metrics["lattice_line_fraction"]) > 0.9696303178337911:
        failures.append("lattice_line_fraction_too_high")
    if float(metrics["width_std"]) < 0.17769216455274134:
        failures.append("width_variation_too_low")
    if float(metrics["occupancy_std"]) < 0.22635137835384048:
        failures.append("occupancy_variation_too_low")
    if float(metrics["line_transition_ratio"]) < 0.014105107915477984:
        failures.append("line_transition_ratio_too_low")
    if float(metrics["alnum_fraction"]) > 0.3:
        failures.append("alnum_fraction_too_high")
    if float(metrics["blank_line_fraction"]) > 0.5611770657132085:
        failures.append("blank_line_fraction_too_high")
    if quality_score < 4.000036849274758:
        failures.append("quality_below_threshold")
    return failures


def evaluate_synthetic_ascii(
    *,
    family_id: str,
    bucket: str,
    prompt: str,
    ascii_text: str,
    generator_prompt: str,
) -> SyntheticAsciiEvaluation:
    quality_score = _filled_style_quality_score(ascii_text)
    failed_checks = _failed_checks(ascii_text, quality_score=quality_score)
    accepted = not failed_checks
    rejection_reason = None if accepted else failed_checks[0]
    return SyntheticAsciiEvaluation(
        family_id=family_id,
        bucket=bucket,
        prompt=prompt,
        generator_prompt=generator_prompt,
        ascii_text=ascii_text,
        trimmed_ascii_text=ascii_text,
        accepted=accepted,
        quality_score=quality_score,
        rejection_reason=rejection_reason,
        removed_top_lines=0,
        removed_bottom_lines=0,
        failed_checks=failed_checks,
    )


def choose_best_ascii_render(candidates: dict[str, SyntheticAsciiEvaluation]) -> RenderChoice:
    accepted = [(name, evaluation) for name, evaluation in candidates.items() if evaluation.accepted]
    pool = accepted if accepted else list(candidates.items())

    def adjusted_score(name: str, evaluation: SyntheticAsciiEvaluation) -> float:
        bias = 0.15 if name == "raw" else 0.11732099502246368
        return evaluation.quality_score + bias

    variant_name, evaluation = max(
        pool,
        key=lambda item: (
            adjusted_score(item[0], item[1]),
            -(item[1].removed_top_lines + item[1].removed_bottom_lines),
            len(item[1].trimmed_ascii_text),
        ),
    )
    return RenderChoice(variant_name=variant_name, evaluation=evaluation)


def _extra_metrics(text: str) -> dict[str, float | int]:
    metrics = analyze_ascii_art(text)
    lines = text.splitlines()
    dense = set("@%#*+")
    border_dense = center_dense = total_border = total_center = 0
    for line in lines:
        if not line:
            continue
        n = len(line)
        edge_width = max(1, n // 8)
        border = line[:edge_width] + line[-edge_width:]
        center = line[n // 4 : max(n // 4 + 1, 3 * n // 4)]
        border_dense += sum(ch in dense for ch in border)
        total_border += len(border)
        center_dense += sum(ch in dense for ch in center)
        total_center += len(center)
    metrics["bg"] = border_dense / max(1, total_border) - center_dense / max(1, total_center)
    return metrics


def overlay_rule_names(
    raw_evaluation: SyntheticAsciiEvaluation,
    edge_evaluation: SyntheticAsciiEvaluation,
    chosen_variant: str,
) -> list[str]:
    raw_metrics = _extra_metrics(raw_evaluation.ascii_text)
    edge_metrics = _extra_metrics(edge_evaluation.ascii_text)
    chosen_metrics = raw_metrics if chosen_variant == "raw" else edge_metrics
    other_metrics = edge_metrics if chosen_variant == "raw" else raw_metrics
    triggered: list[str] = []
    if (
        chosen_variant == "raw"
        and float(chosen_metrics["bg"]) > 0.08
        and float(other_metrics["top3_char_ratio"]) > 0.7
        and float(other_metrics["lattice_line_fraction"]) > 0.8
    ):
        triggered.append("raw_bg_other")
    if (
        chosen_variant == "raw"
        and float(chosen_metrics["bg"]) < -0.02
        and float(chosen_metrics["lattice_line_fraction"]) > 0.78
        and float(chosen_metrics["repeated_bigram_ratio"]) > 0.992
        and float(chosen_metrics["top3_char_ratio"]) < 0.66
    ):
        triggered.append("raw_neg_bg_carpet")
    if (
        chosen_variant == "raw"
        and float(other_metrics["lattice_line_fraction"]) > 0.95
        and float(chosen_metrics["lattice_line_fraction"]) > 0.69
        and float(chosen_metrics["repeated_bigram_ratio"]) > 0.989
        and float(chosen_metrics["top3_char_ratio"]) < 0.64
    ):
        triggered.append("raw_other_extreme")
    if (
        chosen_variant == "negative_edge"
        and float(chosen_metrics["lattice_line_fraction"]) > 0.72
        and float(chosen_metrics["repeated_bigram_ratio"]) > 0.99
        and float(other_metrics["lattice_line_fraction"]) > 0.35
        and float(chosen_metrics["top3_char_ratio"]) < 0.6
        and float(chosen_metrics["dominant_char_ratio"]) < 0.22
    ):
        triggered.append("edge_flat_carpet")
    return triggered


def final_choice_acceptance(
    raw_evaluation: SyntheticAsciiEvaluation,
    edge_evaluation: SyntheticAsciiEvaluation,
    choice: RenderChoice,
) -> tuple[bool, list[str]]:
    triggered = overlay_rule_names(raw_evaluation, edge_evaluation, choice.variant_name)
    return choice.evaluation.accepted and not triggered, triggered
