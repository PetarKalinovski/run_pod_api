from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from runpodx.training.synthetic_ascii.edge_generation import EdgeGenerationConfig, generate_edge_image
from runpodx.training.synthetic_ascii.filtering import (
    choose_best_ascii_render,
    evaluate_synthetic_ascii,
    final_choice_acceptance,
)
from runpodx.training.synthetic_ascii.image_to_ascii import AsciiRenderConfig, render_image_to_ascii
from runpodx.training.synthetic_ascii.public_image_mesh import load_public_image_mesh_config, load_public_image_records_from_mesh
from runpodx.training.synthetic_ascii.storage import append_jsonl, write_summary


DEFAULT_EDGE_CONFIG = EdgeGenerationConfig(
    method="sobel",
    invert=False,
    blur_sigma=1.4,
    threshold=0.21,
    close_radius=0,
    dilate_radius=0,
    post_blur_sigma=0.4,
)


def run_public_image_mesh_chooser_pilot(
    *,
    mesh_config_path: str | Path,
    output_dir: str | Path,
    width: int = 64,
    edge_config: EdgeGenerationConfig = DEFAULT_EDGE_CONFIG,
) -> dict:
    mesh = load_public_image_mesh_config(mesh_config_path)
    records = load_public_image_records_from_mesh(mesh)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    summary_path = output_dir / "summary.json"
    render_raw = AsciiRenderConfig(width=width, invert=False)
    render_edge = AsciiRenderConfig(width=width, invert=True)

    accepted = 0
    rejected = 0
    chosen_counts = {"raw": 0, "negative_edge": 0}
    final_accept_counts = {"raw": 0, "negative_edge": 0}
    by_dataset_attempted: dict[str, int] = {}
    by_dataset_accepted: dict[str, int] = {}

    for record in records:
        by_dataset_attempted[record.source_dataset] = by_dataset_attempted.get(record.source_dataset, 0) + 1
        raw_ascii = render_image_to_ascii(record.image, render_raw)
        edge_image = generate_edge_image(record.image, edge_config)
        edge_ascii = render_image_to_ascii(edge_image, render_edge)

        raw_eval = evaluate_synthetic_ascii(
            family_id=record.subject,
            bucket=record.bucket,
            prompt=record.prompt,
            ascii_text=raw_ascii,
            generator_prompt=record.prompt,
        )
        edge_eval = evaluate_synthetic_ascii(
            family_id=record.subject,
            bucket=record.bucket,
            prompt=record.prompt,
            ascii_text=edge_ascii,
            generator_prompt=record.prompt,
        )
        choice = choose_best_ascii_render({"raw": raw_eval, "negative_edge": edge_eval})
        final_accepted, overlay_rules = final_choice_acceptance(raw_eval, edge_eval, choice)
        chosen_counts[choice.variant_name] += 1

        if final_accepted:
            accepted += 1
            final_accept_counts[choice.variant_name] += 1
            by_dataset_accepted[record.source_dataset] = by_dataset_accepted.get(record.source_dataset, 0) + 1
            append_jsonl(
                accepted_path,
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": record.prompt},
                        {"role": "assistant", "content": choice.evaluation.ascii_text},
                    ],
                    "source": "public_image_to_ascii_mesh",
                    "source_dataset": record.source_dataset,
                    "source_record_id": record.record_id,
                    "bucket": record.bucket,
                    "category": f"public/{record.source_dataset}",
                    "family_id": record.subject,
                    "prompt_family": record.subject,
                    "quality_score": round(choice.evaluation.quality_score, 6),
                    "raw_quality_score": round(raw_eval.quality_score, 6),
                    "edge_quality_score": round(edge_eval.quality_score, 6),
                    "chosen_variant": choice.variant_name,
                    "overlay_rules": overlay_rules,
                    "edge_config": asdict(edge_config),
                },
            )
        else:
            rejected += 1
            append_jsonl(
                rejected_path,
                {
                    "source_dataset": record.source_dataset,
                    "record_id": record.record_id,
                    "prompt": record.prompt,
                    "subject": record.subject,
                    "raw_quality_score": raw_eval.quality_score,
                    "edge_quality_score": edge_eval.quality_score,
                    "raw_accepted": raw_eval.accepted,
                    "edge_accepted": edge_eval.accepted,
                    "chosen_variant": choice.variant_name,
                    "chosen_quality_score": choice.evaluation.quality_score,
                    "overlay_rules": overlay_rules,
                    "rejection_reason": choice.evaluation.rejection_reason or (overlay_rules[0] if overlay_rules else None),
                },
            )

    summary = {
        "mesh_name": mesh.name,
        "mesh_config_path": str(mesh_config_path),
        "attempted": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": (accepted / len(records)) if records else 0.0,
        "chosen_counts": chosen_counts,
        "final_accept_counts": final_accept_counts,
        "by_dataset_attempted": by_dataset_attempted,
        "by_dataset_accepted": by_dataset_accepted,
        "edge_config": asdict(edge_config),
        "render_raw": asdict(render_raw),
        "render_edge": asdict(render_edge),
    }
    write_summary(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chooser-aware multi-source public-image ASCII pilot.")
    parser.add_argument("mesh_config_path")
    parser.add_argument("--output-dir", default="artifacts/public_image_ascii/mesh_chooser_pilot_001")
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    summary = run_public_image_mesh_chooser_pilot(
        mesh_config_path=args.mesh_config_path,
        output_dir=args.output_dir,
        width=args.width,
    )
    print(summary)


if __name__ == "__main__":
    main()
