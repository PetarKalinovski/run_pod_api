from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from runpodx.training.synthetic_ascii.filtering import evaluate_synthetic_ascii
from runpodx.training.synthetic_ascii.image_to_ascii import AsciiRenderConfig, render_image_to_ascii
from runpodx.training.synthetic_ascii.public_image_sources import PublicImageRecord, load_public_image_records
from runpodx.training.synthetic_ascii.storage import append_jsonl, write_summary

Renderer = Callable[[object, AsciiRenderConfig], str]


def run_public_image_ascii_pilot(
    *,
    output_dir: str | Path,
    records: list[PublicImageRecord],
    renderer: Renderer = render_image_to_ascii,
    render_config: AsciiRenderConfig | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_config = render_config or AsciiRenderConfig()

    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    summary_path = output_dir / "summary.json"

    accepted = 0
    rejected = 0
    by_dataset_attempted: dict[str, int] = {}
    by_dataset_accepted: dict[str, int] = {}

    for record in records:
        by_dataset_attempted[record.source_dataset] = by_dataset_attempted.get(record.source_dataset, 0) + 1
        ascii_text = renderer(record.image, render_config)
        evaluation = evaluate_synthetic_ascii(
            family_id=record.subject,
            bucket=record.bucket,
            prompt=record.prompt,
            ascii_text=ascii_text,
            generator_prompt=record.prompt,
        )
        if evaluation.accepted:
            accepted += 1
            by_dataset_accepted[record.source_dataset] = by_dataset_accepted.get(record.source_dataset, 0) + 1
            append_jsonl(
                accepted_path,
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": record.prompt},
                        {"role": "assistant", "content": ascii_text},
                    ],
                    "source": "public_image_to_ascii",
                    "bucket": record.bucket,
                    "category": f"public/{record.source_dataset}",
                    "family_id": record.subject,
                    "quality_score": round(evaluation.quality_score, 6),
                    "source_dataset": record.source_dataset,
                    "source_record_id": record.record_id,
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
                    "quality_score": evaluation.quality_score,
                    "rejection_reason": evaluation.rejection_reason,
                },
            )

    summary = {
        "attempted": len(records),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": (accepted / len(records)) if records else 0.0,
        "by_dataset_attempted": by_dataset_attempted,
        "by_dataset_accepted": by_dataset_accepted,
        "render_config": asdict(render_config),
    }
    write_summary(summary_path, summary)
    return summary


def run_public_image_dataset_pilot(
    *,
    dataset_name: str,
    output_dir: str | Path,
    config_name: str = "default",
    split: str = "train",
    limit: int | None = None,
    width: int = 64,
    renderer: Renderer = render_image_to_ascii,
) -> dict:
    records = load_public_image_records(
        dataset_name,
        config_name=config_name,
        split=split,
        limit=limit,
    )
    return run_public_image_ascii_pilot(
        output_dir=output_dir,
        records=records,
        renderer=renderer,
        render_config=AsciiRenderConfig(width=width),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public-image to ASCII pilot.")
    parser.add_argument("dataset_name")
    parser.add_argument("--output-dir", default="artifacts/public_image_ascii/pilot_001")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    summary = run_public_image_dataset_pilot(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
        width=args.width,
    )
    print(summary)


if __name__ == "__main__":
    main()
