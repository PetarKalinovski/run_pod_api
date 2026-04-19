from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Protocol
import urllib.request

from runpodx.training.synthetic_ascii.filtering import evaluate_synthetic_ascii
from runpodx.training.synthetic_ascii.image_to_ascii import AsciiRenderConfig, render_image_to_ascii
from runpodx.training.synthetic_ascii.prompt_bank import (
    PromptFamily,
    PromptSpec,
    expand_prompt_family,
    load_pilot_prompt_families,
)
from runpodx.training.synthetic_ascii.storage import append_jsonl, write_summary


@dataclass(frozen=True)
class GenerationResult:
    prompt_spec: PromptSpec
    family_id: str
    image_path: str | None


class ImageGenerationBackend(Protocol):
    def generate(self, prompt_spec: PromptSpec) -> GenerationResult: ...


class StubImageGenerationBackend:
    def generate(self, prompt_spec: PromptSpec) -> GenerationResult:
        return GenerationResult(
            prompt_spec=prompt_spec,
            family_id=prompt_spec.family_id,
            image_path=None,
        )


@dataclass(frozen=True)
class ReplicateFluxSchnellBackend:
    api_token: str
    output_dir: Path
    model_version: str = "black-forest-labs/flux-schnell"
    aspect_ratio: str = "1:1"
    output_format: str = "png"
    megapixels: str = "0.25"
    num_outputs: int = 1
    num_inference_steps: int = 4
    go_fast: bool = True
    disable_safety_checker: bool = False

    def _build_prediction_payload(self, prompt_spec: PromptSpec) -> dict:
        seed = abs(hash((prompt_spec.family_id, prompt_spec.generator_prompt))) % (2**32)
        return {
            "version": self.model_version,
            "input": {
                "prompt": prompt_spec.generator_prompt,
                "aspect_ratio": self.aspect_ratio,
                "output_format": self.output_format,
                "megapixels": self.megapixels,
                "num_outputs": self.num_outputs,
                "num_inference_steps": self.num_inference_steps,
                "go_fast": self.go_fast,
                "disable_safety_checker": self.disable_safety_checker,
                "seed": seed,
            },
        }

    def _request_json(self, *, method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "User-Agent": "runpodx-synthetic-ascii/0.1",
        }
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _create_prediction(self, prompt_spec: PromptSpec) -> dict:
        return self._request_json(
            method="POST",
            url="https://api.replicate.com/v1/predictions",
            payload=self._build_prediction_payload(prompt_spec),
            headers={"Prefer": "wait=60"},
        )

    def _wait_for_prediction(self, prediction_id: str) -> dict:
        return self._request_json(
            method="GET",
            url=f"https://api.replicate.com/v1/predictions/{prediction_id}",
        )

    def _download_output_image(self, url: str, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="synthetic_ascii_", suffix=".png", dir=destination_dir)
        os.close(fd)
        request = urllib.request.Request(url, headers={"User-Agent": "runpodx-synthetic-ascii/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            Path(tmp_path).write_bytes(response.read())
        return Path(tmp_path)

    def generate(self, prompt_spec: PromptSpec) -> GenerationResult:
        prediction = self._create_prediction(prompt_spec)
        if prediction.get("status") not in {"succeeded", "failed", "canceled"}:
            prediction = self._wait_for_prediction(prediction["id"])
        if prediction.get("status") != "succeeded":
            raise RuntimeError(f"Replicate prediction failed: {prediction.get('error') or prediction.get('status')}")
        output = prediction.get("output") or []
        if not output:
            raise RuntimeError("Replicate prediction returned no output image")
        destination_dir = self.output_dir / "generated_images"
        destination_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._download_output_image(output[0], destination_dir)
        return GenerationResult(
            prompt_spec=prompt_spec,
            family_id=prompt_spec.family_id,
            image_path=str(image_path),
        )


Renderer = Callable[[str | Path, AsciiRenderConfig], str]


def build_backend(name: str, *, output_dir: str | Path | None = None) -> ImageGenerationBackend:
    normalized = name.strip().lower()
    if normalized == "stub":
        return StubImageGenerationBackend()
    if normalized == "replicate":
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required for replicate backend")
        return ReplicateFluxSchnellBackend(api_token=api_token, output_dir=Path(output_dir or "."))
    raise ValueError(f"Unknown synthetic image backend: {name}")


def build_training_prompt(subject: str) -> str:
    return f"Draw an ASCII art {subject}."


def build_training_row(*, prompt_spec: PromptSpec, ascii_text: str, quality_score: float) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": build_training_prompt(prompt_spec.subject)},
            {"role": "assistant", "content": ascii_text},
        ],
        "source": "synthetic_image_to_ascii",
        "bucket": prompt_spec.bucket,
        "category": f"synthetic/{prompt_spec.family_id}",
        "family_id": prompt_spec.family_id,
        "quality_score": round(quality_score, 6),
        "generator_prompt": prompt_spec.generator_prompt,
    }


def run_synthetic_ascii_pilot(
    *,
    output_dir: str | Path,
    prompt_specs: list[PromptSpec],
    backend: ImageGenerationBackend,
    renderer: Renderer,
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
    by_family_attempted: dict[str, int] = {}
    by_family_accepted: dict[str, int] = {}

    for prompt_spec in prompt_specs:
        by_family_attempted[prompt_spec.family_id] = by_family_attempted.get(prompt_spec.family_id, 0) + 1
        generation = backend.generate(prompt_spec)
        ascii_text = renderer(generation.image_path or "", render_config)
        evaluation = evaluate_synthetic_ascii(
            family_id=prompt_spec.family_id,
            bucket=prompt_spec.bucket,
            prompt=build_training_prompt(prompt_spec.subject),
            ascii_text=ascii_text,
            generator_prompt=prompt_spec.generator_prompt,
        )
        if evaluation.accepted:
            accepted += 1
            by_family_accepted[prompt_spec.family_id] = by_family_accepted.get(prompt_spec.family_id, 0) + 1
            append_jsonl(
                accepted_path,
                build_training_row(
                    prompt_spec=prompt_spec,
                    ascii_text=ascii_text,
                    quality_score=evaluation.quality_score,
                ),
            )
        else:
            rejected += 1
            append_jsonl(
                rejected_path,
                {
                    "family_id": evaluation.family_id,
                    "bucket": evaluation.bucket,
                    "prompt": evaluation.prompt,
                    "generator_prompt": evaluation.generator_prompt,
                    "ascii_text": evaluation.ascii_text,
                    "quality_score": evaluation.quality_score,
                    "rejection_reason": evaluation.rejection_reason,
                },
            )

    summary = {
        "attempted": len(prompt_specs),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": (accepted / len(prompt_specs)) if prompt_specs else 0.0,
        "by_family_attempted": by_family_attempted,
        "by_family_accepted": by_family_accepted,
        "render_config": asdict(render_config),
    }
    write_summary(summary_path, summary)
    return summary


def build_pilot_prompt_specs(*, families: list[PromptFamily], samples_per_family: int, seed: int) -> list[PromptSpec]:
    prompt_specs: list[PromptSpec] = []
    for idx, family in enumerate(families):
        prompt_specs.extend(expand_prompt_family(family, n=samples_per_family, seed=seed + idx))
    return prompt_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic image-to-ASCII pilot.")
    parser.add_argument("--output-dir", default="artifacts/synthetic_ascii/pilot_001")
    parser.add_argument("--backend", default="stub")
    parser.add_argument("--samples-per-family", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    families = load_pilot_prompt_families()
    prompt_specs = build_pilot_prompt_specs(
        families=families,
        samples_per_family=args.samples_per_family,
        seed=args.seed,
    )
    backend = build_backend(args.backend, output_dir=args.output_dir)
    summary = run_synthetic_ascii_pilot(
        output_dir=args.output_dir,
        prompt_specs=prompt_specs,
        backend=backend,
        renderer=render_image_to_ascii,
        render_config=AsciiRenderConfig(width=args.width),
    )
    print(summary)


if __name__ == "__main__":
    main()
