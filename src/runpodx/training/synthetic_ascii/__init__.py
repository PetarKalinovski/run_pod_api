"""Optional synthetic ASCII data tooling.

The public package stays lazy so importing ``runpodx.training.synthetic_ascii`` does
not require optional image/ML dependencies such as Pillow, datasets, torch, or
transformers. Import concrete submodules or symbols when you need them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AsciiRenderConfig": "runpodx.training.synthetic_ascii.image_to_ascii",
    "GenerationResult": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "PILOT_FAMILY_IDS": "runpodx.training.synthetic_ascii.prompt_bank",
    "PromptFamily": "runpodx.training.synthetic_ascii.prompt_bank",
    "PromptSpec": "runpodx.training.synthetic_ascii.prompt_bank",
    "PublicImageRecord": "runpodx.training.synthetic_ascii.public_image_sources",
    "RenderChoice": "runpodx.training.synthetic_ascii.filtering",
    "ReplicateFluxSchnellBackend": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "SyntheticAsciiEvaluation": "runpodx.training.synthetic_ascii.filtering",
    "append_jsonl": "runpodx.training.synthetic_ascii.storage",
    "build_backend": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "build_pilot_prompt_specs": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "build_public_image_record": "runpodx.training.synthetic_ascii.public_image_sources",
    "build_training_prompt": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "build_training_row": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "choose_best_ascii_render": "runpodx.training.synthetic_ascii.filtering",
    "evaluate_synthetic_ascii": "runpodx.training.synthetic_ascii.filtering",
    "expand_prompt_family": "runpodx.training.synthetic_ascii.prompt_bank",
    "load_pilot_prompt_families": "runpodx.training.synthetic_ascii.prompt_bank",
    "load_public_image_records": "runpodx.training.synthetic_ascii.public_image_sources",
    "normalize_domainnet_clipart_row": "runpodx.training.synthetic_ascii.public_image_sources",
    "normalize_officehome_clipart_row": "runpodx.training.synthetic_ascii.public_image_sources",
    "render_image_to_ascii": "runpodx.training.synthetic_ascii.image_to_ascii",
    "run_public_image_ascii_pilot": "runpodx.training.synthetic_ascii.public_image_pipeline",
    "run_synthetic_ascii_pilot": "runpodx.training.synthetic_ascii.generate_synthetic_ascii",
    "write_summary": "runpodx.training.synthetic_ascii.storage",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
