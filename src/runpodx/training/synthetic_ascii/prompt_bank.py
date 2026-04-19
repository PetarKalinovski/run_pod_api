from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptFamily:
    family_id: str
    display_name: str
    bucket: str
    subject: str
    templates: tuple[str, ...]


@dataclass(frozen=True)
class PromptSpec:
    family_id: str
    bucket: str
    subject: str
    generator_prompt: str


PILOT_FAMILY_IDS = [
    "sword",
    "lantern",
    "chair",
    "key",
    "skull",
    "cat",
    "owl",
    "fish",
    "horse",
    "castle",
    "house",
    "dragon",
]


def load_pilot_prompt_families() -> list[PromptFamily]:
    base_templates = (
        "A monochrome ink drawing of a {subject}, centered composition, plain light background, clear silhouette, readable outline, medium detail, no text, no logo, no watermark.",
        "A black-and-white line illustration of a {subject}, centered, clean contour, sparse background, readable shape, medium detail, no text.",
        "A simple fantasy pen-and-ink drawing of a {subject}, plain background, clear edges, readable silhouette, no typography.",
        "A minimal monochrome illustration of a {subject}, centered, high contrast, no background clutter, no text.",
    )
    subjects = {
        "sword": ("Sword", "core", "sword"),
        "lantern": ("Lantern", "core", "lantern"),
        "chair": ("Chair", "core", "chair"),
        "key": ("Key", "core", "key"),
        "skull": ("Skull", "core", "skull"),
        "cat": ("Cat", "core", "cat"),
        "owl": ("Owl", "core", "owl"),
        "fish": ("Fish", "core", "fish"),
        "horse": ("Horse", "core", "horse"),
        "castle": ("Castle", "core", "castle"),
        "house": ("House", "core", "house"),
        "dragon": ("Dragon", "core", "dragon"),
    }
    return [
        PromptFamily(
            family_id=family_id,
            display_name=display_name,
            bucket=bucket,
            subject=subject,
            templates=base_templates,
        )
        for family_id, (display_name, bucket, subject) in subjects.items()
    ]


def expand_prompt_family(family: PromptFamily, n: int, seed: int) -> list[PromptSpec]:
    rng = random.Random(seed)
    specs: list[PromptSpec] = []
    for _ in range(n):
        template = rng.choice(family.templates)
        specs.append(
            PromptSpec(
                family_id=family.family_id,
                bucket=family.bucket,
                subject=family.subject,
                generator_prompt=template.format(subject=family.subject),
            )
        )
    return specs
