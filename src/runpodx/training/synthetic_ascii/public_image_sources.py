from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from runpodx.training.dataset_curation import bucket_prompt


@dataclass(frozen=True)
class PublicImageRecord:
    source_dataset: str
    record_id: str
    prompt: str
    bucket: str
    subject: str
    image: Any
    metadata: dict[str, Any]


def _clean_subject(text: str) -> str:
    text = (text or "").strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    for prefix in ("a ", "an ", "the "):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip()


def _build_simple_public_record(
    dataset_name: str,
    row: dict[str, Any],
    *,
    record_id: str,
    include_label: bool = False,
) -> PublicImageRecord:
    subject = _clean_subject(str(row.get("text", "") or ""))
    prompt = f"Draw an ASCII art {subject}."
    metadata: dict[str, Any] = {"raw_text": row.get("text")}
    if include_label:
        metadata["label"] = row.get("label")
    return PublicImageRecord(
        source_dataset=dataset_name,
        record_id=record_id,
        prompt=prompt,
        bucket=bucket_prompt(subject),
        subject=subject,
        image=row.get("image"),
        metadata=metadata,
    )


def normalize_domainnet_clipart_row(row: dict[str, Any], *, record_id: str) -> PublicImageRecord:
    return _build_simple_public_record(
        "Bruece/domainnet-126-by-class-clipart",
        row,
        record_id=record_id,
        include_label=True,
    )


def normalize_officehome_clipart_row(row: dict[str, Any], *, record_id: str) -> PublicImageRecord:
    return _build_simple_public_record(
        "Bruece/office-home-clipart-caption",
        row,
        record_id=record_id,
        include_label=False,
    )


def build_public_image_record(dataset_name: str, row: dict[str, Any], *, record_id: str) -> PublicImageRecord:
    if dataset_name == "Bruece/domainnet-126-by-class-clipart":
        return normalize_domainnet_clipart_row(row, record_id=record_id)
    if dataset_name == "Bruece/office-home-clipart-caption":
        return normalize_officehome_clipart_row(row, record_id=record_id)
    if dataset_name in {
        "Bruece/domainnet-126-by-class-sketch",
        "Bruece/domainnet-126-by-class-painting",
        "Bruece/domainnet-126-by-class-real",
        "Bruece/office-home-by-class-Art",
        "Bruece/office-home-by-class-Product",
        "Bruece/office-home-by-class-Real_World",
        "Bruece/office-home-by-class-Clipart",
    }:
        return _build_simple_public_record(dataset_name, row, record_id=record_id, include_label=True)
    raise ValueError(f"Unsupported public image dataset: {dataset_name}")


def load_public_image_records(
    dataset_name: str,
    *,
    config_name: str = "default",
    split: str = "train",
    limit: int | None = None,
) -> list[PublicImageRecord]:
    ds = load_dataset(dataset_name, config_name, split=split)
    rows = []
    count = len(ds) if limit is None else min(len(ds), limit)
    for idx in range(count):
        rows.append(build_public_image_record(dataset_name, ds[idx], record_id=f"{dataset_name}:{split}:{idx}"))
    return rows
