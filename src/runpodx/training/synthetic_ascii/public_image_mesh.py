from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import tomllib

from runpodx.training.synthetic_ascii.public_image_sources import PublicImageRecord, load_public_image_records


@dataclass(frozen=True)
class PublicImageSourceSpec:
    dataset_name: str
    config_name: str = "default"
    split: str = "train"
    limit: int | None = None
    enabled: bool = True
    alias: str | None = None


@dataclass(frozen=True)
class PublicImageMeshConfig:
    name: str
    sources: tuple[PublicImageSourceSpec, ...]


RecordLoader = Callable[..., list[PublicImageRecord]]


def load_public_image_mesh_config(path: str | Path) -> PublicImageMeshConfig:
    path = Path(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name") or path.stem)
    raw_sources = data.get("source", [])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Mesh config must define at least one [[source]] entry.")

    sources: list[PublicImageSourceSpec] = []
    for item in raw_sources:
        dataset_name = str(item["dataset_name"])
        sources.append(
            PublicImageSourceSpec(
                dataset_name=dataset_name,
                config_name=str(item.get("config_name", "default")),
                split=str(item.get("split", "train")),
                limit=int(item["limit"]) if item.get("limit") is not None else None,
                enabled=bool(item.get("enabled", True)),
                alias=str(item["alias"]) if item.get("alias") is not None else None,
            )
        )
    return PublicImageMeshConfig(name=name, sources=tuple(sources))


def load_public_image_records_from_mesh(
    mesh: PublicImageMeshConfig,
    *,
    loader: RecordLoader = load_public_image_records,
) -> list[PublicImageRecord]:
    rows: list[PublicImageRecord] = []
    for source in mesh.sources:
        if not source.enabled:
            continue
        source_rows = loader(
            source.dataset_name,
            config_name=source.config_name,
            split=source.split,
            limit=source.limit,
        )
        if source.alias is None:
            rows.extend(source_rows)
            continue
        for row in source_rows:
            rows.append(
                PublicImageRecord(
                    source_dataset=source.alias,
                    record_id=row.record_id,
                    prompt=row.prompt,
                    bucket=row.bucket,
                    subject=row.subject,
                    image=row.image,
                    metadata={**row.metadata, "source_dataset_original": row.source_dataset},
                )
            )
    return rows
