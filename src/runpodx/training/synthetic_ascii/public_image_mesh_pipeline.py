from __future__ import annotations

import argparse
from pathlib import Path

from runpodx.training.synthetic_ascii.image_to_ascii import AsciiRenderConfig
from runpodx.training.synthetic_ascii.public_image_mesh import load_public_image_mesh_config, load_public_image_records_from_mesh
from runpodx.training.synthetic_ascii.public_image_pipeline import run_public_image_ascii_pilot


def run_public_image_mesh_pilot(
    *,
    mesh_config_path: str | Path,
    output_dir: str | Path,
    width: int = 64,
) -> dict:
    mesh = load_public_image_mesh_config(mesh_config_path)
    records = load_public_image_records_from_mesh(mesh)
    summary = run_public_image_ascii_pilot(
        output_dir=output_dir,
        records=records,
        render_config=AsciiRenderConfig(width=width),
    )
    return {**summary, "mesh_name": mesh.name, "mesh_config_path": str(mesh_config_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-source public-image ASCII pilot.")
    parser.add_argument("mesh_config_path")
    parser.add_argument("--output-dir", default="artifacts/public_image_ascii/mesh_pilot_001")
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    summary = run_public_image_mesh_pilot(
        mesh_config_path=args.mesh_config_path,
        output_dir=args.output_dir,
        width=args.width,
    )
    print(summary)


if __name__ == "__main__":
    main()
