from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import read_project, write_json
from .reconstruct import reconstruct_project


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scanlan-worker")
    commands = root.add_subparsers(dest="command", required=True)

    reconstruct = commands.add_parser("reconstruct", help="Build a point cloud from all phases")
    reconstruct.add_argument("project", type=Path)
    reconstruct.add_argument("--engine", choices=["auto", "numpy", "open3d"], default="auto")
    reconstruct.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    reconstruct.add_argument(
        "--targets",
        default="point_cloud,textured_mesh",
        help="Comma-separated point_cloud,textured_mesh,gaussian_splat targets",
    )

    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        targets = tuple(value.strip() for value in arguments.targets.split(",") if value.strip())
        unknown = set(targets) - {"point_cloud", "textured_mesh", "gaussian_splat"}
        if unknown:
            raise ValueError(f"Unknown artifact targets: {', '.join(sorted(unknown))}")
        result = reconstruct_project(arguments.project, arguments.engine, arguments.device, targets)
        print(json.dumps(result))
        return 0
    except Exception as error:  # CLI boundary: preserve a useful project status before exiting.
        if arguments.command == "reconstruct":
            try:
                project = read_project(arguments.project)
                project["processingStatus"] = "failed"
                project["processingError"] = str(error)
                write_json(arguments.project / "project.json", project)
            except Exception:
                pass
        print(f"scanlan-worker: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
