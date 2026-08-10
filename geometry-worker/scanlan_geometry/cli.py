from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scanlan_splat.da3 import da3_runtime_status, refine_da3_depth_request
from scanlan_splat.geometry_ipc import (
    run_da3_request,
    run_lingbot_map_request,
    run_mapanything_request,
)
from scanlan_splat.lingbot import lingbot_runtime_status
from scanlan_splat.lingbot_depth import (
    lingbot_depth_runtime_status,
    refine_depth_request,
)
from scanlan_splat.mapanything import (
    mapanything_runtime_status,
    refine_mapanything_depth_request,
)

from . import __version__


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scanlan-geometry")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer-lingbot-map")
    infer.add_argument("--request", type=Path, required=True)
    infer.add_argument("--progress", type=Path, required=True)
    mapanything = commands.add_parser("infer-mapanything")
    mapanything.add_argument("--request", type=Path, required=True)
    mapanything.add_argument("--progress", type=Path, required=True)
    da3 = commands.add_parser("infer-da3")
    da3.add_argument("--request", type=Path, required=True)
    da3.add_argument("--progress", type=Path, required=True)
    refine = commands.add_parser("refine-rgbd-depth")
    refine.add_argument("--request", type=Path, required=True)
    refine.add_argument("--progress", type=Path, required=True)
    refine_mapanything = commands.add_parser("refine-rgbd-depth-mapanything")
    refine_mapanything.add_argument("--request", type=Path, required=True)
    refine_mapanything.add_argument("--progress", type=Path, required=True)
    refine_da3 = commands.add_parser("refine-rgbd-depth-da3")
    refine_da3.add_argument("--request", type=Path, required=True)
    refine_da3.add_argument("--progress", type=Path, required=True)
    diagnostics = commands.add_parser("diagnostics")
    diagnostics.add_argument("--require-lingbot", action="store_true")
    diagnostics.add_argument("--require-lingbot-depth", action="store_true")
    diagnostics.add_argument("--require-flashinfer", action="store_true")
    diagnostics.add_argument("--require-mapanything", action="store_true")
    diagnostics.add_argument("--require-da3", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "infer-lingbot-map":
            run_lingbot_map_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "infer-mapanything":
            run_mapanything_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "infer-da3":
            run_da3_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "refine-rgbd-depth":
            refine_depth_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "refine-rgbd-depth-mapanything":
            refine_mapanything_depth_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "refine-rgbd-depth-da3":
            refine_da3_depth_request(arguments.request, arguments.progress)
            return 0
        if arguments.command == "diagnostics":
            lingbot_map = lingbot_runtime_status(
                allow_download=arguments.require_lingbot,
                validate_flashinfer=arguments.require_flashinfer,
            )
            lingbot_depth = lingbot_depth_runtime_status(
                verify_model=arguments.require_lingbot_depth,
                smoke_test=arguments.require_lingbot_depth,
            )
            mapanything = mapanything_runtime_status(
                verify_model=arguments.require_mapanything,
                smoke_test=arguments.require_mapanything,
            )
            da3 = da3_runtime_status(
                verify_model=arguments.require_da3,
                smoke_test=arguments.require_da3,
            )
            print(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "version": __version__,
                        "backends": ["lingbot-map", "lingbot-depth", "mapanything", "da3"],
                        "lingbotMap": lingbot_map,
                        "lingbotDepth": lingbot_depth,
                        "mapAnything": mapanything,
                        "da3": da3,
                    }
                )
            )
            return (
                2
                if arguments.require_lingbot
                and not lingbot_map["available"]
                or arguments.require_lingbot_depth
                and not lingbot_depth["available"]
                or arguments.require_flashinfer
                and not lingbot_map["flashinferValidated"]
                or arguments.require_mapanything
                and not mapanything["available"]
                or arguments.require_da3
                and not da3["available"]
                else 0
            )
        return 2
    except Exception as error:
        print(f"scanlan-geometry: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
