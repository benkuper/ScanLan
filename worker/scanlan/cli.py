from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .io import read_project, write_json
from .mesh_repair import settings_from_project
from .reconstruct import reconstruct_project


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scanlan-worker")
    commands = root.add_subparsers(dest="command", required=True)

    reconstruct = commands.add_parser("reconstruct", help="Build a point cloud from all phases")
    reconstruct.add_argument("project", type=Path)
    reconstruct.add_argument("--engine", choices=["auto", "numpy", "open3d"], default="auto")
    reconstruct.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    reconstruct.add_argument(
        "--depth-refinement",
        choices=["off", "lingbot", "mapanything", "da3"],
        default="off",
    )
    reconstruct.add_argument("--depth-refiner", type=Path, default=None)
    reconstruct.add_argument(
        "--targets",
        default="point_cloud,textured_mesh",
        help="Comma-separated point_cloud,textured_mesh,gaussian_splat targets",
    )
    reconstruct.add_argument("--mesh-repair", choices=["on", "off"], default=None)
    reconstruct.add_argument(
        "--mesh-repair-profile",
        choices=["faithful", "architectural", "natural"],
        default=None,
    )
    reconstruct.add_argument(
        "--fill-inferred-holes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    reconstruct.add_argument(
        "--produce-watertight-copy",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    reconstruct.add_argument(
        "--mesh-repair-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    realtime = commands.add_parser(
        "realtime",
        help="Consume the ScanLan RGB-D stream on stdin and emit engine messages on stdout",
    )
    realtime.add_argument("--mode", choices=["points", "mesh"], default="mesh")
    realtime.add_argument(
        "--voxel-size", type=float, default=0.01, help="TSDF voxel size in metres"
    )
    realtime.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    realtime.add_argument(
        "--live-map-mib",
        type=int,
        default=1024,
        help="Hard memory budget for the active sparse live submap",
    )
    realtime.add_argument(
        "--session", type=Path, required=True, help="Capture directory for the tracking journal"
    )

    replay = commands.add_parser(
        "replay", help="Emit a recorded RGB-D capture using the live stream protocol"
    )
    replay.add_argument("capture", type=Path)

    benchmark_live = commands.add_parser(
        "benchmark-live",
        help="Replay a capture through the realtime engine and report latency and memory",
    )
    benchmark_live.add_argument("capture", type=Path)
    benchmark_live.add_argument("--mode", choices=["points", "mesh"], default="mesh")
    benchmark_live.add_argument("--voxel-size", type=float, default=0.01)
    benchmark_live.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    benchmark_live.add_argument("--live-map-mib", type=int, default=1024)
    benchmark_live.add_argument("--session", type=Path, default=None)
    benchmark_live.add_argument("--report", type=Path, default=None)
    benchmark_live.add_argument(
        "--realtime-pacing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay using capture timestamps (disable for an overload stress run)",
    )

    localize = commands.add_parser(
        "localize-photos",
        help="Pose high-resolution photos against an existing RGB-D reconstruction",
    )
    localize.add_argument("project", type=Path)
    localize.add_argument("photos", type=Path, nargs="+")
    localize_media = commands.add_parser(
        "localize-media",
        help="Pose decoded photo/video observations against an RGB-D reconstruction",
    )
    localize_media.add_argument("project", type=Path)
    localize_media.add_argument("manifest", type=Path)

    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command in {"realtime", "replay"}:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
                msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        if arguments.command == "realtime":
            from .realtime import run_realtime_engine

            run_realtime_engine(
                sys.stdin.buffer,
                sys.stdout.buffer,
                mode=arguments.mode,
                voxel_size_m=arguments.voxel_size,
                requested_device=arguments.device,
                session_root=arguments.session,
                live_map_mib=arguments.live_map_mib,
            )
            return 0
        if arguments.command == "replay":
            from .replay import replay_archive

            replay_archive(arguments.capture, sys.stdout.buffer)
            return 0
        if arguments.command == "benchmark-live":
            from .live_benchmark import benchmark_live_capture

            result = benchmark_live_capture(
                arguments.capture,
                mode=arguments.mode,
                voxel_size_m=arguments.voxel_size,
                live_map_mib=arguments.live_map_mib,
                device=arguments.device,
                paced=arguments.realtime_pacing,
                session_root=arguments.session,
            )
            if arguments.report is not None:
                write_json(arguments.report, result)
            print(json.dumps(result, indent=2))
            return 0
        if arguments.command in {"localize-photos", "localize-media"}:
            from .supplemental import localize_supplemental_photos

            if arguments.command == "localize-media":
                manifest_path = arguments.manifest.resolve(strict=True)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if int(manifest.get("schemaVersion", 0)) != 1:
                    raise ValueError("Media observation manifest must use schema 1")
                photos = []
                observation_metadata = {}
                for frame in manifest.get("frames", []):
                    relative = Path(str(frame.get("image", "")))
                    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Media observation manifest contains an unsafe image path")
                    photo = (manifest_path.parent / relative).resolve(strict=True)
                    if not photo.is_relative_to(manifest_path.parent):
                        raise ValueError("Media observation image escapes its immutable cache")
                    photos.append(photo)
                    observation_metadata[photo] = frame
            else:
                photos = arguments.photos
                observation_metadata = None
            result = localize_supplemental_photos(
                arguments.project,
                photos,
                observation_metadata,
            )
        else:
            targets = tuple(value.strip() for value in arguments.targets.split(",") if value.strip())
            unknown = set(targets) - {
                "point_cloud",
                "textured_mesh",
                "gaussian_splat",
                "localization_map",
            }
            if unknown:
                raise ValueError(f"Unknown artifact targets: {', '.join(sorted(unknown))}")
            project = read_project(arguments.project)
            repair_settings = settings_from_project(project)
            overrides = {
                "enabled": (
                    arguments.mesh_repair == "on"
                    if arguments.mesh_repair is not None
                    else repair_settings.enabled
                ),
                "profile": arguments.mesh_repair_profile or repair_settings.profile,
                "fill_inferred_holes": (
                    arguments.fill_inferred_holes
                    if arguments.fill_inferred_holes is not None
                    else repair_settings.fill_inferred_holes
                ),
                "produce_watertight_copy": (
                    arguments.produce_watertight_copy
                    if arguments.produce_watertight_copy is not None
                    else repair_settings.produce_watertight_copy
                ),
                "allow_unrepaired_fallback": (
                    arguments.mesh_repair_fallback
                    if arguments.mesh_repair_fallback is not None
                    else repair_settings.allow_unrepaired_fallback
                ),
            }
            repair_settings = replace(repair_settings, **overrides)
            result = reconstruct_project(
                arguments.project,
                arguments.engine,
                arguments.device,
                targets,
                repair_settings,
                arguments.depth_refinement,
                arguments.depth_refiner,
            )
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
        elif arguments.command in {"localize-photos", "localize-media"}:
            try:
                from .supplemental import write_localization_progress

                progress_path = (
                    arguments.project
                    / "outputs"
                    / "photo-localization-progress.json"
                )
                previous = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file()
                    else {}
                )
                write_localization_progress(
                    arguments.project,
                    status="failed",
                    stage="failed",
                    detail=str(error),
                    progress=float(previous.get("progress", 0.0)),
                    processed_photos=int(previous.get("processedPhotos", 0)),
                    total_photos=(
                        len(arguments.photos)
                        if arguments.command == "localize-photos"
                        else int(previous.get("totalPhotos", 0))
                    ),
                    localized_photos=int(previous.get("localizedPhotos", 0)),
                    failed_photos=int(previous.get("failedPhotos", 0)),
                )
            except Exception:
                pass
        print(f"scanlan-worker: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
