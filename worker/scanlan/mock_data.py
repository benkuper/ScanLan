from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .io import write_json


def create_mock_project(root: Path, phase_count: int = 2, frame_count: int = 10) -> Path:
    root = root.resolve()
    (root / "phases").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    project = {
        "schemaVersion": 3,
        "id": str(uuid4()),
        "name": "Synthetic phased scan",
        "path": str(root),
        "createdAt": now,
        "phases": [],
        "artifacts": {"pointCloud": None, "texturedMesh": None, "gaussianSplat": None},
        "activeJob": None,
        "settings": {
            "captureFps": 10,
            "maxDepthM": 4.2,
            "voxelSizeMm": 30,
            "sensorKind": "azure_kinect",
            "sensorId": "mock",
            "sensorConnection": "usb",
            "sensorAddress": "",
            "useImu": False,
            "depthFieldOfView": "narrow",
            "depthBinned": False,
            "rgbJpegQuality": 92,
            "maxRgbDimension": 0,
            "liveReconstruction": "mesh",
        },
        "processingStatus": "idle",
    }

    width, height = 48, 36
    for phase_index in range(phase_count):
        phase_id = str(uuid4())
        phase_root = root / "phases" / phase_id
        depth_root = phase_root / "depth"
        color_root = phase_root / "color"
        depth_root.mkdir(parents=True)
        color_root.mkdir(parents=True)
        phase_name = f"Synthetic phase {phase_index + 1}"
        phase = {
            "schemaVersion": 3,
            "id": phase_id,
            "name": phase_name,
            "createdAt": now,
            "frameCount": frame_count,
            "durationSeconds": max(1, frame_count // 10),
            "frameFormat": "depth=u16le,color=rgb8,aligned=true",
            "poseSource": "mock_ground_truth",
            "sensor": {
                "kind": "azure_kinect",
                "name": "Synthetic RGB-D camera",
                "connection": "usb",
                "serial": "mock",
                "address": "",
            },
            "camera": {
                "width": width,
                "height": height,
                "fx": 44.0,
                "fy": 44.0,
                "cx": (width - 1) / 2,
                "cy": (height - 1) / 2,
                "depth_scale": 1000.0,
                "max_depth_m": 4.2,
            },
            "rgbCamera": {
                "width": width,
                "height": height,
                "fx": 44.0,
                "fy": 44.0,
                "cx": (width - 1) / 2,
                "cy": (height - 1) / 2,
                "model": "pinhole",
                "distortion": [],
            },
            "rgbFromDepth": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "sourceRgb": {"format": "aligned-rgb8", "quality": 100, "nativeResolution": False, "droppedFrames": 0},
        }
        write_json(phase_root / "phase.json", phase)

        matrix_keys = [f"m{row}{column}" for row in range(4) for column in range(4)]
        with (phase_root / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "source_sequence",
                    "timestamp_us",
                    "depth_path",
                    "color_path",
                    *matrix_keys,
                ],
            )
            writer.writeheader()
            for frame_index in range(frame_count):
                yy, xx = np.mgrid[0:height, 0:width]
                depth = 2100 + 120 * np.sin(xx / 6 + frame_index / 8) + 50 * np.cos(yy / 5)
                depth.astype("<u2").tofile(depth_root / f"{frame_index:06}.u16")
                color = np.stack(
                    (
                        np.broadcast_to((xx * 255 / width).astype(np.uint8), (height, width)),
                        np.broadcast_to((yy * 220 / height).astype(np.uint8), (height, width)),
                        np.full((height, width), 130 + phase_index * 30, dtype=np.uint8),
                    ),
                    axis=-1,
                )
                color.tofile(color_root / f"{frame_index:06}.rgb")
                pose = np.eye(4)
                pose[0, 3] = phase_index * 0.35 + frame_index * 0.01
                pose[2, 3] = phase_index * 0.06
                row = {
                    "index": frame_index,
                    "source_sequence": frame_index,
                    "timestamp_us": frame_index * 100_000,
                    "depth_path": f"depth/{frame_index:06}.u16",
                    "color_path": f"color/{frame_index:06}.rgb",
                }
                row.update(dict(zip(matrix_keys, pose.reshape(-1), strict=True)))
                writer.writerow(row)

        project["phases"].append(
            {
                "id": phase_id,
                "name": phase_name,
                "createdAt": now,
                "durationSeconds": phase["durationSeconds"],
                "frameCount": frame_count,
                "status": "complete",
                "overlapHint": "Known mock overlap" if phase_index else "Reference phase",
            }
        )
    write_json(root / "project.json", project)
    return root
