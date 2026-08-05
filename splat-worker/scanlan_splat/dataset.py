from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def resolve_dataset(path: Path) -> Path:
    """Resolve the reconstruction worker's immutable dataset pointer."""
    path = path.resolve()
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        resolved = (path.parent / pointer["path"]).resolve()
        if not resolved.is_relative_to(path.parent.resolve()):
            raise ValueError("Dataset pointer escapes the reconstruction cache")
        return resolved
    return path


def load_dataset(path: Path) -> tuple[Path, dict[str, Any]]:
    root = resolve_dataset(path)
    manifest = root / "dataset.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"RGB-D Gaussian dataset is missing: {manifest}")
    dataset = json.loads(manifest.read_text(encoding="utf-8"))
    if int(dataset.get("schemaVersion", 0)) != 3:
        raise ValueError("ScanLan 2DGS requires canonical dataset schema 3")
    if not dataset.get("metric", False):
        raise ValueError("ScanLan only trains Gaussian surfaces from metric RGB-D datasets")
    frames = dataset.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ScanLan 2DGS requires at least one calibrated RGB-D frame")
    for index, frame in enumerate(frames):
        intrinsics = frame.get("intrinsics") if isinstance(frame, dict) else None
        if (
            not isinstance(intrinsics, dict)
            or intrinsics.get("model") != "pinhole"
            or intrinsics.get("distortion") != []
        ):
            raise ValueError(
                f"Canonical frame {index} is not undistorted to the pinhole camera model"
            )
        try:
            width = int(intrinsics["width"])
            height = int(intrinsics["height"])
            calibration = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Canonical frame {index} has incomplete intrinsics") from error
        if (
            width <= 0
            or height <= 0
            or calibration[0] <= 0.0
            or calibration[1] <= 0.0
            or not all(math.isfinite(value) for value in calibration)
        ):
            raise ValueError(f"Canonical frame {index} has invalid intrinsics")
        pose = frame.get("worldFromRgbCamera")
        if (
            not isinstance(pose, list)
            or len(pose) != 16
            or not all(math.isfinite(float(value)) for value in pose)
        ):
            raise ValueError(f"Canonical frame {index} has an invalid camera pose")
        for field in ("image", "depth", "depthMask"):
            relative = Path(str(frame.get(field, "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Canonical frame {index} has an unsafe {field} path")
    return root, dataset
