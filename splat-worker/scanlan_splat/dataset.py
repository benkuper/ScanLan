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
        raise FileNotFoundError(f"Gaussian dataset is missing: {manifest}")
    dataset = json.loads(manifest.read_text(encoding="utf-8"))
    if int(dataset.get("schemaVersion", 0)) != 3:
        raise ValueError("ScanLan Gaussian training requires canonical dataset schema 3")
    frames = dataset.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ScanLan Gaussian training requires calibrated registered frames")
    metric = bool(dataset.get("metric", False))
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
        required_fields = ["image"]
        if metric:
            required_fields.extend(("depth", "depthMask"))
        elif bool(frame.get("depth")) != bool(frame.get("depthMask")):
            raise ValueError(
                f"Canonical frame {index} must provide both depth and depthMask or neither"
            )
        for field in required_fields:
            relative = Path(str(frame.get(field, "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Canonical frame {index} has an unsafe {field} path")
            if not (root / relative).is_file():
                raise FileNotFoundError(
                    f"Canonical frame {index} is missing {field}: {relative}"
                )
    initialization = Path(str(dataset.get("initialization", "")))
    if (
        not initialization.parts
        or initialization.is_absolute()
        or ".." in initialization.parts
        or not (root / initialization).is_file()
    ):
        raise FileNotFoundError("Gaussian sparse initialization is missing or unsafe")
    return root, dataset
