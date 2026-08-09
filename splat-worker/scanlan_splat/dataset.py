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
    schema = int(dataset.get("schemaVersion", 0))
    if schema not in (3, 4):
        raise ValueError("ScanLan Gaussian training requires canonical dataset schema 3 or 4")
    frames = dataset.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ScanLan Gaussian training requires calibrated registered frames")
    metric = bool(dataset.get("metric", False))
    depth_frame_count = 0
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
        has_depth = bool(frame.get("depth"))
        has_mask = bool(frame.get("depthMask"))
        if has_depth != has_mask:
            raise ValueError(
                f"Canonical frame {index} must provide both depth and depthMask or neither"
            )
        if schema == 3 and metric and not has_depth:
            raise ValueError(f"Canonical metric frame {index} is missing registered depth")
        if bool(frame.get("metricAnchor", False)) and not has_depth:
            raise ValueError(f"Canonical frame {index} is a metric anchor without depth")
        if has_depth:
            depth_frame_count += 1
            for optional_field in ("depthConfidence", "generatedDepthMask"):
                if frame.get(optional_field):
                    relative = Path(str(frame[optional_field]))
                    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            f"Canonical frame {index} has an unsafe {optional_field} path"
                        )
                    if not (root / relative).is_file():
                        raise FileNotFoundError(
                            f"Canonical frame {index} is missing {optional_field}: {relative}"
                        )
        if schema == 4:
            confidence = float(frame.get("poseConfidence", 0.0))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Canonical frame {index} has invalid pose confidence")
        required_fields = ["image"]
        if has_depth:
            required_fields.extend(("depth", "depthMask"))
        for field in required_fields:
            relative = Path(str(frame.get(field, "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Canonical frame {index} has an unsafe {field} path")
            if not (root / relative).is_file():
                raise FileNotFoundError(
                    f"Canonical frame {index} is missing {field}: {relative}"
                )
    if metric and depth_frame_count == 0:
        raise ValueError("Canonical metric dataset has no registered depth anchors")
    initialization = Path(str(dataset.get("initialization", "")))
    if (
        not initialization.parts
        or initialization.is_absolute()
        or ".." in initialization.parts
        or not (root / initialization).is_file()
    ):
        raise FileNotFoundError("Gaussian sparse initialization is missing or unsafe")
    return root, dataset
