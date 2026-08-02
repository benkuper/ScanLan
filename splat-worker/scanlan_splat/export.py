from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

SH_C0 = 0.28209479177387814


def export_3dgs_ply(
    path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(means)
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{index}" for index in range(45)],
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in names])
    vertices = np.zeros(count, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = means.astype(np.float32).T
    dc = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = dc.astype(np.float32).T
    vertices["opacity"] = opacity_logits.reshape(-1).astype(np.float32)
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales.astype(np.float32).T
    normalized = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)
    for index in range(4):
        vertices[f"rot_{index}"] = normalized[:, index].astype(np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment canonical ScanLan 3DGS export\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_splat_sidecars(
    output_root: Path,
    fingerprint: str,
    metric: bool,
    versions: dict[str, str],
    training: dict[str, Any],
) -> None:
    transform = {
        "schemaVersion": 1,
        "applyAtGameObject": True,
        "matrixStorage": "row-major",
        "projectFromSplat": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "note": "Non-uniform viewer transforms are intentionally not baked into Gaussian covariances.",
    }
    manifest = {
        "schemaVersion": 1,
        "sourceFingerprint": fingerprint,
        "trainer": "scanlan_splatfacto_depth" if training.get("usesDepth") else "scanlan_splatfacto_rgb",
        "versions": versions,
        "metric": metric,
        "units": "metres" if metric else "arbitrary",
        "coordinateConvention": {
            "handedness": "right",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "pose": "worldFromCamera",
            "matrixStorage": "row-major",
        },
        "properties": [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0..2", "f_rest_0..44",
            "opacity", "scale_0..2", "rot_0..3",
        ],
        "training": training,
    }
    (output_root / "room-splat.transform.json").write_text(
        json.dumps(transform, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "splat-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
