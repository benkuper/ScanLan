from __future__ import annotations

import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SH_C0 = 0.28209479177387814


def export_material_gaussians(
    path: Path,
    *,
    diffuse_linear: np.ndarray,
    view_sh_linear: np.ndarray,
    emission_linear: np.ndarray,
    transmission: np.ndarray,
    roughness: np.ndarray,
    metallic: np.ndarray,
    confidence: np.ndarray,
    geometric_opacity: np.ndarray,
    optical_opacity: np.ndarray,
) -> None:
    """Atomically publish the lossless P17 appearance decomposition sidecar."""

    count = len(diffuse_linear)
    vectors = {
        "diffuse_linear": (diffuse_linear, (count, 3)),
        "emission_linear": (emission_linear, (count, 3)),
    }
    scalars = {
        "transmission": transmission,
        "roughness": roughness,
        "metallic": metallic,
        "confidence": confidence,
        "geometric_opacity": geometric_opacity,
        "optical_opacity": optical_opacity,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (value, shape) in vectors.items():
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError(f"Material Gaussian {name} must be finite {shape}")
        arrays[name] = array
    view = np.asarray(view_sh_linear, dtype=np.float32)
    if view.ndim != 3 or view.shape[0] != count or view.shape[2] != 3 or not np.isfinite(view).all():
        raise ValueError("Material Gaussian view-dependent SH must be NxKx3")
    arrays["view_sh_linear"] = view
    for name, value in scalars.items():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError(f"Material Gaussian {name} must be finite N")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(f"Material Gaussian {name} must remain in [0, 1]")
        arrays[name] = array
    arrays.update(
        contract=np.asarray("scanlan-gaussian-material-v1"),
        color_space=np.asarray("linear-srgb"),
        aligned_with=np.asarray("room-splat.ply vertex order"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_splat_preview(
    path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    quaternions: np.ndarray,
    limit: int | None = None,
) -> None:
    """Write the compact 32-byte/splat format used by the realtime viewer.

    The canonical PLY remains the lossless interchange artifact. ScanLan's
    trainer only produces degree-zero colors, so keeping the PLY's 45 empty
    higher-order SH fields in memory while previewing is unnecessary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(means)
    if limit is not None and limit > 0 and count > limit:
        indices = np.linspace(0, count - 1, limit, dtype=np.int64)
        means = np.asarray(means)[indices]
        colors = np.asarray(colors)[indices]
        opacity_logits = np.asarray(opacity_logits)[indices]
        log_scales = np.asarray(log_scales)[indices]
        quaternions = np.asarray(quaternions)[indices]
        count = limit
    payload = bytearray(count * 32)
    float_view = np.ndarray((count, 8), dtype="<f4", buffer=payload)
    byte_view = np.ndarray((count, 32), dtype=np.uint8, buffer=payload)

    float_view[:, 0:3] = np.asarray(means, dtype=np.float32)
    float_view[:, 3:6] = np.exp(np.asarray(log_scales, dtype=np.float32))
    byte_view[:, 24:27] = np.rint(
        np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    logits = np.clip(np.asarray(opacity_logits, dtype=np.float32).reshape(-1), -80.0, 80.0)
    byte_view[:, 27] = np.rint((1.0 / (1.0 + np.exp(-logits))) * 255.0).astype(np.uint8)

    normalized = np.asarray(quaternions, dtype=np.float32)
    normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-8)
    byte_view[:, 28:32] = np.rint(np.clip(normalized * 128.0 + 128.0, 0.0, 255.0)).astype(np.uint8)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.025)
    finally:
        temporary.unlink(missing_ok=True)


def export_3dgs_ply(
    path: Path,
    means: np.ndarray,
    colors: np.ndarray,
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    quaternions: np.ndarray,
    *,
    sh_coefficients: np.ndarray | None = None,
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
    if sh_coefficients is None:
        dc = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0
        rest = np.zeros((count, 45), dtype=np.float32)
    else:
        sh = np.asarray(sh_coefficients, dtype=np.float32)
        if sh.ndim != 3 or sh.shape[0] != count or sh.shape[2] != 3 or sh.shape[1] < 1:
            raise ValueError("Spherical-harmonic coefficients must be N×K×3")
        dc = sh[:, 0, :]
        rest = np.zeros((count, 45), dtype=np.float32)
        coefficient_count = min(15, sh.shape[1] - 1)
        if coefficient_count:
            channel_coefficients = rest.reshape(count, 3, 15)
            channel_coefficients[:, :, :coefficient_count] = np.transpose(
                sh[:, 1 : coefficient_count + 1, :], (0, 2, 1)
            )
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = dc.astype(np.float32).T
    for index in range(45):
        vertices[f"f_rest_{index}"] = rest[:, index]
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
    material: dict[str, Any] | None = None,
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
        "preview": {
            "path": "room-splat.preview.splat",
            "format": "splat",
            "bytesPerGaussian": 32,
        },
        "refinedCameras": {
            "path": "room-splat-cameras.json",
            "pose": "worldFromCamera",
            "matrixStorage": "row-major",
        },
        "training": training,
    }
    if material is not None:
        manifest["material"] = material
    (output_root / "room-splat.transform.json").write_text(
        json.dumps(transform, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "splat-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
