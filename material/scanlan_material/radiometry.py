from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageCms, ImageOps


RADIOMETRY_VERSION = "scanlan-linear-srgb-v1"


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    encoded = np.asarray(value, dtype=np.float32)
    if np.any(~np.isfinite(encoded)) or np.any(encoded < 0.0) or np.any(encoded > 1.0):
        raise ValueError("sRGB input must be finite and remain in [0, 1]")
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    linear = np.asarray(value, dtype=np.float32)
    if np.any(~np.isfinite(linear)) or np.any(linear < 0.0) or np.any(linear > 1.0):
        raise ValueError("bounded linear RGB must be finite and remain in [0, 1]")
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def to_canonical_srgb(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    transposed = ImageOps.exif_transpose(image)
    embedded = transposed.info.get("icc_profile")
    metadata: dict[str, Any] = {
        "sourceMode": transposed.mode,
        "embeddedIcc": bool(embedded),
        "assumedEncoding": None,
    }
    if embedded:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
            converted = ImageCms.profileToProfile(
                transposed,
                source_profile,
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            )
            metadata["conversion"] = "embedded-icc-to-srgb-relative-colorimetric"
        except (OSError, ValueError) as error:
            raise ValueError(f"embedded ICC profile could not be converted: {error}") from error
    else:
        # ScanLan's capture and media preparation contracts publish 8-bit RGB
        # JPEG/PNG without a profile as IEC sRGB. Unknown HDR transfer curves
        # are rejected below rather than being silently treated as sRGB.
        if transposed.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            raise ValueError(
                f"unsupported unprofiled image mode {transposed.mode}; convert HDR/wide-gamut input explicitly"
            )
        converted = transposed.convert("RGB")
        metadata["conversion"] = "declared-srgb"
        metadata["assumedEncoding"] = "IEC-61966-2-1-sRGB"
    converted.info["icc_profile"] = _srgb_profile_bytes()
    return converted, metadata


def load_linear_rgb(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as opened:
        canonical, metadata = to_canonical_srgb(opened)
        encoded = np.asarray(canonical, dtype=np.float32) / 255.0
    linear = srgb_to_linear(encoded)
    metadata.update(
        {
            "width": int(linear.shape[1]),
            "height": int(linear.shape[0]),
            "storage": "float32-linear-srgb",
        }
    )
    return linear, metadata


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _safe_image_paths(dataset_root: Path, dataset: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for index, frame in enumerate(dataset.get("frames", [])):
        relative = Path(str(frame.get("image", "")))
        if not relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"dataset frame {index} contains an unsafe image path")
        resolved = (dataset_root / relative).resolve(strict=True)
        if not resolved.is_relative_to(dataset_root.resolve()):
            raise ValueError(f"dataset frame {index} escapes the dataset root")
        paths.append(resolved)
    if not paths:
        raise ValueError("dataset contains no material preparation frames")
    return paths


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset_radiometry(
    dataset_root: Path,
    output_root: Path,
    *,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    dataset_path = dataset_root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    images = _safe_image_paths(dataset_root, dataset)
    selected = list(range(len(images))) if frame_indices is None else list(frame_indices)
    if not selected or any(index < 0 or index >= len(images) for index in selected):
        raise ValueError("radiometric frame selection is empty or outside the dataset")

    fingerprint = hashlib.sha256()
    fingerprint.update(RADIOMETRY_VERSION.encode("ascii"))
    fingerprint.update(str(dataset.get("fingerprint", "")).encode("utf-8"))
    for index in selected:
        fingerprint.update(str(index).encode("ascii"))
        fingerprint.update(_digest(images[index]).encode("ascii"))
    key = fingerprint.hexdigest()
    destination = output_root.resolve() / key
    manifest_path = destination / "radiometry.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_paths = [Path(str(frame.get("linearRgb", ""))) for frame in existing.get("frames", [])]
        cache_complete = len(cached_paths) == len(selected) and all(
            path.as_posix()
            and not path.is_absolute()
            and ".." not in path.parts
            and (destination / path).is_file()
            for path in cached_paths
        )
        if existing.get("fingerprint") == key and cache_complete:
            return existing

    staging = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "linear").mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for output_index, source_index in enumerate(selected):
            source = images[source_index]
            linear, color = load_linear_rgb(source)
            relative = Path("linear") / f"{output_index:06}.npy"
            with (staging / relative).open("wb") as handle:
                # IEEE float16 is lossless enough for bounded 8-bit source
                # values and halves offline material-stage I/O.
                np.save(handle, linear.astype(np.float16), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            records.append(
                {
                    "frameIndex": source_index,
                    "sourceImage": images[source_index].relative_to(dataset_root).as_posix(),
                    "sourceSha256": _digest(source),
                    "linearRgb": relative.as_posix(),
                    "width": color["width"],
                    "height": color["height"],
                    "color": color,
                }
            )
        manifest = {
            "schemaVersion": 1,
            "radiometryVersion": RADIOMETRY_VERSION,
            "fingerprint": key,
            "datasetFingerprint": dataset.get("fingerprint"),
            "workingColorSpace": "linear-sRGB-D65",
            "sourceTransfer": "IEC-61966-2-1-sRGB-after-ICC-conversion",
            "storage": "little-endian-npy-float16-RGB",
            "frames": records,
        }
        _atomic_json(staging / "radiometry.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
