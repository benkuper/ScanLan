from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTRACT_VERSION = "scanlan-material-v1"

# Material identity and optical failure mode are deliberately independent.
# A polished metal is both METAL and HIGH_SPECULAR; an emissive display can
# also be GLASS_OR_TRANSMISSIVE. Downstream geometry must not infer one from
# the other.
MATERIAL_CLASSES = (
    "unknown",
    "opaque_dielectric",
    "metal",
    "emissive",
    "thin_or_fibrous",
    "dynamic",
    "sky",
)
OPTICAL_RISKS = (
    "glass_or_transmissive",
    "mirror",
    "high_specular",
    "emissive",
    "thin_geometry",
    "dynamic",
    "sky",
)


def _probability_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must remain in [0, 1]")
    return array


@dataclass(frozen=True)
class MaterialPrediction:
    """One source-aligned material observation.

    Class probabilities are mutually exclusive. Optical risks are independent
    probabilities because failure modes overlap. PBR fields are optional until
    P16, but their ranges and color semantics are frozen here.
    """

    class_probabilities: np.ndarray
    optical_risk_probabilities: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    albedo_linear: np.ndarray | None = None
    roughness: np.ndarray | None = None
    metallic: np.ndarray | None = None
    transmission: np.ndarray | None = None
    normal_camera: np.ndarray | None = None
    emission_linear: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def validated(self) -> "MaterialPrediction":
        classes = np.asarray(self.class_probabilities, dtype=np.float32)
        if classes.ndim != 3 or classes.shape[2] != len(MATERIAL_CLASSES):
            raise ValueError(
                f"class probabilities must be HxWx{len(MATERIAL_CLASSES)}"
            )
        height, width = classes.shape[:2]
        classes = _probability_array("class probabilities", classes, classes.shape)
        risks = _probability_array(
            "optical-risk probabilities",
            self.optical_risk_probabilities,
            (height, width, len(OPTICAL_RISKS)),
        )
        valid = np.asarray(self.valid_mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("valid mask must be source-aligned HxW")
        confidence = _probability_array(
            "confidence", self.confidence, (height, width)
        )
        probability_sum = np.sum(classes, axis=2)
        if np.any(valid & (np.abs(probability_sum - 1.0) > 2e-3)):
            raise ValueError("valid class probabilities must sum to one")
        if np.any(~valid & ((confidence > 0.0) | (np.sum(risks, axis=2) > 0.0))):
            raise ValueError("invalid pixels must have zero confidence and optical risk")

        scalar_fields: dict[str, np.ndarray | None] = {
            "roughness": self.roughness,
            "metallic": self.metallic,
            "transmission": self.transmission,
        }
        checked_scalars: dict[str, np.ndarray | None] = {}
        for name, value in scalar_fields.items():
            checked_scalars[name] = (
                None
                if value is None
                else _probability_array(name, value, (height, width))
            )

        color_fields: dict[str, np.ndarray | None] = {
            "albedo_linear": self.albedo_linear,
            "emission_linear": self.emission_linear,
        }
        checked_colors: dict[str, np.ndarray | None] = {}
        for name, value in color_fields.items():
            if value is None:
                checked_colors[name] = None
                continue
            color = np.asarray(value, dtype=np.float32)
            if color.shape != (height, width, 3) or not np.isfinite(color).all():
                raise ValueError(f"{name} must be finite source-aligned linear RGB")
            if np.any(color < 0.0):
                raise ValueError(f"{name} cannot contain negative radiance")
            if name == "albedo_linear" and np.any(color > 1.0):
                raise ValueError("linear albedo must remain in [0, 1]")
            checked_colors[name] = color

        normal = None
        if self.normal_camera is not None:
            normal = np.asarray(self.normal_camera, dtype=np.float32)
            if normal.shape != (height, width, 3) or not np.isfinite(normal).all():
                raise ValueError("camera normals must be finite source-aligned XYZ")
            lengths = np.linalg.norm(normal, axis=2)
            if np.any(valid & (np.abs(lengths - 1.0) > 2e-2)):
                raise ValueError("valid camera normals must be unit length")

        return MaterialPrediction(
            classes,
            risks,
            valid,
            confidence,
            checked_colors["albedo_linear"],
            checked_scalars["roughness"],
            checked_scalars["metallic"],
            checked_scalars["transmission"],
            normal,
            checked_colors["emission_linear"],
            dict(self.metadata or {}),
        )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def write_prediction(path: Path, prediction: MaterialPrediction) -> None:
    checked = prediction.validated()
    arrays: dict[str, np.ndarray] = {
        "class_probabilities": checked.class_probabilities.astype(np.float16),
        "optical_risk_probabilities": checked.optical_risk_probabilities.astype(np.float16),
        "valid_mask": checked.valid_mask.astype(np.uint8),
        "confidence": checked.confidence.astype(np.float16),
        "contract": np.asarray(CONTRACT_VERSION),
        "metadata_json": np.asarray(
            json.dumps(dict(checked.metadata or {}), sort_keys=True, separators=(",", ":"))
        ),
    }
    for name in (
        "albedo_linear",
        "roughness",
        "metallic",
        "transmission",
        "normal_camera",
        "emission_linear",
    ):
        value = getattr(checked, name)
        if value is not None:
            arrays[name] = value.astype(np.float16)
    _atomic_npz(path, arrays)


def read_prediction(path: Path) -> MaterialPrediction:
    with np.load(path, allow_pickle=False) as archive:
        contract = str(archive["contract"].item())
        if contract != CONTRACT_VERSION:
            raise ValueError(f"unsupported material contract: {contract}")
        optional = {
            name: np.asarray(archive[name], dtype=np.float32) if name in archive else None
            for name in (
                "albedo_linear",
                "roughness",
                "metallic",
                "transmission",
                "normal_camera",
                "emission_linear",
            )
        }
        result = MaterialPrediction(
            np.asarray(archive["class_probabilities"], dtype=np.float32),
            np.asarray(archive["optical_risk_probabilities"], dtype=np.float32),
            np.asarray(archive["valid_mask"], dtype=bool),
            np.asarray(archive["confidence"], dtype=np.float32),
            optional["albedo_linear"],
            optional["roughness"],
            optional["metallic"],
            optional["transmission"],
            optional["normal_camera"],
            optional["emission_linear"],
            json.loads(str(archive["metadata_json"].item())),
        )
    return result.validated()
