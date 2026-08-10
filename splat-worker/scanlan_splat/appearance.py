from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GAUSSIAN_MATERIAL_SCHEMA = 1
GAUSSIAN_MATERIAL_CONTRACT = "scanlan-gaussian-material-v1"
MATERIAL_ARRAYS = (
    "material_albedo_linear",
    "material_roughness",
    "material_metallic",
    "material_transmission",
    "material_emission_linear",
    "material_confidence",
)


@dataclass(frozen=True)
class GaussianMaterialSeeds:
    albedo_linear: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    transmission: np.ndarray
    emission_linear: np.ndarray
    confidence: np.ndarray

    def validated(self, count: int) -> "GaussianMaterialSeeds":
        albedo = np.asarray(self.albedo_linear, dtype=np.float32)
        emission = np.asarray(self.emission_linear, dtype=np.float32)
        if albedo.shape != (count, 3) or emission.shape != (count, 3):
            raise ValueError("Gaussian material colors must be Nx3")
        if not np.isfinite(albedo).all() or np.any(albedo < 0.0) or np.any(albedo > 1.0):
            raise ValueError("Gaussian material albedo must be finite linear RGB in [0, 1]")
        if not np.isfinite(emission).all() or np.any(emission < 0.0):
            raise ValueError("Gaussian material emission must be finite non-negative linear RGB")
        scalars: dict[str, np.ndarray] = {}
        for name in ("roughness", "metallic", "transmission", "confidence"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"Gaussian material {name} must be finite N")
            if np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"Gaussian material {name} must remain in [0, 1]")
            scalars[name] = value
        return GaussianMaterialSeeds(
            albedo,
            scalars["roughness"],
            scalars["metallic"],
            scalars["transmission"],
            emission,
            scalars["confidence"],
        )


def resolve_gaussian_material_seeds(
    root: Path,
    dataset: dict[str, Any],
    count: int,
) -> GaussianMaterialSeeds | None:
    """Load a declared intrinsic prior without guessing from RGB or opacity."""

    parameters_path = dataset.get("initializationParameters")
    path = root / str(parameters_path or "")
    available: set[str] = set()
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            available = set(archive.files).intersection(MATERIAL_ARRAYS)
    contract = dataset.get("gaussianMaterial")
    if contract is None:
        if available:
            raise ValueError("Gaussian material arrays require a declared gaussianMaterial contract")
        return None
    if not isinstance(contract, dict) or int(contract.get("schemaVersion", 0)) != GAUSSIAN_MATERIAL_SCHEMA:
        raise ValueError("Unsupported Gaussian material contract schema")
    if contract.get("colorSpace") != "linear-srgb":
        raise ValueError("Gaussian material contract must declare linear-srgb")
    declared_parameters = str(contract.get("parameters", ""))
    if not declared_parameters or declared_parameters != str(parameters_path or ""):
        raise ValueError("Gaussian material contract must use the initialization parameter sidecar")
    relative = Path(declared_parameters)
    if relative.is_absolute() or ".." in relative.parts or not path.is_file():
        raise ValueError("Gaussian material parameter sidecar is missing or unsafe")
    missing = set(MATERIAL_ARRAYS).difference(available)
    if missing:
        raise ValueError(f"Gaussian material sidecar is incomplete: {', '.join(sorted(missing))}")
    with np.load(path, allow_pickle=False) as archive:
        result = GaussianMaterialSeeds(
            np.asarray(archive["material_albedo_linear"], dtype=np.float32),
            np.asarray(archive["material_roughness"], dtype=np.float32),
            np.asarray(archive["material_metallic"], dtype=np.float32),
            np.asarray(archive["material_transmission"], dtype=np.float32),
            np.asarray(archive["material_emission_linear"], dtype=np.float32),
            np.asarray(archive["material_confidence"], dtype=np.float32),
        )
    return result.validated(count)


def gaussian_material_manifest(parameters: str) -> dict[str, Any]:
    relative = Path(parameters)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Gaussian material parameter path must stay inside the dataset")
    return {
        "schemaVersion": GAUSSIAN_MATERIAL_SCHEMA,
        "contract": GAUSSIAN_MATERIAL_CONTRACT,
        "parameters": parameters,
        "colorSpace": "linear-srgb",
        "components": ["diffuse", "view-dependent", "emissive", "transmissive"],
    }


def srgb_to_linear_tensor(value: Any) -> Any:
    import torch

    return torch.where(
        value <= 0.04045,
        value / 12.92,
        torch.pow((value + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb_tensor(value: Any) -> Any:
    import torch

    value = value.clamp_min(0.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * torch.pow(value, 1.0 / 2.4) - 0.055,
    )


def _decoded_material(parameters: Any) -> dict[str, Any]:
    import torch

    return {
        "emission": torch.exp(parameters["emission_log"].clamp(-16.0, 4.0)),
        "transmission": torch.sigmoid(parameters["transmission_logits"]),
        "roughness": torch.sigmoid(parameters["roughness_logits"]),
        "metallic": torch.sigmoid(parameters["metallic_logits"]),
        "confidence": parameters["material_confidence"].clamp(0.0, 1.0),
        "emission_anchor": torch.exp(parameters["emission_anchor_log"].clamp(-16.0, 4.0)),
        "transmission_anchor": torch.sigmoid(parameters["transmission_anchor_logits"]),
    }


def compose_material_render_state(
    parameters: Any,
    material_aware: bool,
    sh_c0: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Compose radiance while retaining geometric and optical opacity separately.

    Smooth, metallic material confidence stops photometric gradients from
    leaking into transmission. The declared transmission prior still anchors
    that branch through the explicit material regularizer.
    """

    import torch

    if not material_aware:
        coefficients = torch.cat(
            (parameters["diffuse_sh0"], parameters["view_shN"]), dim=1
        )
        return coefficients, torch.sigmoid(parameters["opacities"]), {}
    decoded = _decoded_material(parameters)
    specular_gate = (
        decoded["confidence"]
        * (1.0 - decoded["roughness"])
        * (0.5 + 0.5 * decoded["metallic"])
    ).clamp(0.0, 1.0)
    transmission_for_rgb = (
        decoded["transmission"] * (1.0 - specular_gate)
        + decoded["transmission"].detach() * specular_gate
    )
    diffuse_and_emission = (
        parameters["diffuse_sh0"] + decoded["emission"][:, None, :] / sh_c0
    )
    coefficients = torch.cat(
        (diffuse_and_emission, parameters["view_shN"]), dim=1
    )
    optical_opacity = torch.sigmoid(parameters["opacities"]) * (
        1.0 - transmission_for_rgb
    )
    decoded["specular_gate"] = specular_gate
    decoded["optical_opacity"] = optical_opacity
    return coefficients, optical_opacity, decoded


def material_regularization(parameters: Any, material_aware: bool) -> tuple[Any, dict[str, Any]]:
    import torch

    if not material_aware:
        zero = parameters["means"].sum() * 0.0
        return zero, {"prior": zero, "unsupported": zero, "viewDependent": zero}
    decoded = _decoded_material(parameters)
    confidence = decoded["confidence"]
    prior = (
        confidence
        * (
            (decoded["transmission"] - decoded["transmission_anchor"]).square()
            + torch.mean(
                (decoded["emission"] - decoded["emission_anchor"]).square(), dim=1
            )
        )
    ).mean()
    unsupported = (
        (1.0 - confidence)
        * (
            decoded["transmission"].square()
            + torch.mean(decoded["emission"].square(), dim=1)
        )
    ).mean()
    view_dependent = (
        confidence[:, None, None]
        * decoded["roughness"][:, None, None]
        * parameters["view_shN"].square()
    ).mean()
    total = 1e-3 * prior + 2e-3 * unsupported + 1e-5 * view_dependent
    return total, {
        "prior": prior,
        "unsupported": unsupported,
        "viewDependent": view_dependent,
    }
