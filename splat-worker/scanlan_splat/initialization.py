from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


INITIALIZATION_CONTRACT_SCHEMA = 1


class InitializationKind(StrEnum):
    SPARSE_SFM = "sparse_sfm"
    DENSE_SURFACE = "dense_surface"
    DIRECT_GAUSSIAN = "direct_gaussian"


class GaussianRepresentation(StrEnum):
    VOLUMETRIC_3D = "volumetric_3d"
    SURFACE_DISCS_2D = "surface_discs_2d"
    PREDICTED_ANISOTROPIC_3D = "predicted_anisotropic_3d"


@dataclass(frozen=True)
class InitializationContract:
    kind: InitializationKind
    representation: GaussianRepresentation
    parameters_path: str | None
    adaptive_densification: bool
    source: str

    @property
    def is_dense(self) -> bool:
        return self.kind is not InitializationKind.SPARSE_SFM

    @property
    def is_direct(self) -> bool:
        return self.kind is InitializationKind.DIRECT_GAUSSIAN

    @property
    def uses_2dgs(self) -> bool:
        return self.representation is GaussianRepresentation.SURFACE_DISCS_2D


def _legacy_contract(dataset: dict[str, Any]) -> InitializationContract:
    parameters_path = dataset.get("initializationParameters")
    direct = bool(dataset.get("directGaussianPrior", False))
    dense = bool(dataset.get("denseGeometryPrior", False)) or bool(
        dataset.get("metric") and parameters_path
    )
    if direct:
        return InitializationContract(
            InitializationKind.DIRECT_GAUSSIAN,
            GaussianRepresentation.PREDICTED_ANISOTROPIC_3D,
            str(parameters_path) if parameters_path else None,
            True,
            "legacy_inferred",
        )
    if dense:
        return InitializationContract(
            InitializationKind.DENSE_SURFACE,
            (
                GaussianRepresentation.SURFACE_DISCS_2D
                if dataset.get("metric")
                else GaussianRepresentation.VOLUMETRIC_3D
            ),
            str(parameters_path) if parameters_path else None,
            not bool(dataset.get("metric")),
            "legacy_inferred",
        )
    return InitializationContract(
        InitializationKind.SPARSE_SFM,
        GaussianRepresentation.VOLUMETRIC_3D,
        None,
        True,
        "legacy_inferred",
    )


def resolve_initialization_contract(dataset: dict[str, Any]) -> InitializationContract:
    """Resolve and validate the versioned Gaussian initialization policy.

    Existing immutable datasets remain readable through an explicit legacy
    inference path. New producers publish the contract so a trainer never has
    to guess whether confidence is geometry evidence or renderer opacity.
    """
    value = dataset.get("gaussianInitialization")
    if value is None:
        return _legacy_contract(dataset)
    if not isinstance(value, dict):
        raise ValueError("Gaussian initialization contract must be an object")
    if int(value.get("schemaVersion", 0)) != INITIALIZATION_CONTRACT_SCHEMA:
        raise ValueError("Unsupported Gaussian initialization contract schema")
    try:
        kind = InitializationKind(str(value["kind"]))
        representation = GaussianRepresentation(str(value["representation"]))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Gaussian initialization contract has an invalid kind or representation"
        ) from error
    parameters = value.get("parameters")
    parameters_path = str(parameters) if parameters else None
    adaptive = bool(
        value.get(
            "adaptiveDensification",
            kind is not InitializationKind.DENSE_SURFACE,
        )
    )

    if kind is InitializationKind.SPARSE_SFM:
        if parameters_path:
            raise ValueError("Sparse SfM initialization cannot declare dense parameters")
        if representation is not GaussianRepresentation.VOLUMETRIC_3D:
            raise ValueError("Sparse SfM initialization requires volumetric 3D Gaussians")
    else:
        if not parameters_path:
            raise ValueError("Dense Gaussian initialization requires a parameter sidecar")
        relative_parameters = Path(parameters_path)
        if relative_parameters.is_absolute() or ".." in relative_parameters.parts:
            raise ValueError(
                "Gaussian initialization parameter sidecar must stay inside the dataset"
            )
        if kind is InitializationKind.DIRECT_GAUSSIAN:
            if representation is not GaussianRepresentation.PREDICTED_ANISOTROPIC_3D:
                raise ValueError(
                    "Direct Gaussian initialization must preserve predicted anisotropy"
                )
        elif representation is GaussianRepresentation.PREDICTED_ANISOTROPIC_3D:
            raise ValueError(
                "Predicted anisotropic representation requires direct Gaussian initialization"
            )

    declared_path = dataset.get("initializationParameters")
    if parameters_path and str(declared_path or "") != parameters_path:
        raise ValueError("Gaussian initialization contract disagrees with the parameter sidecar")
    if bool(dataset.get("directGaussianPrior", False)) != (
        kind is InitializationKind.DIRECT_GAUSSIAN
    ):
        raise ValueError("Gaussian initialization contract disagrees with the direct-prior flag")
    if bool(dataset.get("denseGeometryPrior", False)) and kind is InitializationKind.SPARSE_SFM:
        raise ValueError("Gaussian initialization contract discards a declared dense prior")

    return InitializationContract(
        kind,
        representation,
        parameters_path,
        adaptive,
        "manifest",
    )


def initialization_manifest(
    kind: InitializationKind,
    representation: GaussianRepresentation,
    *,
    parameters: str | None,
    adaptive_densification: bool,
) -> dict[str, Any]:
    return {
        "schemaVersion": INITIALIZATION_CONTRACT_SCHEMA,
        "kind": str(kind),
        "representation": str(representation),
        "parameters": parameters,
        "adaptiveDensification": adaptive_densification,
    }
