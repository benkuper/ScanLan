from __future__ import annotations

import unittest

from scanlan_splat.initialization import (
    GaussianRepresentation,
    InitializationKind,
    initialization_manifest,
    resolve_initialization_contract,
)


class InitializationContractTests(unittest.TestCase):
    def test_sparse_dense_and_direct_contracts_are_distinct(self) -> None:
        sparse = {
            "gaussianInitialization": initialization_manifest(
                InitializationKind.SPARSE_SFM,
                GaussianRepresentation.VOLUMETRIC_3D,
                parameters=None,
                adaptive_densification=True,
            )
        }
        dense = {
            "metric": True,
            "initializationParameters": "surface.npz",
            "gaussianInitialization": initialization_manifest(
                InitializationKind.DENSE_SURFACE,
                GaussianRepresentation.SURFACE_DISCS_2D,
                parameters="surface.npz",
                adaptive_densification=False,
            ),
        }
        direct = {
            "denseGeometryPrior": True,
            "directGaussianPrior": True,
            "initializationParameters": "direct.npz",
            "gaussianInitialization": initialization_manifest(
                InitializationKind.DIRECT_GAUSSIAN,
                GaussianRepresentation.PREDICTED_ANISOTROPIC_3D,
                parameters="direct.npz",
                adaptive_densification=True,
            ),
        }

        self.assertEqual(
            resolve_initialization_contract(sparse).kind,
            InitializationKind.SPARSE_SFM,
        )
        self.assertTrue(resolve_initialization_contract(dense).uses_2dgs)
        self.assertTrue(resolve_initialization_contract(direct).is_direct)

    def test_conflicting_direct_prior_fails_closed(self) -> None:
        dataset = {
            "directGaussianPrior": True,
            "initializationParameters": "direct.npz",
            "gaussianInitialization": initialization_manifest(
                InitializationKind.DENSE_SURFACE,
                GaussianRepresentation.VOLUMETRIC_3D,
                parameters="direct.npz",
                adaptive_densification=True,
            ),
        }

        with self.assertRaisesRegex(ValueError, "direct-prior flag"):
            resolve_initialization_contract(dataset)

    def test_parameter_sidecar_cannot_escape_the_dataset(self) -> None:
        dataset = {
            "initializationParameters": "../outside.npz",
            "gaussianInitialization": initialization_manifest(
                InitializationKind.DENSE_SURFACE,
                GaussianRepresentation.VOLUMETRIC_3D,
                parameters="../outside.npz",
                adaptive_densification=True,
            ),
        }

        with self.assertRaisesRegex(ValueError, "stay inside"):
            resolve_initialization_contract(dataset)


if __name__ == "__main__":
    unittest.main()
