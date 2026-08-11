from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.appearance import (
    GAUSSIAN_MATERIAL_CONTRACT,
    compose_material_render_state,
    gaussian_material_manifest,
    material_regularization,
    resolve_gaussian_material_seeds,
)
from scanlan_splat.export import SH_C0, export_material_gaussians
from scanlan_splat.train import _material_parameter_initialization


def _write_sidecar(path: Path, *, complete: bool = True) -> None:
    values = {
        "points": np.zeros((2, 3), dtype=np.float32),
        "material_albedo_linear": np.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32),
        "material_roughness": np.asarray([0.1, 0.8], dtype=np.float32),
        "material_metallic": np.asarray([1.0, 0.0], dtype=np.float32),
        "material_transmission": np.asarray([0.7, 0.0], dtype=np.float32),
        "material_emission_linear": np.asarray([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]], dtype=np.float32),
        "material_confidence": np.asarray([0.9, 0.75], dtype=np.float32),
    }
    if not complete:
        values.pop("material_emission_linear")
    np.savez(path, **values)


class MaterialAppearanceTests(unittest.TestCase):
    def test_missing_material_prior_preserves_existing_display_rgb_initialization(self) -> None:
        colors = np.asarray([[0.1, 0.4, 0.8]], dtype=np.float32)
        parameters = _material_parameter_initialization(colors, None)

        reconstructed = parameters["diffuse_sh0"][:, 0, :] * SH_C0 + 0.5
        np.testing.assert_allclose(reconstructed, colors, atol=1e-7)
        self.assertEqual(float(parameters["material_confidence"][0]), 0.0)

    def test_declared_linear_material_seed_contract_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_sidecar(root / "initialization.npz")
            dataset = {
                "initializationParameters": "initialization.npz",
                "gaussianMaterial": gaussian_material_manifest("initialization.npz"),
            }
            seeds = resolve_gaussian_material_seeds(root, dataset, 2)

            self.assertIsNotNone(seeds)
            self.assertEqual(dataset["gaussianMaterial"]["contract"], GAUSSIAN_MATERIAL_CONTRACT)
            np.testing.assert_allclose(seeds.transmission, [0.7, 0.0])
            np.testing.assert_allclose(seeds.emission_linear[1], [2.0, 0.5, 0.0])

    def test_material_arrays_without_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_sidecar(root / "initialization.npz")
            with self.assertRaisesRegex(ValueError, "require a declared"):
                resolve_gaussian_material_seeds(
                    root, {"initializationParameters": "initialization.npz"}, 2
                )

    def test_partial_material_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_sidecar(root / "initialization.npz", complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_gaussian_material_seeds(
                    root,
                    {
                        "initializationParameters": "initialization.npz",
                        "gaussianMaterial": gaussian_material_manifest("initialization.npz"),
                    },
                    2,
                )

    def test_render_state_separates_emission_and_optical_opacity(self) -> None:
        import torch

        parameters = {
            "means": torch.zeros((1, 3), requires_grad=True),
            "opacities": torch.zeros(1, requires_grad=True),
            "diffuse_sh0": torch.zeros((1, 1, 3), requires_grad=True),
            "view_shN": torch.zeros((1, 15, 3), requires_grad=True),
            "emission_log": torch.full((1, 3), np.log(0.2), requires_grad=True),
            "transmission_logits": torch.zeros(1, requires_grad=True),
            "roughness_logits": torch.zeros(1, requires_grad=True),
            "metallic_logits": torch.zeros(1, requires_grad=True),
            "material_confidence": torch.ones(1, requires_grad=True),
            "emission_anchor_log": torch.full((1, 3), np.log(0.2), requires_grad=True),
            "transmission_anchor_logits": torch.zeros(1, requires_grad=True),
        }
        coefficients, optical_opacity, decoded = compose_material_render_state(
            parameters, True, SH_C0
        )

        np.testing.assert_allclose(
            coefficients[0, 0].detach().numpy(), np.full(3, 0.2 / SH_C0), rtol=1e-5
        )
        self.assertAlmostEqual(float(optical_opacity.detach()), 0.25, places=6)
        self.assertAlmostEqual(float(decoded["transmission"].detach()), 0.5, places=6)
        regularization, pieces = material_regularization(parameters, True)
        self.assertAlmostEqual(float(regularization.detach()), 0.0, places=7)
        self.assertAlmostEqual(float(pieces["prior"].detach()), 0.0, places=7)

    def test_specular_gate_blocks_photometric_transmission_gradient(self) -> None:
        import torch

        transmission = torch.zeros(1, requires_grad=True)
        parameters = {
            "opacities": torch.zeros(1, requires_grad=True),
            "diffuse_sh0": torch.zeros((1, 1, 3), requires_grad=True),
            "view_shN": torch.zeros((1, 15, 3), requires_grad=True),
            "emission_log": torch.full((1, 3), -16.0, requires_grad=True),
            "transmission_logits": transmission,
            "roughness_logits": torch.full((1,), -16.0, requires_grad=True),
            "metallic_logits": torch.full((1,), 16.0, requires_grad=True),
            "material_confidence": torch.ones(1, requires_grad=True),
            "emission_anchor_log": torch.full((1, 3), -16.0, requires_grad=True),
            "transmission_anchor_logits": torch.zeros(1, requires_grad=True),
        }
        _coefficients, optical_opacity, _decoded = compose_material_render_state(
            parameters, True, SH_C0
        )
        optical_opacity.sum().backward()

        self.assertLess(abs(float(transmission.grad)), 1e-6)

    def test_gsplat_topology_change_keeps_every_material_row_aligned(self) -> None:
        import torch
        from gsplat.strategy.ops import duplicate

        values = {
            "means": torch.zeros((2, 3)),
            "scales": torch.zeros((2, 3)),
            "quats": torch.zeros((2, 4)),
            "opacities": torch.zeros(2),
            "diffuse_sh0": torch.zeros((2, 1, 3)),
            "view_shN": torch.zeros((2, 15, 3)),
            "emission_log": torch.asarray([[-4.0] * 3, [-2.0] * 3]),
            "transmission_logits": torch.asarray([-3.0, 2.0]),
            "roughness_logits": torch.asarray([1.0, -1.0]),
            "metallic_logits": torch.asarray([-2.0, 2.0]),
            "material_confidence": torch.asarray([0.25, 0.9]),
            "emission_anchor_log": torch.asarray([[-4.0] * 3, [-2.0] * 3]),
            "transmission_anchor_logits": torch.asarray([-3.0, 2.0]),
        }
        parameters = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(value) for name, value in values.items()}
        )
        optimizers = {
            name: torch.optim.Adam([parameters[name]], lr=1e-3)
            for name in parameters
        }
        duplicate(parameters, optimizers, {}, torch.asarray([True, False]))

        self.assertTrue(all(len(value) == 3 for value in parameters.values()))
        self.assertAlmostEqual(float(parameters["material_confidence"][2].detach()), 0.25)
        self.assertAlmostEqual(float(parameters["transmission_logits"][2].detach()), -3.0)
        np.testing.assert_allclose(
            parameters["emission_log"][2].detach().numpy(), [-4.0, -4.0, -4.0]
        )

    def test_lossless_material_sidecar_round_trips_every_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "room-splat-material.npz"
            export_material_gaussians(
                path,
                diffuse_linear=np.full((2, 3), 0.25, dtype=np.float32),
                view_sh_linear=np.zeros((2, 15, 3), dtype=np.float32),
                emission_linear=np.full((2, 3), 0.1, dtype=np.float32),
                transmission=np.asarray([0.2, 0.3], dtype=np.float32),
                roughness=np.asarray([0.4, 0.5], dtype=np.float32),
                metallic=np.asarray([0.6, 0.7], dtype=np.float32),
                confidence=np.asarray([0.8, 0.9], dtype=np.float32),
                geometric_opacity=np.asarray([0.9, 0.8], dtype=np.float32),
                optical_opacity=np.asarray([0.72, 0.56], dtype=np.float32),
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(str(archive["contract"].item()), GAUSSIAN_MATERIAL_CONTRACT)
                np.testing.assert_allclose(archive["transmission"], [0.2, 0.3])
                np.testing.assert_allclose(archive["optical_opacity"], [0.72, 0.56])


if __name__ == "__main__":
    unittest.main()
