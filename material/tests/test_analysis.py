from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_material.analysis import (
    MaterialCamera,
    MaterialView,
    fuse_material_surface,
    read_surface_material,
    run_two_pass_analysis,
    select_coarse_cameras,
    select_final_views,
    vertex_normals,
    write_surface_material,
)
from scanlan_material.contracts import MATERIAL_CLASSES, OPTICAL_RISKS, MaterialPrediction


def camera(identifier: str, x: float = 0.0, depth: np.ndarray | None = None) -> MaterialCamera:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return MaterialCamera(identifier, 8, 8, 8.0, 8.0, 3.5, 3.5, pose, depth_m=depth)


def prediction(
    material: str = "opaque_dielectric",
    *,
    risk: str | None = None,
    risk_probability: float = 0.0,
    confidence: float = 0.95,
) -> MaterialPrediction:
    classes = np.zeros((8, 8, len(MATERIAL_CLASSES)), dtype=np.float32)
    classes[..., MATERIAL_CLASSES.index(material)] = 1.0
    risks = np.zeros((8, 8, len(OPTICAL_RISKS)), dtype=np.float32)
    if risk is not None:
        risks[..., OPTICAL_RISKS.index(risk)] = risk_probability
    valid = np.ones((8, 8), dtype=bool)
    return MaterialPrediction(
        classes,
        risks,
        valid,
        np.full((8, 8), confidence, dtype=np.float32),
        albedo_linear=np.full((8, 8, 3), 0.25, dtype=np.float32),
        roughness=np.full((8, 8), 0.6, dtype=np.float32),
        metallic=np.zeros((8, 8), dtype=np.float32),
        transmission=np.zeros((8, 8), dtype=np.float32),
        normal_camera=np.broadcast_to(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (8, 8, 3)
        ).copy(),
    )


def plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-0.25, -0.25, 1.0],
            [0.25, -0.25, 1.0],
            [-0.25, 0.25, 1.0],
            [0.25, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return vertices, vertex_normals(vertices, triangles), triangles


class TwoPassMaterialTests(unittest.TestCase):
    def test_coarse_sampling_spreads_over_measured_camera_path(self) -> None:
        cameras = [camera(str(index), index * 0.1) for index in range(9)]
        selected = select_coarse_cameras(cameras, 3)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 8)
        self.assertIn(selected[1], (3, 4, 5))

    def test_final_selection_preserves_a_high_risk_view(self) -> None:
        vertices, normals, _ = plane()
        views = [
            MaterialView(camera("left", -0.2), prediction()),
            MaterialView(
                camera("risk", 0.0),
                prediction(risk="glass_or_transmissive", risk_probability=0.98),
            ),
            MaterialView(camera("right", 0.2), prediction()),
        ]
        selected = select_final_views(vertices, normals, views, maximum_views=1)
        self.assertEqual(views[selected[0]].camera.identifier, "risk")

    def test_fusion_keeps_single_view_risk_and_builds_connected_region(self) -> None:
        vertices, normals, triangles = plane()
        views = [
            MaterialView(camera("opaque", -0.1), prediction()),
            MaterialView(
                camera("glass-warning", 0.1),
                prediction(risk="glass_or_transmissive", risk_probability=0.95),
            ),
        ]
        fused = fuse_material_surface(vertices, normals, triangles, views)
        self.assertTrue(np.all(fused.valid_mask))
        self.assertTrue(np.all(fused.support_count == 2))
        self.assertTrue(np.all(fused.effective_view_count > 1.5))
        self.assertTrue(
            np.all(
                fused.optical_risk_probabilities[
                    :, OPTICAL_RISKS.index("glass_or_transmissive")
                ]
                > 0.7
            )
        )
        self.assertEqual(len(np.unique(fused.region_ids)), 1)
        self.assertEqual(fused.metadata["sourceViewCount"], 2)

    def test_metric_depth_rejects_occluded_surface_vertex(self) -> None:
        vertices, normals, triangles = plane()
        depth = np.ones((8, 8), dtype=np.float32)
        depth[2, 2] = 0.5
        fused = fuse_material_surface(
            vertices,
            normals,
            triangles,
            [MaterialView(camera("depth", depth=depth), prediction())],
        )
        self.assertEqual(np.count_nonzero(fused.valid_mask), 3)
        self.assertEqual(np.count_nonzero(fused.region_ids < 0), 1)

    def test_isolated_mesh_vertex_remains_unsupported(self) -> None:
        vertices, normals, triangles = plane()
        vertices = np.concatenate((vertices, np.asarray([[4.0, 4.0, 4.0]], dtype=np.float32)))
        normals = np.concatenate((normals, np.zeros((1, 3), dtype=np.float32)))
        fused = fuse_material_surface(
            vertices,
            normals,
            triangles,
            [MaterialView(camera("one"), prediction())],
        )
        self.assertFalse(fused.valid_mask[-1])
        self.assertEqual(fused.region_ids[-1], -1)

    def test_surface_contract_round_trip_is_atomic_and_typed(self) -> None:
        vertices, normals, triangles = plane()
        fused = fuse_material_surface(
            vertices,
            normals,
            triangles,
            [MaterialView(camera("one"), prediction(material="metal"))],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "surface-material.npz"
            write_surface_material(path, fused)
            restored = read_surface_material(path)
        self.assertEqual(restored.metadata["sourceViews"], ["one"])
        np.testing.assert_array_equal(restored.region_ids, fused.region_ids)
        self.assertEqual(
            int(np.argmax(restored.class_probabilities[0])), MATERIAL_CLASSES.index("metal")
        )

    def test_two_pass_runner_only_invokes_final_backend_for_selected_views(self) -> None:
        vertices, normals, triangles = plane()
        cameras = [camera(str(index), (index - 2) * 0.1) for index in range(5)]
        coarse_calls: list[str] = []
        final_calls: list[str] = []

        def coarse(value: MaterialCamera) -> MaterialPrediction:
            coarse_calls.append(value.identifier)
            return prediction(
                risk="mirror" if value.identifier == "2" else None,
                risk_probability=0.99,
            )

        def final(value: MaterialCamera) -> MaterialPrediction:
            final_calls.append(value.identifier)
            return prediction()

        fused = run_two_pass_analysis(
            vertices,
            normals,
            triangles,
            cameras,
            coarse,
            final,
            maximum_coarse_views=5,
            maximum_final_views=2,
        )
        self.assertEqual(len(coarse_calls), 5)
        self.assertEqual(len(final_calls), 2)
        self.assertIn("2", final_calls)
        self.assertEqual(fused.metadata["finalViews"], final_calls)

    def test_prediction_must_match_calibrated_source_grid(self) -> None:
        invalid_camera = MaterialCamera("bad", 7, 8, 8.0, 8.0, 3.0, 3.5, np.eye(4))
        with self.assertRaisesRegex(ValueError, "source-aligned"):
            MaterialView(invalid_camera, prediction()).validated()

    def test_final_views_cannot_publish_partial_pbr_fields(self) -> None:
        vertices, normals, triangles = plane()
        without_pbr = prediction()
        without_pbr = MaterialPrediction(
            without_pbr.class_probabilities,
            without_pbr.optical_risk_probabilities,
            without_pbr.valid_mask,
            without_pbr.confidence,
        )
        with self.assertRaisesRegex(ValueError, "optional albedo_linear"):
            fuse_material_surface(
                vertices,
                normals,
                triangles,
                [
                    MaterialView(camera("pbr"), prediction()),
                    MaterialView(camera("identity-only", 0.1), without_pbr),
                ],
            )


if __name__ == "__main__":
    unittest.main()
