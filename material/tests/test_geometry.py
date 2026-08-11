from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_material.analysis import FusedMaterialSurface
from scanlan_material.contracts import MATERIAL_CLASSES, OPTICAL_RISKS, MaterialPrediction
from scanlan_material.geometry import (
    PROVENANCE_GENERATED,
    PROVENANCE_LEARNED,
    PROVENANCE_MEASURED,
    GeometryProposal,
    apply_depth_confidence_policy,
    evaluate_repair_boundary,
    neutral_geometry_policy,
    prediction_geometry_policy,
    read_geometry_result,
    refine_material_geometry,
    surface_geometry_policy,
    write_geometry_result,
)


def prediction(
    *,
    risk: str | None = None,
    risk_probability: float = 0.0,
    material: str = "opaque_dielectric",
    confidence: float = 1.0,
) -> MaterialPrediction:
    classes = np.zeros((2, 3, len(MATERIAL_CLASSES)), dtype=np.float32)
    classes[..., MATERIAL_CLASSES.index(material)] = 1.0
    risks = np.zeros((2, 3, len(OPTICAL_RISKS)), dtype=np.float32)
    if risk is not None:
        risks[..., OPTICAL_RISKS.index(risk)] = risk_probability
    return MaterialPrediction(
        classes,
        risks,
        np.ones((2, 3), dtype=bool),
        np.full((2, 3), confidence, dtype=np.float32),
    )


def surface(
    vertex_count: int,
    *,
    risk: str | None = None,
    risk_probability: float = 0.0,
    material: str = "opaque_dielectric",
    confidence: float = 0.95,
) -> FusedMaterialSurface:
    classes = np.zeros((vertex_count, len(MATERIAL_CLASSES)), dtype=np.float32)
    classes[:, MATERIAL_CLASSES.index(material)] = 1.0
    risks = np.zeros((vertex_count, len(OPTICAL_RISKS)), dtype=np.float32)
    if risk is not None:
        risks[:, OPTICAL_RISKS.index(risk)] = risk_probability
    return FusedMaterialSurface(
        classes,
        risks,
        np.ones(vertex_count, dtype=bool),
        np.full(vertex_count, confidence, dtype=np.float32),
        np.full(vertex_count, 3, dtype=np.uint16),
        np.full(vertex_count, 2.5, dtype=np.float32),
        np.zeros(vertex_count, dtype=np.int32),
        metadata={"sourceViewCount": 3},
    ).validated()


def square() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )


class MaterialGeometryTests(unittest.TestCase):
    def test_missing_material_output_is_a_neutral_no_op(self) -> None:
        policy = neutral_geometry_policy((2, 3))
        np.testing.assert_array_equal(policy.sensor_depth_multiplier, np.ones((2, 3)))
        np.testing.assert_array_equal(policy.generated_depth_multiplier, np.ones((2, 3)))
        self.assertFalse(np.any(policy.protected_mask))
        self.assertEqual(policy.metadata["materialEvidence"], "missing")

    def test_glass_warning_reduces_depth_authority_and_vetoes_blind_repair(self) -> None:
        policy = prediction_geometry_policy(
            prediction(risk="glass_or_transmissive", risk_probability=0.95, confidence=0.9)
        )
        self.assertTrue(np.all(policy.protected_mask))
        self.assertTrue(np.all(policy.sensor_depth_multiplier < 0.20))
        self.assertTrue(np.all(policy.generated_depth_multiplier < 0.20))
        self.assertTrue(np.all(policy.repair_authority == 0.0))

    def test_dynamic_identity_discards_static_geometry(self) -> None:
        policy = prediction_geometry_policy(prediction(material="dynamic", confidence=0.95))
        self.assertTrue(np.all(policy.discard_mask))
        self.assertTrue(np.all(policy.sensor_depth_multiplier == 0.0))
        self.assertTrue(np.all(policy.refinement_authority == 0.0))

    def test_depth_policy_only_reduces_upstream_confidence(self) -> None:
        policy = prediction_geometry_policy(
            prediction(risk="high_specular", risk_probability=0.8, confidence=1.0)
        )
        confidence = np.full((2, 3), 0.75, dtype=np.float32)
        measured = apply_depth_confidence_policy(confidence, policy, PROVENANCE_MEASURED)
        learned = apply_depth_confidence_policy(confidence, policy, PROVENANCE_LEARNED)
        self.assertTrue(np.all(measured <= confidence))
        self.assertTrue(np.all(learned <= confidence))
        self.assertTrue(np.all(measured < learned))

    def test_material_risk_vetoes_nearby_depth_supported_boundary_fill(self) -> None:
        vertices, _ = square()
        policy = surface_geometry_policy(
            surface(len(vertices), risk="mirror", risk_probability=0.9)
        )
        decision = evaluate_repair_boundary(
            vertices,
            vertices + np.asarray([0.001, 0.0, 0.0]),
            policy,
            maximum_distance_m=0.01,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.matched_sample_count, 4)
        self.assertEqual(decision.protected_fraction, 1.0)
        self.assertIn("preserves", decision.reason)

    def test_opaque_multiview_proposal_refines_the_second_pass(self) -> None:
        vertices, triangles = square()
        candidate = vertices.copy()
        candidate[:, 2] = 0.01
        result = refine_material_geometry(
            vertices,
            triangles,
            surface(len(vertices)),
            GeometryProposal(
                candidate,
                np.full(len(vertices), 0.9, dtype=np.float32),
                np.full(len(vertices), 2.5, dtype=np.float32),
                np.full(len(vertices), 0.002, dtype=np.float32),
                np.full(len(vertices), PROVENANCE_GENERATED, dtype=np.uint8),
            ),
            voxel_size_m=0.01,
        )
        self.assertTrue(np.all(result.accepted_mask))
        self.assertTrue(np.all(result.vertices[:, 2] > 0.0))
        self.assertTrue(np.all(result.vertices[:, 2] <= 0.01))
        self.assertEqual(result.metadata["acceptedVertexCount"], 4)

    def test_protected_surface_rejects_sensor_but_accepts_strict_multiview_recovery(self) -> None:
        vertices, triangles = square()
        material = surface(
            len(vertices), risk="glass_or_transmissive", risk_probability=0.95
        )
        candidate = vertices.copy()
        candidate[:, 2] = 0.004
        common = dict(
            vertices=candidate,
            confidence=np.full(len(vertices), 0.95, dtype=np.float32),
            effective_view_count=np.full(len(vertices), 3.0, dtype=np.float32),
            heldout_residual_m=np.full(len(vertices), 0.002, dtype=np.float32),
        )
        sensor_result = refine_material_geometry(
            vertices,
            triangles,
            material,
            GeometryProposal(
                provenance=np.full(len(vertices), PROVENANCE_MEASURED, dtype=np.uint8),
                **common,
            ),
            voxel_size_m=0.01,
        )
        learned_result = refine_material_geometry(
            vertices,
            triangles,
            material,
            GeometryProposal(
                provenance=np.full(len(vertices), PROVENANCE_LEARNED, dtype=np.uint8),
                **common,
            ),
            voxel_size_m=0.01,
        )
        self.assertFalse(np.any(sensor_result.accepted_mask))
        self.assertTrue(np.all(learned_result.accepted_mask))
        self.assertEqual(learned_result.metadata["protectedRecoveredVertexCount"], 4)

    def test_topology_gate_rejects_a_locally_inverting_proposal(self) -> None:
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        candidate = vertices.copy()
        candidate[2, 1] = -0.05
        result = refine_material_geometry(
            vertices,
            triangles,
            surface(len(vertices)),
            GeometryProposal(
                candidate,
                np.full(len(vertices), 0.95, dtype=np.float32),
                np.full(len(vertices), 3.0, dtype=np.float32),
                np.full(len(vertices), 0.001, dtype=np.float32),
                np.full(len(vertices), PROVENANCE_GENERATED, dtype=np.uint8),
            ),
            voxel_size_m=0.1,
        )
        self.assertFalse(np.any(result.accepted_mask))
        np.testing.assert_array_equal(result.vertices, vertices)
        self.assertGreater(result.metadata["rejectedTopologyVertexCount"], 0)

    def test_refinement_result_round_trip_is_typed_and_atomic(self) -> None:
        vertices, triangles = square()
        candidate = vertices.copy()
        candidate[:, 2] = 0.005
        result = refine_material_geometry(
            vertices,
            triangles,
            surface(len(vertices)),
            GeometryProposal(
                candidate,
                np.full(len(vertices), 0.9, dtype=np.float32),
                np.full(len(vertices), 2.5, dtype=np.float32),
                np.full(len(vertices), 0.001, dtype=np.float32),
                np.full(len(vertices), PROVENANCE_GENERATED, dtype=np.uint8),
                metadata={"fixture": "square"},
            ),
            voxel_size_m=0.01,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "material-geometry.npz"
            write_geometry_result(path, result)
            restored = read_geometry_result(path)
        self.assertEqual(restored.metadata["fixture"], "square")
        np.testing.assert_allclose(restored.vertices, result.vertices)
        np.testing.assert_array_equal(restored.accepted_mask, result.accepted_mask)


if __name__ == "__main__":
    unittest.main()
