from __future__ import annotations

import json
import unittest

import numpy as np

from scanlan_validation import (
    CameraValidationConfig,
    GeometryValidationConfig,
    validate_camera_trajectory,
    validate_depth,
    validate_geometry,
    validate_ray_depths,
    validate_scale,
)


class ValidationEngineTests(unittest.TestCase):
    def test_camera_gate_rejects_confidence_rigid_and_continuity_failures(self) -> None:
        poses = np.repeat(np.eye(4)[None], 6, axis=0)
        poses[:, 0, 3] = [0.0, 0.1, 0.2, 8.0, 0.4, 0.5]
        confidence = np.ones(6)
        confidence[4] = 0.2
        result = validate_camera_trajectory(
            poses,
            confidence,
            CameraValidationConfig(maximum_translation_step=2.0),
        )
        self.assertFalse(result.accepted)
        self.assertFalse(result.frame_mask[3])
        self.assertFalse(result.frame_mask[4])
        self.assertGreater(result.drift_risk, 0.0)

    def test_camera_gate_normalizes_irregular_sampling(self) -> None:
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, 0, 3] = [0.0, 0.1, 0.5, 0.6, 1.0]
        result = validate_camera_trajectory(
            poses,
            sample_positions=np.asarray([0, 1, 5, 6, 10]),
        )
        self.assertTrue(result.accepted)

    def test_rejected_nonfinite_camera_report_remains_json_serializable(self) -> None:
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[1, 0, 3] = np.nan
        result = validate_camera_trajectory(poses)
        self.assertFalse(result.accepted)
        self.assertFalse(result.frame_mask[1])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_scale_gate_recovers_similarity_and_rejects_outlier(self) -> None:
        source = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
            dtype=np.float64,
        )
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        target = 2.5 * (source @ rotation.T) + np.asarray([4.0, -2.0, 1.0])
        target[-1] += 20.0
        result = validate_scale(source, target)
        self.assertTrue(result.accepted)
        assert result.transform is not None
        self.assertAlmostEqual(result.transform.scale, 2.5, places=5)
        self.assertFalse(result.inlier_mask[-1])

    def test_depth_gate_preserves_metric_scale(self) -> None:
        measured = np.linspace(0.5, 4.0, 1024).reshape(32, 32)
        accepted = validate_depth(measured, measured + 0.005)
        rejected = validate_depth(measured, measured * 1.2)
        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)

    def test_ray_gate_distinguishes_support_free_space_and_occlusion(self) -> None:
        result = validate_ray_depths(
            np.asarray([2.0, 1.0, 3.0, 2.0]),
            np.asarray([2.01, 2.0, 2.0, 0.0]),
        )
        np.testing.assert_array_equal(result.support_mask, [True, False, False, False])
        np.testing.assert_array_equal(result.free_space_violation_mask, [False, True, False, False])
        np.testing.assert_array_equal(result.occluded_mask, [False, False, True, False])
        np.testing.assert_array_equal(result.unknown_mask, [False, False, False, True])

    def test_geometry_gate_is_fail_closed(self) -> None:
        points = np.asarray([[0, 0, 1], [np.nan, 0, 0], [2e6, 0, 0], [0, 0, 2]])
        result = validate_geometry(
            points,
            np.asarray([0.9, 0.9, 0.9, 0.2]),
            free_space_violation_mask=np.asarray([False, False, False, True]),
            config=GeometryValidationConfig(minimum_confidence=0.5),
        )
        np.testing.assert_array_equal(result.point_mask, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
