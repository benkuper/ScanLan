from __future__ import annotations

import unittest

try:
    import torch

    from scanlan_splat.pose import (
        POSE_ROTATION_PARAMETER_LIMIT,
        POSE_TRANSLATION_LIMIT_M,
        constrain_pose_offsets_,
        pose_correction_statistics,
        pose_delta_matrix,
        pose_regularization,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PoseRefinementTests(unittest.TestCase):
    def test_zero_offsets_are_identity_corrections(self) -> None:
        corrections = pose_delta_matrix(torch.zeros((2, 9), dtype=torch.float32))
        torch.testing.assert_close(
            corrections,
            torch.eye(4).expand(2, 4, 4),
        )

    def test_pose_projection_fixes_gauge_and_limits_corrections(self) -> None:
        offsets = torch.full((3, 9), 1.0, dtype=torch.float32)
        constrain_pose_offsets_(offsets)

        torch.testing.assert_close(offsets[0], torch.zeros(9))
        self.assertLessEqual(
            float(torch.linalg.vector_norm(offsets[1:, :3], dim=-1).max()),
            POSE_TRANSLATION_LIMIT_M + 1e-6,
        )
        self.assertLessEqual(
            float(offsets[1:, 3:].abs().max()),
            POSE_ROTATION_PARAMETER_LIMIT + 1e-6,
        )

    def test_rotation_and_regularization_remain_well_formed(self) -> None:
        offsets = torch.zeros((3, 9), dtype=torch.float32)
        offsets[1, 3:] = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01, 0.02])
        corrections = pose_delta_matrix(offsets)
        rotations = corrections[..., :3, :3]
        torch.testing.assert_close(
            rotations.transpose(-1, -2) @ rotations,
            torch.eye(3).expand(3, 3, 3),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(3), atol=1e-5, rtol=1e-5)
        pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.assertGreater(float(pose_regularization(offsets, pairs)), 0.0)
        self.assertGreater(pose_correction_statistics(offsets)["maximumRotationDegrees"], 0.0)


if __name__ == "__main__":
    unittest.main()
