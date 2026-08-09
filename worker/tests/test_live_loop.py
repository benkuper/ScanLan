from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from scanlan.live_loop import (
    LocalAnchorDatabase,
    PoseGraphLoop,
    SubmapPoseGraph,
    interpolate_transform,
    verify_loop_candidate,
)


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = (x, y, z)
    return value


@dataclass(frozen=True)
class _Frame:
    sequence: int


@dataclass(frozen=True)
class _Anchor:
    frame: _Frame


class LiveLoopTests(unittest.TestCase):
    def test_anchor_database_stays_bounded_and_queries_capture_history(self) -> None:
        database: LocalAnchorDatabase[_Anchor] = LocalAnchorDatabase(8, 3)
        for sequence in range(16):
            database.add(_Anchor(_Frame(sequence)))

        self.assertLessEqual(len(database.entries), 8)
        self.assertEqual(database.entries[0].frame.sequence, 0)
        queried = set()
        for _ in range(len(database.entries) * 2):
            queried.update(
                anchor.frame.sequence
                for anchor in database.candidates(
                    previous_sequence=15,
                    pending_sequence=None,
                    limit=3,
                )
            )
        self.assertTrue(
            {anchor.frame.sequence for anchor in database.entries[:-1]}.issubset(queried)
        )

    def test_viewport_transform_interpolates_without_overshoot(self) -> None:
        before = np.eye(4)
        after = _translation(0.4)
        half = interpolate_transform(before, after, 0.5)

        self.assertAlmostEqual(float(half[0, 3]), 0.2)
        np.testing.assert_allclose(interpolate_transform(before, after, 1.0), after)

    def test_pose_graph_replay_is_deterministic_and_closes_drift(self) -> None:
        import open3d as o3d

        def solve() -> list[np.ndarray]:
            graph = SubmapPoseGraph(o3d, 0.02)
            graph.add_submap("submap-0", _translation(0.0))
            graph.add_submap("submap-1", _translation(1.1))
            graph.add_submap("submap-2", _translation(2.2))
            graph.add_loop(
                PoseGraphLoop(
                    "submap-2",
                    "submap-0",
                    _translation(2.0),
                    np.eye(6) * 1_000.0,
                    0.9,
                    0.01,
                    300,
                )
            )
            solution = graph.optimize()
            self.assertTrue(solution.accepted, solution.reason)
            return [solution.transforms[key] for key in sorted(solution.transforms)]

        first = solve()
        second = solve()
        for first_transform, second_transform in zip(first, second, strict=True):
            np.testing.assert_allclose(first_transform, second_transform, atol=1e-12)
        self.assertAlmostEqual(float(first[-1][0, 3]), 2.0, delta=0.02)

    def test_geometric_loop_verification_accepts_a_small_known_alignment(self) -> None:
        import open3d as o3d

        random = np.random.default_rng(42)
        source = random.uniform(-0.5, 0.5, size=(4_000, 3))
        source[:, 2] += 0.15 * np.sin(source[:, 0] * 8.0) * np.cos(source[:, 1] * 7.0)
        target_from_source = _translation(0.08, -0.02, 0.01)
        target = source + target_from_source[:3, 3]

        result = verify_loop_candidate(
            o3d,
            source_points=source,
            target_points=target,
            initial_target_from_source=_translation(0.07, -0.015, 0.01),
            voxel_size_m=0.01,
        )

        self.assertTrue(result.accepted, result.reason)
        self.assertGreater(result.correspondence_count, 200)
        np.testing.assert_allclose(
            result.target_from_source[:3, 3], target_from_source[:3, 3], atol=0.01
        )


if __name__ == "__main__":
    unittest.main()
