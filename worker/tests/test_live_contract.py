from __future__ import annotations

import unittest

from scanlan.live_contract import (
    LIVE_CONTRACT_VERSION,
    LiveFailurePolicy,
    LiveSubmapDescriptor,
    TrackingState,
    contract_status,
    pose_uncertainty,
    tracking_confidence,
    submap_message,
)
from scanlan.realtime import AlignmentQuality


class LiveContractTests(unittest.TestCase):
    def test_status_exposes_versioned_fail_closed_policy(self) -> None:
        status = contract_status(
            {"state": TrackingState.READY.value},
            failure_policy=LiveFailurePolicy(),
        )

        self.assertEqual(status["contractVersion"], LIVE_CONTRACT_VERSION)
        self.assertEqual(status["trackingState"], "ready")
        self.assertTrue(status["failurePolicy"]["rejected_pose_freezes_integration"])
        self.assertFalse(status["failurePolicy"]["preview_failure_is_fatal_to_capture"])

    def test_searching_state_explicitly_freezes_integration(self) -> None:
        status = contract_status({"state": TrackingState.SEARCHING.value})

        self.assertTrue(status["integrationFrozen"])
        self.assertEqual(status["scaleStatus"], "SENSOR_METRIC")

    def test_tracking_confidence_and_uncertainty_are_bounded(self) -> None:
        quality = AlignmentQuality(True, 0.75, 0.90, 0.006, 2_000, "accepted")

        confidence = tracking_confidence(quality)
        position_mm, angle_degrees = pose_uncertainty(quality)

        self.assertGreater(confidence, 0.75)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsNotNone(position_mm)
        self.assertIsNotNone(angle_degrees)

    def test_submap_contract_rejects_non_rigid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "4x4"):
            LiveSubmapDescriptor(
                id="submap-0",
                local_origin=(1.0,),
                global_from_local=tuple(float(index % 5 == 0) for index in range(16)),
                state="active",
                first_sequence=0,
                last_sequence=0,
                voxel_size_m=0.01,
                voxel_count=1,
                point_count=1,
                observation_count=1,
                confidence=1.0,
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(1.0, 1.0, 1.0),
                resident="gpu",
            )

    def test_submap_message_uses_cross_language_field_names(self) -> None:
        identity = tuple(float(index % 5 == 0) for index in range(16))
        submap = LiveSubmapDescriptor(
            id="submap-0",
            local_origin=identity,
            global_from_local=identity,
            state="active",
            first_sequence=2,
            last_sequence=7,
            voxel_size_m=0.01,
            voxel_count=20,
            point_count=10,
            observation_count=3,
            confidence=0.8,
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(1.0, 1.0, 1.0),
            resident="gpu",
        )

        message = submap_message(7, [submap])

        self.assertEqual(message["contractVersion"], 2)
        self.assertEqual(message["submaps"][0]["firstSequence"], 2)
        self.assertNotIn("first_sequence", message["submaps"][0])


if __name__ == "__main__":
    unittest.main()
