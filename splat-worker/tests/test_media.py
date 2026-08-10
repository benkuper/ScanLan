from __future__ import annotations

import json
import os
import time
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scanlan_splat.cli import _publish_failure
from scanlan_splat.media import (
    _CameraSolveTelemetry,
    MediaPreparationOptions,
    _adaptive_keyframe_reason,
    _configure_sfm,
    _camera_recovery_pairs,
    _cpu_match_pairs,
    _descriptor_distance,
    _extract_lingbot_context,
    _extract_video_streaming,
    _feature_extraction_groups,
    _feature_extraction_batch_size,
    _guided_match_pairs,
    _geometry_fusion_confidence,
    _limited_size,
    _minimum_useful_registration_count,
    _materialize_observation_inputs,
    _media_dataset_fingerprint,
    _progress_heartbeat,
    _source_fingerprint,
    _tracked_visual_motion,
    _video_intrinsic_spread,
    _write_json_atomic,
    _write_initialization_parameters,
    adaptive_frame_selection_status,
    prepare_media_observations,
)
from scanlan_splat.lingbot import LingbotGeometry


class MediaPreparationTests(unittest.TestCase):
    @staticmethod
    def _camera_proposal(count: int):
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
        poses[:, 0, 3] = np.linspace(0.0, 3.0, count)
        poses[:, 2, 3] = 0.1 * np.sin(np.linspace(0.0, np.pi, count))
        return SimpleNamespace(
            world_from_cameras=poses,
            frame_confidence=np.linspace(0.75, 1.0, count),
            backend="fixture learned cameras",
        )

    def test_default_video_sampling_preserves_handheld_overlap(self) -> None:
        options = MediaPreparationOptions()

        self.assertEqual(options.video_fps, 15.0)
        self.assertEqual(options.maximum_video_frames, 3_000)

    def test_dense_fusion_sidecar_separates_geometry_confidence_from_opacity(self) -> None:
        geometry = LingbotGeometry(
            world_from_cameras=np.repeat(np.eye(4)[None], 2, axis=0),
            intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
            points=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
            colors=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            scales=np.full((2, 3), 0.02, dtype=np.float32),
            quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            source_frame_indices=np.asarray([0, 1], dtype=np.int32),
            frame_confidence=np.asarray([0.6, 0.9], dtype=np.float32),
            backend="fixture",
            model_path="fixture",
            processed_size=(32, 32),
            opacities=np.asarray([0.01, 0.8], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initialization-parameters.npz"
            _write_initialization_parameters(
                path,
                geometry.points,
                geometry.colors,
                geometry.scales,
                geometry.quaternions,
                confidence=geometry.opacities,
                fusion_confidence=_geometry_fusion_confidence(geometry),
                source_frame_indices=geometry.source_frame_indices,
                provenance=np.full(2, 2, dtype=np.uint8),
            )
            with np.load(path, allow_pickle=False) as values:
                np.testing.assert_allclose(values["confidence"], geometry.opacities)
                np.testing.assert_allclose(values["fusion_confidence"], [0.6, 0.9])
                np.testing.assert_array_equal(values["source_frame_indices"], [0, 1])
                np.testing.assert_array_equal(values["provenance"], [2, 2])

    def test_adaptive_frame_policy_is_exposed_to_release_diagnostics(self) -> None:
        status = adaptive_frame_selection_status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["mode"], "adaptive_optical_flow")
        self.assertTrue(status["maximumFramesIsSafetyCeiling"])
        self.assertIn("tracked_overlap", status["signals"])
        self.assertIn("camera_motion", status["signals"])

    def test_adaptive_keyframes_follow_motion_instead_of_elapsed_video_length(self) -> None:
        import cv2

        random = np.random.default_rng(42)
        reference = random.integers(0, 256, (240, 320), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (3, 3), 0.0)
        translated = cv2.warpAffine(
            reference,
            np.asarray([[1.0, 0.0, 28.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        stationary = _tracked_visual_motion(reference, reference)
        moving = _tracked_visual_motion(reference, translated)

        self.assertTrue(stationary["reliable"])
        self.assertGreater(float(stationary["trackedRatio"]), 0.95)
        self.assertIsNone(_adaptive_keyframe_reason(0.5, stationary, 0.0))
        self.assertIn(
            _adaptive_keyframe_reason(0.5, moving, 0.0),
            {"camera_motion", "tracked_overlap"},
        )
        self.assertEqual(_adaptive_keyframe_reason(2.1, stationary, 0.0), "maximum_gap")

    def test_cli_failure_publication_preserves_progress_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = root / "outputs" / "splat-progress.json"
            progress.parent.mkdir()
            progress.write_text(
                json.dumps({"stage": "camera_solving", "progress": 0.56}),
                encoding="utf-8",
            )

            _publish_failure(root, RuntimeError("registered 87 of 181 cameras"))

            value = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(value["stage"], "failed")
            self.assertEqual(value["detail"], "registered 87 of 181 cameras")
            self.assertEqual(value["progress"], 0.56)

    def test_atomic_json_write_retries_windows_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.json"
            real_replace = os.replace
            attempts = 0

            def replace_after_transient_lock(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(5, "Access is denied", str(destination))
                real_replace(source, destination)

            with (
                patch("scanlan_splat.media.os.replace", side_effect=replace_after_transient_lock),
                patch("scanlan_splat.media.time.sleep"),
            ):
                _write_json_atomic(path, {"stage": "feature_matching"})

            self.assertEqual(attempts, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["stage"], "feature_matching")

    def test_image_limit_preserves_aspect_ratio_without_upscaling(self) -> None:
        self.assertEqual(_limited_size(4000, 2000, 1000), (1000, 500))
        self.assertEqual(_limited_size(640, 480, 1000), (640, 480))

    def test_descriptor_distance_is_exposure_invariant(self) -> None:
        left = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        left /= np.linalg.norm(left)
        self.assertAlmostEqual(_descriptor_distance(left, left), 0.0, places=6)
        self.assertGreater(_descriptor_distance(left, -left), 1.9)

    def test_cpu_pairing_is_complete_for_photos_and_bounded_for_video(self) -> None:
        names = [f"{index:03}.jpg" for index in range(300)]

        photo_pairs = _cpu_match_pairs(names, sequential=False)
        video_pairs = _cpu_match_pairs(names, sequential=True)

        self.assertEqual(len(photo_pairs), 300 * 299 // 2)
        self.assertLess(len(video_pairs), len(photo_pairs) // 5)
        self.assertIn(("000.jpg", "128.jpg"), video_pairs)

    def test_learned_camera_proposal_builds_bounded_connected_pair_graph(self) -> None:
        names = [f"{index:04}.jpg" for index in range(240)]
        pairs = _guided_match_pairs(
            names,
            self._camera_proposal(len(names)),
            sequential=True,
        )

        self.assertLess(len(pairs), len(names) * 40)
        self.assertIn(("0000.jpg", "0001.jpg"), pairs)
        self.assertIn(("0000.jpg", "0128.jpg"), pairs)
        touched = {name for pair in pairs for name in pair}
        self.assertEqual(touched, set(names))

        photo_names = names[:16]
        photo_pairs = _guided_match_pairs(
            photo_names,
            self._camera_proposal(len(photo_names)),
            sequential=False,
        )
        self.assertLess(len(photo_pairs), len(photo_names) * (len(photo_names) - 1) // 2)

    def test_degenerate_learned_trajectory_cannot_guide_matching(self) -> None:
        proposal = self._camera_proposal(6)
        proposal.world_from_cameras[:, :3, 3] = 0.0

        self.assertEqual(
            _guided_match_pairs(
                [f"{index}.jpg" for index in range(6)],
                proposal,
                sequential=False,
            ),
            [],
        )

    def test_camera_recovery_targets_registered_learned_neighbours(self) -> None:
        names = [f"{index:02}.jpg" for index in range(8)]
        registered = {names[index] for index in (0, 1, 2, 5, 6, 7)}
        pairs = _camera_recovery_pairs(
            names,
            self._camera_proposal(len(names)),
            registered,
            {("02.jpg", "03.jpg")},
        )

        self.assertTrue(pairs)
        self.assertNotIn(("02.jpg", "03.jpg"), pairs)
        self.assertTrue(
            all(
                (left in registered) != (right in registered)
                for left, right in pairs
            )
        )

    def test_feature_extraction_batches_bound_native_image_queues(self) -> None:
        self.assertEqual(
            _feature_extraction_batch_size(360, use_cuda=True, worker_threads=30),
            16,
        )
        self.assertEqual(
            _feature_extraction_batch_size(360, use_cuda=False, worker_threads=30),
            8,
        )
        self.assertEqual(
            _feature_extraction_batch_size(5, use_cuda=True, worker_threads=30),
            5,
        )

    def test_native_stage_heartbeat_reports_elapsed_without_fake_eta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _progress_heartbeat(
                root,
                "camera_solving",
                "Solving cameras",
                0.45,
                compute_backend="COLMAP CPU",
                metrics={"imageCount": 12},
            ):
                time.sleep(1.1)

            progress = json.loads(
                (root / "outputs" / "splat-progress.json").read_text(encoding="utf-8")
            )
            self.assertIn("elapsed", progress["detail"])
            self.assertGreaterEqual(progress["metrics"]["elapsedSeconds"], 1)
            self.assertIsNone(progress["etaSeconds"])

    def test_video_mapper_limits_recovery_models_and_has_a_time_budget(self) -> None:
        *_unused, mapping = _configure_sfm(180, 8192, True, True)

        self.assertTrue(mapping.multiple_models)
        self.assertEqual(mapping.max_num_models, 3)
        self.assertEqual(mapping.min_model_size, 12)
        self.assertEqual(mapping.ba_global_frames_ratio, 1.25)
        self.assertEqual(mapping.ba_global_points_ratio, 1.25)
        self.assertEqual(mapping.ba_global_max_num_iterations, 50)
        self.assertEqual(mapping.max_runtime_seconds, 225)

    def test_registration_gate_retains_a_dominant_partial_video_model(self) -> None:
        self.assertEqual(_minimum_useful_registration_count(3), 3)
        self.assertEqual(_minimum_useful_registration_count(181), 82)
        self.assertEqual(_minimum_useful_registration_count(239), 108)

    def test_camera_solve_telemetry_tracks_current_and_best_models(self) -> None:
        telemetry = _CameraSolveTelemetry(181)
        telemetry.initial_pair_registered()
        for _ in range(12):
            telemetry.next_image_registered()
        telemetry.initial_pair_registered()
        for _ in range(3):
            telemetry.next_image_registered()

        self.assertEqual(
            telemetry.snapshot(),
            {
                "imageCount": 181,
                "registeredCameras": 5,
                "bestRegisteredCameras": 14,
                "modelAttempts": 2,
            },
        )
        self.assertIn("5/181 registered in model 2", telemetry.detail(65))
        self.assertIn("best 14", telemetry.detail(65))
        self.assertGreater(telemetry.progress(), 0.45)

    def test_video_frames_share_one_camera_per_source(self) -> None:
        groups = _feature_extraction_groups(
            [
                {"image": "video-000002.jpg", "source": "clip-a.mp4"},
                {"image": "photo-000000.jpg", "source": "still-a.jpg"},
                {"image": "video-000001.jpg", "source": "clip-a.mp4"},
                {"image": "video-000003.jpg", "source": "clip-b.mov"},
                {"image": "photo-000001.jpg", "source": "still-b.png"},
            ]
        )

        self.assertEqual(
            groups,
            [
                (["video-000001.jpg", "video-000002.jpg"], True),
                (["video-000003.jpg"], True),
                (["photo-000000.jpg", "photo-000001.jpg"], False),
            ],
        )

    def test_video_intrinsic_spread_detects_per_frame_camera_drift(self) -> None:
        def frame(fx: float, width: int = 2560) -> dict[str, object]:
            return {
                "sourcePath": "locked.mp4",
                "intrinsics": {
                    "width": width,
                    "height": 1440,
                    "fx": fx,
                    "fy": fx,
                    "cx": width / 2,
                    "cy": 720,
                },
            }

        self.assertAlmostEqual(_video_intrinsic_spread([frame(2200), frame(2200)]), 0.0)
        self.assertGreater(_video_intrinsic_spread([frame(1400), frame(2500)]), 0.25)

    def test_streaming_video_selection_is_bounded_and_reports_progress(self) -> None:
        import av

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fixture.mp4"
            images = root / "images"
            images.mkdir()
            container = av.open(str(path), mode="w")
            stream = container.add_stream("mpeg4", rate=12)
            stream.width = 96
            stream.height = 72
            stream.pix_fmt = "yuv420p"
            for index in range(24):
                pixels = np.full((72, 96, 3), index * 9, dtype=np.uint8)
                pixels[:, (index * 3) % 72 : (index * 3) % 72 + 12, :] = 255
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            updates: list[tuple[float, dict[str, object]]] = []

            records, statistics = _extract_video_streaming(
                path,
                images,
                0,
                MediaPreparationOptions(
                    video_fps=12.0,
                    maximum_video_frames=3,
                    maximum_image_dimension=96,
                    minimum_image_dimension=32,
                ),
                3,
                root,
                lambda fraction, _detail, _eta, metrics: updates.append(
                    (fraction, metrics)
                ),
            )

            self.assertGreaterEqual(len(records), 1)
            self.assertLessEqual(len(records), 3)
            self.assertTrue(all((images / record["image"]).is_file() for record in records))
            self.assertEqual(statistics["decodedFrameCount"], 24)
            self.assertEqual(updates[-1][0], 1.0)
            self.assertEqual(updates[-1][1]["selectedFrames"], len(records))

    def test_adaptive_video_selection_keeps_more_views_during_fast_motion(self) -> None:
        import av
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            random = np.random.default_rng(7)
            texture = random.integers(0, 256, (120, 160, 3), dtype=np.uint8)

            def write_video(path: Path, moving: bool) -> None:
                container = av.open(str(path), mode="w")
                stream = container.add_stream("mpeg4", rate=15)
                stream.width = 160
                stream.height = 120
                stream.pix_fmt = "yuv420p"
                for index in range(60):
                    pixels = (
                        cv2.warpAffine(
                            texture,
                            np.asarray(
                                [[1.0, 0.0, float(index * 3)], [0.0, 1.0, 0.0]],
                                dtype=np.float32,
                            ),
                            (160, 120),
                            borderMode=cv2.BORDER_REFLECT,
                        )
                        if moving
                        else texture
                    )
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
                container.close()

            options = MediaPreparationOptions(
                video_fps=15.0,
                maximum_video_frames=100,
                maximum_image_dimension=160,
                minimum_image_dimension=64,
            )
            counts = {}
            for label, moving in (("static", False), ("moving", True)):
                video = root / f"{label}.mp4"
                images = root / label
                images.mkdir()
                write_video(video, moving)
                records, statistics = _extract_video_streaming(
                    video,
                    images,
                    0,
                    options,
                    100,
                    root,
                    lambda *_args: None,
                )
                counts[label] = len(records)
                self.assertEqual(statistics["selectionMode"], "adaptive_optical_flow")

            self.assertGreater(counts["moving"], counts["static"] * 2)

    def test_lingbot_context_keeps_exact_training_views_in_a_smooth_stream(self) -> None:
        import av

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.mp4"
            input_images = root / "selected"
            input_images.mkdir()
            container = av.open(str(video), mode="w")
            stream = container.add_stream("mpeg4", rate=12)
            stream.width = 96
            stream.height = 72
            stream.pix_fmt = "yuv420p"
            for index in range(24):
                pixels = np.full((72, 96, 3), index * 7, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            records = []
            for index, timestamp in enumerate((0.25, 1.25)):
                name = f"selected-{index}.jpg"
                Image.new("RGB", (96, 72), (index * 80, 20, 10)).save(
                    input_images / name
                )
                records.append(
                    {
                        "image": name,
                        "timestampSeconds": timestamp,
                    }
                )

            with patch("scanlan_splat.media._progress"):
                paths, output_indices, statistics = _extract_lingbot_context(
                    video,
                    records,
                    input_images,
                    root / "context",
                    root,
                )

            self.assertGreater(len(paths), len(records))
            self.assertEqual(statistics["trainingViewCount"], 2)
            self.assertEqual(
                [paths[index] for index in output_indices],
                [input_images / "selected-0.jpg", input_images / "selected-1.jpg"],
            )
            self.assertTrue(all(path.is_file() for path in paths))

    def test_hybrid_media_observations_are_immutable_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.zeros((72, 96, 3), dtype=np.uint8)
            pixels[12:60, 20:76] = [220, 80, 30]
            sources = []
            for index in range(3):
                source = root / f"detail-{index}.png"
                Image.fromarray(np.roll(pixels, index * 3, axis=1)).save(source)
                sources.append(source)
            options = MediaPreparationOptions(
                maximum_image_dimension=96,
                minimum_image_dimension=32,
            )

            first = prepare_media_observations(root, sources, options)
            pointer_path = root / "outputs" / "cache" / "media-observations" / "current.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            manifest_path = pointer_path.parent / pointer["path"] / "observations.json"

            self.assertEqual(first["schemaVersion"], 1)
            self.assertEqual(len(first["frames"]), 3)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((manifest_path.parent / first["frames"][0]["image"]).is_file())
            with patch("scanlan_splat.media._collect_images", side_effect=AssertionError("cache miss")):
                second = prepare_media_observations(root, sources, options)
            self.assertEqual(first["fingerprint"], second["fingerprint"])

    def test_camera_analysis_materializes_cached_observations_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observation_root = root / "observations"
            images = observation_root / "images"
            inputs = root / "analysis-inputs"
            images.mkdir(parents=True)
            inputs.mkdir()
            for index in range(3):
                Image.new("RGB", (64, 48), (index * 40, 20, 10)).save(
                    images / f"video-{index:06d}.jpg"
                )
            observations = {
                "frames": [
                    {
                        "image": f"images/video-{index:06d}.jpg",
                        "sourcePath": "clip.mp4",
                        "width": 64,
                        "height": 48,
                        "sharpness": float(index),
                        "timestampSeconds": index / 2,
                    }
                    for index in range(3)
                ],
                "videoSources": [{"path": "clip.mp4", "selectedFrameCount": 3}],
            }

            records, videos = _materialize_observation_inputs(
                observation_root, observations, inputs
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(videos[0]["selectedFrameCount"], 3)
            self.assertTrue(all((inputs / record["image"]).is_file() for record in records))

    def test_feature_settings_invalidate_analysis_but_keep_decoded_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jpg"
            source.write_bytes(b"stable-media")
            lower = MediaPreparationOptions(maximum_features=4_096)
            higher = MediaPreparationOptions(maximum_features=8_192)

            lower_observations = _source_fingerprint([source], lower)
            higher_observations = _source_fingerprint([source], higher)

            self.assertEqual(lower_observations, higher_observations)
            self.assertNotEqual(
                _media_dataset_fingerprint(lower_observations, lower),
                _media_dataset_fingerprint(higher_observations, higher),
            )


if __name__ == "__main__":
    unittest.main()
