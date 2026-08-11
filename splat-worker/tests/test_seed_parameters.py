from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scanlan_splat.train import (
    FRAME_REUSE_PER_LOAD,
    MAX_METRIC_ITERATIONS,
    RGBD_SURFACE_OPACITY,
    RGBD_SURFACE_SCALE_MULTIPLIER,
    MAXIMUM_PRODUCTION_L1,
    MINIMUM_PRODUCTION_PSNR_DB,
    MINIMUM_PRODUCTION_SSIM,
    _cache_local_frame_order,
    _exponential_lr_gamma,
    _finish_training_step,
    _frame_tensors,
    _metric_surface_scale_limit,
    _photometric_quality_accepted,
    _prepare_dense_seed_scales,
    _read_seed_parameters,
    _reset_opacity_if_due,
    _rgbd_gaussian_limit,
    _ssim,
    _source_resolution_crop,
    _source_resolution_frame_order,
    _source_resolution_start_step,
    _uses_source_resolution,
    _training_frame_order,
    _training_limits,
    _update_smoothed_loss,
)


def _write_initialization(path: Path) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = np.asarray(
        [(1.0, 2.0, 3.0, 255, 128, 0)],
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    with path.open("wb") as handle:
        handle.write(header)
        vertex.tofile(handle)


class SeedParameterTests(unittest.TestCase):
    def test_direct_gaussian_prior_preserves_predicted_anisotropic_axis(self) -> None:
        scales = np.asarray([[0.001, 0.08, 0.12]], dtype=np.float32)
        direct = _prepare_dense_seed_scales(
            scales, 0.10, direct_gaussian_prior=True
        )
        depth = _prepare_dense_seed_scales(
            scales, 0.10, direct_gaussian_prior=False
        )

        np.testing.assert_allclose(direct, [[0.001, 0.08, 0.10]])
        self.assertAlmostEqual(float(depth[0, 2]), 0.00008, places=7)

    def test_training_limits_leave_12_gib_kernel_headroom(self) -> None:
        self.assertEqual(_training_limits(8.0), (720, 1_000_000))
        self.assertEqual(_training_limits(12.0), (960, 2_000_000))
        self.assertEqual(_training_limits(16.0), (1280, 3_000_000))
        self.assertEqual(_training_limits(24.0), (1600, 4_000_000))

    def test_smoothed_loss_dampens_single_frame_variation(self) -> None:
        smoothed = _update_smoothed_loss(None, 0.3)
        self.assertEqual(smoothed, 0.3)
        self.assertAlmostEqual(_update_smoothed_loss(smoothed, 0.5), 0.301)

    def test_production_photometric_gate_rejects_divergent_splats(self) -> None:
        self.assertTrue(
            _photometric_quality_accepted(
                {
                    "medianPsnrDb": MINIMUM_PRODUCTION_PSNR_DB,
                    "medianSsim": MINIMUM_PRODUCTION_SSIM,
                    "medianL1": MAXIMUM_PRODUCTION_L1,
                }
            )
        )
        self.assertFalse(
            _photometric_quality_accepted(
                {"medianPsnrDb": 12.0, "medianSsim": 0.4, "medianL1": 0.25}
            )
        )

    def test_metric_rgbd_training_is_bounded_by_seed_density_and_scale(self) -> None:
        self.assertEqual(_rgbd_gaussian_limit(350_000, 3_000_000), 1_050_000)
        self.assertEqual(_rgbd_gaussian_limit(100_000, 3_000_000), 500_000)
        scales = np.asarray(
            [[0.02, 0.03, 0.001], [0.08, 0.10, 0.004]],
            dtype=np.float32,
        )
        self.assertAlmostEqual(
            _metric_surface_scale_limit(4.0, scales) or 0.0,
            0.20,
            places=6,
        )
        self.assertIsNone(_metric_surface_scale_limit(4.0, None))
        self.assertEqual(RGBD_SURFACE_SCALE_MULTIPLIER, 1.3)
        self.assertEqual(RGBD_SURFACE_OPACITY, 0.45)
        self.assertEqual(MAX_METRIC_ITERATIONS, 2_000)

    def test_position_learning_rate_decays_to_one_percent(self) -> None:
        gamma = _exponential_lr_gamma(30_000)
        self.assertAlmostEqual(gamma**30_000, 0.01, places=10)

    def test_rgb_ssim_ignores_pixels_without_registered_depth(self) -> None:
        import torch

        predicted = torch.zeros((9, 9, 3), dtype=torch.float32)
        target = torch.ones((9, 9, 3), dtype=torch.float32)
        mask = torch.zeros((9, 9), dtype=torch.bool)
        mask[3:6, 3:6] = True
        target[mask] = 0.0

        self.assertAlmostEqual(float(_ssim(predicted, target, mask)), 1.0, places=6)
        self.assertLess(float(_ssim(predicted, target)), 0.5)

    def test_frame_order_reuses_only_views_that_fit_the_host_cache(self) -> None:
        order = _cache_local_frame_order(11, epoch=3, cache_size=4)

        np.testing.assert_array_equal(
            np.bincount(order, minlength=11),
            np.full(11, FRAME_REUSE_PER_LOAD),
        )
        for start in range(0, len(order), 4 * FRAME_REUSE_PER_LOAD):
            self.assertLessEqual(
                len(np.unique(order[start : start + 4 * FRAME_REUSE_PER_LOAD])),
                4,
            )

    def test_hybrid_frame_order_balances_metric_and_media_views(self) -> None:
        frames = [
            *({"depth": "depth.png"} for _ in range(2)),
            *({"image": "media.jpg"} for _ in range(6)),
        ]

        order = _training_frame_order(frames, epoch=2, cache_size=4)
        metric_exposures = int(np.count_nonzero(order < 2))
        media_exposures = len(order) - metric_exposures

        self.assertEqual(metric_exposures, media_exposures)

    def test_optimizer_steps_finish_before_densification_replaces_parameters(self) -> None:
        events: list[tuple[str, object | None]] = []

        class RecordingScaler:
            def step(self, optimizer: object) -> None:
                events.append(("optimizer", optimizer))

            def update(self) -> None:
                events.append(("scaler", None))

        class RecordingStrategy:
            def step_post_backward(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs
                events.append(("densification", None))

        gaussian_optimizer = object()
        pose_optimizer = object()
        strategy = RecordingStrategy()
        parameters = object()
        strategy_state: dict[str, object] = {}
        info: dict[str, object] = {}

        _finish_training_step(
            RecordingScaler(),
            {"means": gaussian_optimizer},
            pose_optimizer,
            True,
            strategy,
            parameters,
            strategy_state,
            600,
            info,
        )

        self.assertEqual(
            events,
            [
                ("optimizer", gaussian_optimizer),
                ("optimizer", pose_optimizer),
                ("scaler", None),
                ("densification", None),
            ],
        )
        self.assertEqual(
            strategy.args[:5],
            (parameters, {"means": gaussian_optimizer}, strategy_state, 600, info),
        )
        self.assertEqual(strategy.kwargs, {"packed": True})

    def test_opacity_reset_runs_on_the_configured_interval(self) -> None:
        class Strategy:
            reset_every = 3_000
            prune_opa = 0.005

        calls: list[dict[str, object]] = []
        parameters = object()
        optimizers = {"opacities": object()}
        state = {"grad2d": object()}

        self.assertFalse(
            _reset_opacity_if_due(
                Strategy(), parameters, optimizers, state, 2_999,
                reset=lambda **kwargs: calls.append(kwargs),
            )
        )
        self.assertTrue(
            _reset_opacity_if_due(
                Strategy(), parameters, optimizers, state, 3_000,
                reset=lambda **kwargs: calls.append(kwargs),
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["params"], parameters)
        self.assertEqual(calls[0]["value"], 0.01)

    def test_rgbd_sidecar_supplies_surface_scales_and_rotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_initialization(root / "initialization.ply")
            np.savez(
                root / "initialization-2dgs.npz",
                points=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                colors=np.asarray([[255, 128, 0]], dtype=np.uint8),
                scales=np.asarray([[0.02, 0.03, 0.001]], dtype=np.float32),
                quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )

            points, colors, scales, quaternions, confidence, opacity = _read_seed_parameters(
                root,
                {
                    "initialization": "initialization.ply",
                    "initializationParameters": "initialization-2dgs.npz",
                },
            )

            self.assertTrue(np.allclose(points, [[1.0, 2.0, 3.0]]))
            self.assertTrue(np.allclose(colors, [[1.0, 128.0 / 255.0, 0.0]]))
            self.assertTrue(np.allclose(scales, [[0.02, 0.03, 0.001]]))
            self.assertTrue(np.allclose(quaternions, [[1.0, 0.0, 0.0, 0.0]]))
            self.assertTrue(np.allclose(confidence, [1.0]))
            self.assertIsNone(opacity)

    def test_source_resolution_tiles_cover_the_calibrated_image(self) -> None:
        frame = {
            "frameIndex": 0,
            "intrinsics": {"width": 2560, "height": 1440},
        }
        crops = [_source_resolution_crop(frame, sample, 960) for sample in range(6)]

        self.assertEqual(
            crops,
            [
                (0, 0, 960, 960),
                (800, 0, 960, 960),
                (1600, 0, 960, 960),
                (0, 480, 960, 960),
                (800, 480, 960, 960),
                (1600, 480, 960, 960),
            ],
        )
        start = _source_resolution_start_step([frame], 30, 960)
        self.assertEqual(start, 24)
        self.assertFalse(_uses_source_resolution(23, start))
        self.assertTrue(_uses_source_resolution(24, start))
        order = _source_resolution_frame_order(
            [
                frame,
                {"frameIndex": 1, "intrinsics": {"width": 800, "height": 600}},
            ],
            960,
        )
        np.testing.assert_array_equal(np.bincount(order), [6, 1])

    def test_source_crop_preserves_focal_length_and_shifts_principal_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.zeros((6, 8, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(8, dtype=np.uint8)
            Image.fromarray(pixels).save(root / "frame.png")
            frame = {
                "frameIndex": 0,
                "image": "frame.png",
                "intrinsics": {
                    "width": 8,
                    "height": 6,
                    "fx": 7.0,
                    "fy": 7.0,
                    "cx": 3.5,
                    "cy": 2.5,
                },
                "worldFromRgbCamera": np.eye(4).reshape(-1).tolist(),
            }

            tensors = _frame_tensors(root, frame, 4, (4, 0, 4, 4))

            self.assertEqual((tensors["width"], tensors["height"]), (4, 4))
            self.assertEqual(tensors["sourceCrop"], (4, 0, 4, 4))
            self.assertAlmostEqual(float(tensors["K"][0, 0]), 7.0)
            self.assertAlmostEqual(float(tensors["K"][0, 2]), -0.5)
            self.assertEqual(int(tensors["rgb"][0, 0, 0]), 4)


if __name__ == "__main__":
    unittest.main()
