from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from scanlan.depth_refinement import (
    _close_memmaps,
    _metric_gate,
    _sensor_anchor_calibration,
    _true_rgb_coverage,
    validate_predictions,
)
from scanlan.dataset import build_posed_dataset
from scanlan.io import CameraModel, FrameRecord, PhaseData, RgbCameraModel
from scanlan.mesh import PosedFrame


def _posed_sequence(root: Path) -> tuple[list[PosedFrame], np.ndarray]:
    width = height = 32
    camera = CameraModel(width, height, 40.0, 40.0, 15.5, 15.5, 1000.0, 8.0)
    rgb_camera = RgbCameraModel(
        width, height, 40.0, 40.0, 15.5, 15.5, "pinhole", ()
    )
    records: list[FrameRecord] = []
    (root / "phase.json").write_text('{"id":"phase"}\n', encoding="utf-8")
    (root / "frames.csv").write_text("index\n0\n1\n2\n", encoding="utf-8")
    raw_template = np.full((height, width), 2000, dtype="<u2")
    raw_template[12:20, 12:20] = 0
    # Even a weak/isolated sensor measurement remains immutable.
    raw_template[16, 16] = 1900
    # A nonzero sensor return beyond the working range is not a completion hole.
    raw_template[13, 13] = 9000
    for index in range(3):
        depth_path = root / f"depth-{index}.u16"
        color_path = root / f"color-{index}.rgb"
        raw_template.tofile(depth_path)
        np.full((height, width, 3), 128, dtype=np.uint8).tofile(color_path)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = (index - 1) * 0.03
        records.append(
            FrameRecord(index, index, index * 33_333, depth_path, color_path, None, None, pose)
        )
    phase = PhaseData(
        root,
        {"id": "phase", "name": "test", "sensor": {"kind": "test"}},
        camera,
        rgb_camera,
        np.eye(4, dtype=np.float64),
        records,
        [],
    )
    frames = [
        PosedFrame(
            "test",
            "phase",
            phase,
            index,
            np.asarray(record.pose),
            (1.0, 1.0, 1.0),
            False,
        )
        for index, record in enumerate(records)
    ]
    return frames, raw_template


class DepthRefinementTests(unittest.TestCase):
    def test_sensor_anchor_calibration_must_pass_held_out_depth(self) -> None:
        height, width = 128, 160
        yy, xx = np.indices((height, width), dtype=np.float32)
        measured = 2.0 + 0.002 * xx + 0.001 * yy
        predicted = measured + 0.08 + 0.02 * np.sin(xx / 18.0)
        reliable = np.ones((height, width), dtype=bool)
        model_valid = np.ones((height, width), dtype=bool)
        calibrated, valid, report = _sensor_anchor_calibration(
            measured,
            reliable,
            predicted,
            model_valid,
        )
        self.assertTrue(report["accepted"], report)
        self.assertGreater(report["heldOutSampleCount"], 256)
        self.assertTrue(np.isfinite(calibrated[valid]).all())
        self.assertLess(report["medianResidualMm"], 10.0)

    def test_rgb_coverage_ignores_nonfinite_model_depth_without_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frames, raw = _posed_sequence(Path(temporary))
            prediction = np.full(raw.shape, 2.0, dtype=np.float32)
            prediction[0, 0] = np.nan

            with np.errstate(all="raise"):
                coverage = _true_rgb_coverage(frames[0], prediction)

            self.assertFalse(coverage[0, 0])
            self.assertTrue(coverage[16, 16])

    def test_prediction_memmaps_are_closed_before_cache_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "published"
            source.mkdir()
            prediction_path = source / "prediction.npy"
            np.save(prediction_path, np.ones((8, 8), dtype=np.float32))
            prediction = np.load(prediction_path, allow_pickle=False, mmap_mode="r")

            _close_memmaps([prediction])
            source.replace(destination)

            self.assertTrue((destination / "prediction.npy").is_file())

    def test_metric_gate_rejects_a_depth_scale_change(self) -> None:
        measured = np.full((24, 24), 2.0, dtype=np.float32)
        valid = np.ones(measured.shape, dtype=bool)
        accepted = _metric_gate(measured, valid, measured + 0.01, valid)
        rejected = _metric_gate(measured, valid, measured * 1.12, valid)
        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertGreater(rejected["scaleBiasPercent"], 10.0)

    def test_guarded_completion_preserves_sensor_pixels_and_requires_other_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames, raw = _posed_sequence(root)
            predictions = [np.full(raw.shape, 2.0, dtype=np.float32) for _ in frames]
            masks = [np.ones(raw.shape, dtype=bool) for _ in frames]

            overrides, report, _manifest = validate_predictions(
                frames, predictions, masks, root / "validated"
            )

            self.assertFalse((root / "validated" / "validation-state").exists())
            self.assertEqual(report["acceptedFrameCount"], 3)
            self.assertGreater(report["generatedPixelCount"], 30)
            self.assertEqual(report["generatedFusionWeight"], 0.5)
            for frame in frames:
                override = overrides[f"phase:{frame.frame_index}"]
                measured = np.fromfile(override.measured_depth_path, dtype="<u2").reshape(raw.shape)
                refined = np.fromfile(override.refined_depth_path, dtype="<u2").reshape(raw.shape)
                generated = np.fromfile(override.generated_mask_path, dtype=np.uint8).reshape(raw.shape)
                confidence = np.fromfile(override.confidence_path, dtype=np.uint8).reshape(raw.shape)
                np.testing.assert_array_equal(measured, raw)
                np.testing.assert_array_equal(refined[raw > 0], raw[raw > 0])
                self.assertEqual(refined[16, 16], 1900)
                self.assertEqual(generated[16, 16], 0)
                self.assertEqual(confidence[16, 16], 255)
                self.assertEqual(refined[13, 13], 9000)
                self.assertEqual(generated[13, 13], 0)
                self.assertTrue(np.any((raw == 0) & (refined == 2000) & (generated == 1)))

            refined_frames = []
            for frame in frames:
                override = overrides[f"phase:{frame.frame_index}"]
                refined_frames.append(
                    replace(
                        frame,
                        measured_depth_path=override.measured_depth_path,
                        refined_depth_path=override.refined_depth_path,
                        generated_depth_mask_path=override.generated_mask_path,
                        depth_confidence_path=override.confidence_path,
                        depth_refinement_metrics=override.metrics,
                    )
                )
            dataset = build_posed_dataset(root / "cache", refined_frames)
            dataset_root = root / "cache" / "datasets" / dataset["fingerprint"]
            record = dataset["frames"][0]
            self.assertEqual(record["depthProvenance"], "measured+lingbot-v0.5")
            with Image.open(dataset_root / record["depthConfidence"]) as image:
                confidence_image = np.asarray(image)
            self.assertTrue(np.any(confidence_image == 96))
            self.assertTrue((dataset_root / record["generatedDepthMask"]).is_file())

    def test_repeated_identical_viewpoints_do_not_confirm_generated_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames, raw = _posed_sequence(root)
            frames = [
                PosedFrame(
                    frame.phase_name,
                    frame.phase_id,
                    frame.source,
                    frame.frame_index,
                    np.eye(4),
                    frame.display_axes,
                    frame.image_y_up,
                )
                for frame in frames
            ]
            predictions = [np.full(raw.shape, 2.0, dtype=np.float32) for _ in frames]
            masks = [np.ones(raw.shape, dtype=bool) for _ in frames]
            _overrides, report, _manifest = validate_predictions(
                frames, predictions, masks, root / "validated"
            )
            self.assertEqual(report["generatedPixelCount"], 0)


if __name__ == "__main__":
    unittest.main()
