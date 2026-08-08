from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scanlan_splat.dataset import load_dataset


def _write_dataset(root: Path, *, schema: int = 3, model: str = "pinhole", distortion: list[float] | None = None) -> None:
    root.mkdir(parents=True)
    for relative in (
        "initialization.ply",
        "images/000000.jpg",
        "depths/000000.png",
        "masks/000000.png",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    frame = {
        "intrinsics": {
            "width": 8,
            "height": 6,
            "fx": 7.0,
            "fy": 7.0,
            "cx": 3.5,
            "cy": 2.5,
            "model": model,
            "distortion": [] if distortion is None else distortion,
        },
        "worldFromRgbCamera": [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ],
        "image": "images/000000.jpg",
        "depth": "depths/000000.png",
        "depthMask": "masks/000000.png",
    }
    if schema == 4:
        frame.update(metricAnchor=True, poseConfidence=1.0, sourceType="rgbd")
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schemaVersion": schema,
                "metric": True,
                "initialization": "initialization.ply",
                "frames": [frame],
            }
        ),
        encoding="utf-8",
    )


class DatasetTests(unittest.TestCase):
    def test_schema_three_pinhole_dataset_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset(root)

            resolved, dataset = load_dataset(root)

            self.assertEqual(resolved, root.resolve())
            self.assertEqual(dataset["schemaVersion"], 3)

    def test_distorted_training_frames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset(root, model="opencv_rational", distortion=[0.0] * 8)

            with self.assertRaisesRegex(ValueError, "not undistorted"):
                load_dataset(root)

    def test_previous_dataset_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset(root, schema=2)

            with self.assertRaisesRegex(ValueError, "schema 3"):
                load_dataset(root)

    def test_registered_rgb_only_dataset_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset(root)
            manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            manifest["metric"] = False
            manifest["frames"][0].pop("depth")
            manifest["frames"][0].pop("depthMask")
            (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

            _, dataset = load_dataset(root)

            self.assertFalse(dataset["metric"])

    def test_schema_four_allows_localized_media_without_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset(root, schema=4)
            manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            media = dict(manifest["frames"][0])
            media.pop("depth")
            media.pop("depthMask")
            media.update(
                metricAnchor=False,
                poseConfidence=0.8,
                sourceType="high_quality_media",
            )
            manifest["frames"].append(media)
            manifest["sourceMode"] = "hybrid"
            (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

            _, dataset = load_dataset(root)

            self.assertEqual(dataset["schemaVersion"], 4)
            self.assertEqual(len(dataset["frames"]), 2)


if __name__ == "__main__":
    unittest.main()
