from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from scanlan_splat.media import (
    _descriptor_distance,
    _limited_size,
    _select_video_candidates,
    _video_candidates,
)


class MediaPreparationTests(unittest.TestCase):
    def test_image_limit_preserves_aspect_ratio_without_upscaling(self) -> None:
        self.assertEqual(_limited_size(4000, 2000, 1000), (1000, 500))
        self.assertEqual(_limited_size(640, 480, 1000), (640, 480))

    def test_descriptor_distance_is_exposure_invariant(self) -> None:
        left = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        left /= np.linalg.norm(left)
        self.assertAlmostEqual(_descriptor_distance(left, left), 0.0, places=6)
        self.assertGreater(_descriptor_distance(left, -left), 1.9)

    def test_video_selection_keeps_sharpest_frame_per_time_bucket(self) -> None:
        image = Image.new("RGB", (16, 16))
        a = np.zeros(256, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(256, dtype=np.float32)
        b[1] = 1.0
        candidates = [
            (0.00, image, 0.1, a),
            (0.20, image, 0.9, a),
            (0.55, image, 0.8, b),
            (1.05, image, 0.7, a),
        ]

        selected = _select_video_candidates(candidates, target_fps=2.0, maximum_frames=10)

        self.assertEqual([round(value[0], 2) for value in selected], [0.2, 0.55, 1.05])

    def test_bundled_video_runtime_decodes_frames(self) -> None:
        import av

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.mp4"
            container = av.open(str(path), mode="w")
            stream = container.add_stream("mpeg4", rate=6)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            for index in range(6):
                pixels = np.full((48, 64, 3), index * 35, dtype=np.uint8)
                pixels[:, index * 8 : index * 8 + 8, :] = 255
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            candidates, statistics = _video_candidates(path, target_fps=2.0)

            self.assertEqual(statistics["decodedFrameCount"], 6)
            self.assertGreaterEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
