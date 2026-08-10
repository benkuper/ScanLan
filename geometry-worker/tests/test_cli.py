from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scanlan_geometry.cli import main


class GeometryWorkerCliTests(unittest.TestCase):
    def test_mapanything_commands_route_versioned_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            progress = Path(directory) / "progress.json"
            with patch("scanlan_geometry.cli.run_mapanything_request") as infer:
                result = main(
                    ["infer-mapanything", "--request", str(request), "--progress", str(progress)]
                )
            self.assertEqual(result, 0)
            infer.assert_called_once_with(request, progress)
            with patch(
                "scanlan_geometry.cli.refine_mapanything_depth_request"
            ) as refine:
                result = main(
                    [
                        "refine-rgbd-depth-mapanything",
                        "--request",
                        str(request),
                        "--progress",
                        str(progress),
                    ]
                )
            self.assertEqual(result, 0)
            refine.assert_called_once_with(request, progress)

    def test_lingbot_depth_command_routes_the_existing_request_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            progress = Path(directory) / "progress.json"
            with patch("scanlan_geometry.cli.refine_depth_request") as refine:
                result = main(
                    [
                        "refine-rgbd-depth",
                        "--request",
                        str(request),
                        "--progress",
                        str(progress),
                    ]
                )

            self.assertEqual(result, 0)
            refine.assert_called_once_with(request, progress)

    def test_lingbot_map_command_routes_the_versioned_geometry_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            progress = Path(directory) / "progress.json"
            with patch("scanlan_geometry.cli.run_lingbot_map_request") as infer:
                result = main(
                    [
                        "infer-lingbot-map",
                        "--request",
                        str(request),
                        "--progress",
                        str(progress),
                    ]
                )

            self.assertEqual(result, 0)
            infer.assert_called_once_with(request, progress)


if __name__ == "__main__":
    unittest.main()
