from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scanlan_geometry import cli
from scanlan_geometry.cli import main


class GeometryWorkerCliTests(unittest.TestCase):
    def test_module_entrypoint_is_declared(self) -> None:
        source = Path(cli.__file__).read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', source)

    def test_da3_commands_route_versioned_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            progress = Path(directory) / "progress.json"
            with patch("scanlan_geometry.cli.run_da3_request") as infer:
                self.assertEqual(
                    main(["infer-da3", "--request", str(request), "--progress", str(progress)]),
                    0,
                )
                infer.assert_called_once_with(request, progress)
            with patch("scanlan_geometry.cli.refine_da3_depth_request") as refine:
                self.assertEqual(
                    main(
                        [
                            "refine-rgbd-depth-da3",
                            "--request",
                            str(request),
                            "--progress",
                            str(progress),
                        ]
                    ),
                    0,
                )
                refine.assert_called_once_with(request, progress)

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

    def test_backend_policy_command_routes_request_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            report = Path(directory) / "report.json"
            with patch(
                "scanlan_geometry.cli.evaluate_backend_policy_file",
                return_value={"schemaVersion": 1, "decisions": {}},
            ) as evaluate:
                result = main(
                    [
                        "backend-policy",
                        "--request",
                        str(request),
                        "--report",
                        str(report),
                    ]
                )
            self.assertEqual(result, 0)
            evaluate.assert_called_once_with(request, report)


if __name__ == "__main__":
    unittest.main()
