from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scanlan.backend_policy import _source_profile, select_depth_backend


class BackendPolicyIntegrationTests(unittest.TestCase):
    def test_source_profile_preserves_sensor_and_measured_shape(self) -> None:
        phases = [
            SimpleNamespace(
                frames=[object(), object(), object()],
                camera=SimpleNamespace(width=1024, height=768),
            )
        ]
        profile = _source_profile(
            {"settings": {"sensorKind": "femto_mega"}, "mediaSources": []},
            phases,
        )
        self.assertEqual(profile["kind"], "rgbd")
        self.assertEqual(profile["sensorKind"], "femto_mega")
        self.assertEqual(profile["frameCount"], 3)
        self.assertEqual(profile["maximumImageDimension"], 1024)

    def test_policy_worker_result_selects_supported_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            executable = root / "geometry.exe"
            executable.touch()
            phases = [
                SimpleNamespace(
                    frames=[object()] * 20,
                    camera=SimpleNamespace(width=640, height=576),
                )
            ]

            def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                report = Path(command[command.index("--report") + 1])
                report.write_text(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "decisions": {"depthCompletion": {"selected": "mapanything"}},
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("scanlan.backend_policy.subprocess.run", side_effect=run):
                backend, report = select_depth_backend(
                    root,
                    {"settings": {"sensorKind": "femto_mega"}},
                    phases,
                    executable,
                )
            self.assertEqual(backend, "mapanything")
            self.assertEqual(
                report["decisions"]["depthCompletion"]["selected"], "mapanything"
            )

    def test_policy_failure_is_an_atomic_off_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            executable = root / "geometry.exe"
            executable.touch()
            with patch(
                "scanlan.backend_policy.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "", "runtime failed"),
            ):
                backend, report = select_depth_backend(
                    root,
                    {"settings": {"sensorKind": "femto_mega"}},
                    [],
                    executable,
                )
            self.assertEqual(backend, "off")
            self.assertEqual(
                report["decisions"]["depthCompletion"]["selectionMode"],
                "protected-baseline",
            )
            persisted = json.loads(
                (root / "outputs" / "backend-policy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["decisions"]["depthCompletion"]["selected"], "off")


if __name__ == "__main__":
    unittest.main()
