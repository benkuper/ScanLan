from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scanlan_splat.train import _load_checkpoint_or_quarantine, _write_checkpoint_atomic


class CheckpointTests(unittest.TestCase):
    def test_unreadable_checkpoint_is_preserved_and_training_can_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "splat-checkpoint.pt"
            path.write_bytes(b"corrupt checkpoint")

            def unreadable(_path: Path) -> object:
                raise RuntimeError("invalid archive")

            checkpoint, quarantine = _load_checkpoint_or_quarantine(path, unreadable)

            self.assertIsNone(checkpoint)
            self.assertIsNotNone(quarantine)
            self.assertFalse(path.exists())
            self.assertEqual(quarantine.read_bytes(), b"corrupt checkpoint")

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "splat-checkpoint.pt"
            path.write_bytes(b"previous checkpoint")

            def interrupted_save(_value: object, destination: Path) -> None:
                destination.write_bytes(b"partial replacement")
                raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _write_checkpoint_atomic(path, {}, interrupted_save)

            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_completed_checkpoint_write_atomically_replaces_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "splat-checkpoint.pt"
            path.write_bytes(b"previous checkpoint")

            def completed_save(value: bytes, destination: Path) -> None:
                destination.write_bytes(value)

            _write_checkpoint_atomic(path, b"new checkpoint", completed_save)

            self.assertEqual(path.read_bytes(), b"new checkpoint")
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
