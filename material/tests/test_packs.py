from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scanlan_material.packs import (
    MODEL_CANDIDATES,
    ModelCandidate,
    ModelPack,
    resolve_model_pack,
    write_pack_manifest,
)


class ModelPackTests(unittest.TestCase):
    def test_commercial_pack_excludes_every_restricted_candidate(self) -> None:
        commercial = resolve_model_pack(ModelPack.COMMERCIAL)
        research = resolve_model_pack(ModelPack.RESEARCH)
        self.assertTrue(commercial)
        self.assertLess(len(commercial), len(research))
        self.assertTrue(all(value.commercial_use for value in commercial))
        self.assertTrue(all(not value.output_restricted for value in commercial))
        self.assertIn("rgb-to-x", {value.identifier for value in research})
        self.assertNotIn("rgb-to-x", {value.identifier for value in commercial})

    def test_unverified_model_terms_cannot_enter_commercial_pack(self) -> None:
        invalid = ModelCandidate(
            "invalid",
            ("material",),
            "https://example.test/source",
            "a" * 40,
            "https://example.test/model",
            "b" * 40,
            "MIT",
            "UNVERIFIED",
            True,
            False,
            "bakeoff",
            "fixture",
        )
        with self.assertRaisesRegex(ValueError, "unverified terms"):
            resolve_model_pack(ModelPack.COMMERCIAL, [invalid])

    def test_manifest_surfaces_research_output_restrictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_pack_manifest(
                Path(temporary) / "model-pack.json", ModelPack.RESEARCH
            )
        self.assertFalse(manifest["commercialUse"])
        self.assertTrue(manifest["outputRestrictions"])
        self.assertEqual(len(manifest["models"]), len(MODEL_CANDIDATES))


if __name__ == "__main__":
    unittest.main()
