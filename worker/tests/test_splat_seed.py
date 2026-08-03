from __future__ import annotations

import unittest

import numpy as np

from scanlan.splat_seed import GaussianSeeds, adaptive_quadtree_cells, compact_seed_batches


class SplatSeedTests(unittest.TestCase):
    def test_quadtree_keeps_flat_regions_compact_and_splits_visual_detail(self) -> None:
        valid = np.ones((32, 32), dtype=bool)
        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        checker = np.indices((32, 32)).sum(axis=0) % 2
        detailed = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)

        flat_cells = adaptive_quadtree_cells(flat, valid, max_cell_size=64)
        detailed_cells = adaptive_quadtree_cells(detailed, valid, max_cell_size=64)

        self.assertEqual(flat_cells, [(0, 0, 32, 32)])
        self.assertGreater(len(detailed_cells), len(flat_cells))

    def test_quadtree_does_not_bridge_depth_mask_boundaries(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        valid = np.ones((32, 32), dtype=bool)
        valid[:, 16:] = False

        cells = adaptive_quadtree_cells(image, valid, max_cell_size=64)

        self.assertGreater(len(cells), 1)
        self.assertTrue(all(x0 < 16 for x0, _y0, _x1, _y1 in cells))

    def test_compaction_keeps_the_finest_seed_per_spatial_voxel(self) -> None:
        identity = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        coarse = GaussianSeeds(
            points=np.asarray([[0.001, 0.001, 0.001]], dtype=np.float32),
            colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
            scales=np.asarray([[0.05, 0.05, 0.001]], dtype=np.float32),
            quaternions=identity,
        )
        fine = GaussianSeeds(
            points=np.asarray([[0.002, 0.002, 0.002]], dtype=np.float32),
            colors=np.asarray([[0, 255, 0]], dtype=np.uint8),
            scales=np.asarray([[0.01, 0.01, 0.001]], dtype=np.float32),
            quaternions=identity,
        )

        compact = compact_seed_batches([coarse, fine], voxel_size_m=0.01)

        self.assertEqual(len(compact.points), 1)
        self.assertEqual(compact.colors[0].tolist(), [0, 255, 0])


if __name__ == "__main__":
    unittest.main()
