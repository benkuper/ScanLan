from __future__ import annotations

import unittest

import torch

from scanlan_splat.depth_loss import masked_robust_depth_loss


class DepthLossTests(unittest.TestCase):
    def test_low_confidence_generated_depth_has_less_influence(self) -> None:
        predicted = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.1, 1.0])
        mask = torch.tensor([True, True])
        unweighted = masked_robust_depth_loss(predicted, target, mask)
        weighted = masked_robust_depth_loss(
            predicted,
            target,
            mask,
            torch.tensor([1.0, 0.1]),
        )
        self.assertLess(float(weighted), float(unweighted))


if __name__ == "__main__":
    unittest.main()
