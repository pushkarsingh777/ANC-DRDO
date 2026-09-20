"""
Strict Causality & Zero Lookahead Unit Test (FR-2, FR-3).

Verifies that the model has zero future lookahead:
If two audio inputs are identical up to sample index T_split, their model outputs
must be mathematically identical up to T_split regardless of future input values.
"""
import os
import sys
import unittest
# pyrefly: ignore [missing-import]
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CausalConvTasNet


class TestModelCausality(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CausalConvTasNet(
            n_filters=128,
            encoder_kernel_size=32,
            encoder_stride=16,
            tcn_channels=64,
            tcn_hidden_channels=256,
            kernel_size=3,
            n_blocks_per_repeat=4,
            n_repeats=2,
            norm_type="ChLN",
        )
        self.model.eval()

    def test_zero_future_lookahead(self):
        total_samples = 32000
        split_point = 16000  # 1.0s split point

        # Signal A: Random audio
        x_a = torch.randn(1, total_samples)

        # Signal B: Identical to A up to split_point, completely different noise afterwards
        x_b = x_a.clone()
        x_b[:, split_point:] = torch.randn(1, total_samples - split_point) * 5.0 + 3.0

        with torch.no_grad():
            out_a = self.model(x_a)
            out_b = self.model(x_b)

        # Calculate max discrepancy in the first half
        # Due to causal left padding and causal dilated convolutions,
        # the output up to split_point must NOT be affected by x[split_point:]
        diff = torch.abs(out_a[:, :split_point] - out_b[:, :split_point])
        max_diff = torch.max(diff).item()

        print(f"\n[Causality Test] Max difference before split point: {max_diff:.6e}")
        self.assertLess(
            max_diff,
            1e-5,
            f"Future lookahead detected! Output before split point changed by {max_diff:.6e}",
        )


if __name__ == "__main__":
    unittest.main()
