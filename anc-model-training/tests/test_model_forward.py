"""
Unit tests for Causal Conv-TasNet Forward/Backward & Shapes (FR-2).
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
from src.losses import SISNRLoss, CombinedLoss


class TestModelForward(unittest.TestCase):
    def setUp(self):
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

    def test_parameter_budget(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n[Test] Model parameter count: {num_params:,}")
        self.assertLess(num_params, 5_000_000, "Parameter count exceeds 5M budget")

    def test_forward_shape(self):
        batch_size = 2
        time_samples = 32000  # 2.0s at 16kHz
        x = torch.randn(batch_size, time_samples)
        
        out = self.model(x)
        self.assertEqual(out.shape, (batch_size, time_samples), "Output shape mismatch")
        self.assertTrue(torch.all(torch.isfinite(out)), "Output contains NaN or Inf")

    def test_backward_pass(self):
        x = torch.randn(2, 16000, requires_grad=False)
        target = torch.randn(2, 16000)
        
        self.model.train()
        out = self.model(x)
        
        criterion = SISNRLoss()
        loss = criterion(out, target)
        loss.backward()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")


if __name__ == "__main__":
    unittest.main()
