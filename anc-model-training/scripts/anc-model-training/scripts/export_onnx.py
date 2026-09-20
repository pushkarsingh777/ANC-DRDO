#!/usr/bin/env python3
"""
ONNX Export and Numerical Verification Script (FR-4.1).

Exports a trained CausalConvTasNet model to ONNX format with dynamic sequence length,
and verifies output parity against ONNX Runtime.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best_checkpoint.pt --output checkpoints/causal_conv_tasnet.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml
# pyrefly: ignore [missing-import]
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CausalConvTasNet


def main():
    parser = argparse.ArgumentParser(description="Export Causal Conv-TasNet to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pt", help="Path to checkpoint .pt")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="checkpoints/causal_conv_tasnet.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    args = parser.parse_args()

    # Load config and model
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ckpt = None

    model = CausalConvTasNet.from_config(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print("Using uninitialized model for export.")
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Dummy input: 2.0s audio at 16kHz
    dummy_input = torch.randn(1, 32000, dtype=torch.float32)

    print(f"Exporting model to {args.output} with opset {args.opset}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["mixture"],
        output_names=["enhanced_speech"],
        dynamic_axes={
            "mixture": {0: "batch_size", 1: "time_samples"},
            "enhanced_speech": {0: "batch_size", 1: "time_samples"},
        },
    )
    print(f"Model exported successfully. File size: {os.path.getsize(args.output) / (1024**2):.2f} MB")

    # Verify parity with onnxruntime
    try:
        # pyrefly: ignore [missing-import]
        import onnxruntime as ort
        print("\nVerifying parity with ONNX Runtime...")
        session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        
        test_audio = np.random.randn(1, 48000).astype(np.float32)  # test 3.0s variable length
        with torch.no_grad():
            torch_out = model(torch.from_numpy(test_audio)).numpy()
            
        ort_inputs = {session.get_inputs()[0].name: test_audio}
        ort_out = session.run(None, ort_inputs)[0]
        
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"Max absolute difference: {max_diff:.6e}")
        if max_diff < 1e-4:
            print("Parity verification: PASSED (PyTorch and ONNX Runtime outputs match).")
        else:
            print("WARNING: Discrepancy observed between PyTorch and ONNX Runtime outputs.")
    except ImportError:
        print("onnxruntime not installed. Skipping numerical verification.")


if __name__ == "__main__":
    main()
