# ANC Model Training & Evaluation Framework (FR-2)

Implements the **AI/ML Model Training & Evaluation** phase of the Adaptive Noise Cancellation PRD (Problem Statement 26052).

## Architecture

This package provides a strictly causal, waveform-domain **Causal Conv-TasNet** speech enhancement model:
* **Encoder:** 1D Conv with learnable filterbank on raw waveform ($L=32$ samples $= 2\text{ms}$ at 16 kHz, stride $= 16$ samples $= 1\text{ms}$).
* **Separator:** Dilated 1D Depthwise-Separable Temporal Convolutional Network (TCN) blocks ($R=3$ repeats, $X=8$ blocks with dilations $1, 2, 4, \dots, 128$) with Channel-wise Layer Normalization.
* **Decoder:** 1D Transposed Conv reconstructing enhanced speech waveform.
* **Loss:** Scale-Invariant Signal-to-Noise Ratio (SI-SNR) + Multi-resolution STFT loss.
* **Latency:** Zero future lookahead with strictly causal left-padded convolutions.

## Directory Structure

```
anc-model-training/
├── configs/
│   └── train_config.yaml      # All hyperparameters (batch size, LR, loss weights)
├── src/
│   ├── dataset.py             # PyTorch Dataset for CSV manifests + fixed-length slicing
│   ├── losses.py              # SI-SNR / SI-SDR & multi-resolution STFT loss
│   ├── metrics.py             # Delta SNR, STOI, PESQ, and RTF benchmarking
│   └── models/
│       ├── causal_layers.py   # Causal Conv1d, ChannelLayerNorm, CumulativeLayerNorm
│       └── conv_tasnet.py     # Complete Causal Conv-TasNet architecture
├── scripts/
│   ├── train.py               # AMP FP16 training engine with checkpointing & TensorBoard
│   ├── evaluate.py            # Held-out test set evaluation generating KPI metrics report
│   └── export_onnx.py         # ONNX export and runtime numerical parity verification
└── tests/
    ├── test_causality.py      # Validates zero future lookahead
    ├── test_dataset_loader.py # Tests batch collation, shapes, and SNR mixing
    └── test_model_forward.py  # Tests model forward/backward passes and parameter count
```

## Quickstart

### 1. Run Unit Tests
```bash
python -m unittest tests/test_model_forward.py
python -m unittest tests/test_causality.py
python -m unittest tests/test_dataset_loader.py
```

### 2. Smoke Test Training
```bash
python scripts/train.py --config configs/train_config.yaml --smoke-test
```

### 3. Full Training
```bash
python scripts/train.py --config configs/train_config.yaml
```

### 4. Evaluate Checkpoint on Test Set
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_checkpoint.pt
```

### 5. Export to ONNX (for Edge / TensorRT)
```bash
python scripts/export_onnx.py --checkpoint checkpoints/best_checkpoint.pt --output checkpoints/causal_conv_tasnet.onnx
```
