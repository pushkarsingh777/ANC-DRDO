#!/usr/bin/env python3
"""
Model Training Engine for Causal Conv-TasNet Speech Enhancement (FR-2).

Features:
- Config-driven training via YAML
- Automatic Mixed Precision (AMP FP16) for high throughput on GPU
- Gradient clipping and learning rate scheduling (ReduceLROnPlateau / Cosine)
- Best and latest checkpointing
- TensorBoard experiment tracking
- Smoke-test mode for rapid pipeline validation

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --smoke-test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Ensure root path is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import CausalConvTasNet
from src.dataset import get_dataloaders
from src.losses import CombinedLoss


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    grad_clip: float = 5.0,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_si_snr = 0.0
    count = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for idx, batch in enumerate(pbar):
        if max_batches and idx >= max_batches:
            break
            
        mixture = batch["mixture"].to(device)  # (B, T)
        clean = batch["clean"].to(device)      # (B, T)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                est_speech = model(mixture)
            loss, metrics = criterion(est_speech.float(), clean.float())
                
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            est_speech = model(mixture)
            loss, metrics = criterion(est_speech.float(), clean.float())
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        bs = mixture.shape[0]
        total_loss += loss.item() * bs
        total_si_snr += metrics.get("si_snr_db", -loss.item()) * bs
        count += bs
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "si_snr": f"{metrics.get('si_snr_db', 0.0):.2f}dB"})
        
    return total_loss / max(count, 1), total_si_snr / max(count, 1)


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_si_snr = 0.0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for idx, batch in enumerate(pbar):
            if max_batches and idx >= max_batches:
                break
                
            mixture = batch["mixture"].to(device)
            clean = batch["clean"].to(device)
            
            est_speech = model(mixture)
            loss, metrics = criterion(est_speech, clean)
            
            bs = mixture.shape[0]
            total_loss += loss.item() * bs
            total_si_snr += metrics.get("si_snr_db", -loss.item()) * bs
            count += bs
            
    return total_loss / max(count, 1), total_si_snr / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Causal Conv-TasNet for Speech Enhancement")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on small batch count")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device_str == 'cuda' else 'CPU'})")
    
    manifests_dir = cfg["paths"]["manifests_dir"]
    train_loader, val_loader, _ = get_dataloaders(manifests_dir, cfg)
    print(f"Loaded manifests: Train={len(train_loader.dataset)} items, Val={len(val_loader.dataset)} items")
    
    model = CausalConvTasNet.from_config(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: CausalConvTasNet | Trainable Parameters: {num_params:,} ({num_params * 4 / (1024**2):.2f} MB FP32)")
    
    train_cfg = cfg["training"]
    criterion = CombinedLoss(
        si_snr_weight=train_cfg.get("si_snr_weight", 1.0),
        stft_weight=train_cfg.get("stft_weight", 0.0),
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    
    scheduler_type = train_cfg.get("scheduler", "plateau")
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.get("scheduler_factor", 0.5),
            patience=train_cfg.get("scheduler_patience", 3),
            min_lr=train_cfg.get("min_lr", 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("epochs", 50),
            eta_min=train_cfg.get("min_lr", 1e-6),
        )
        
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)
    
    start_epoch = 1
    best_val_loss = float("inf")
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        
    num_epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
    if args.smoke_test:
        num_epochs = 2
        print("Smoke test mode enabled: running 2 epochs on max 10 batches each.")
        
    max_train_batches = 10 if args.smoke_test else None
    max_val_batches = 5 if args.smoke_test else None
    
    print("\nStarting Training Pipeline...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train SI-SNR':^14} | {'Val Loss':^10} | {'Val SI-SNR':^12} | {'LR':^9} | {'Time':^7}")
    print("-" * 85)
    
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        
        train_loss, train_si_snr = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=train_cfg.get("grad_clip_norm", 5.0),
            max_batches=max_train_batches,
        )
        
        val_loss, val_si_snr = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )
        
        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0
        
        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("SISNR/train", train_si_snr, epoch)
        writer.add_scalar("SISNR/val", val_si_snr, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_si_snr:^11.2f} dB | {val_loss:^10.4f} | {val_si_snr:^9.2f} dB | {current_lr:^9.2e} | {epoch_time:^6.1f}s")
        
        # Save checkpoints
        latest_path = os.path.join(checkpoints_dir, "latest_checkpoint.pt")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "config": cfg,
        }
        torch.save(ckpt_data, latest_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoints_dir, "best_checkpoint.pt")
            torch.save(ckpt_data, best_path)
            
    print("-" * 85)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {os.path.join(checkpoints_dir, 'best_checkpoint.pt')}")
    writer.close()


if __name__ == "__main__":
    main()
