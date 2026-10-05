import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

from labels import (
    NUM_ORGANS, NUM_SKELETAL, NUM_SOFT_TISSUE,
    SKELETAL_CLASSES, SOFT_TISSUE_CLASSES,
    SKELETAL_INDICES, SOFT_TISSUE_INDICES,
    ORGAN_NAMES, get_sex_organ_mask,
    get_skeletal_mask, get_soft_tissue_mask
)
from model_same import SAMeOrganGNN
from losses import SAMeJointLoss
from dataset import PointCloudOrganDataset


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_skel, total_soft = 0.0, 0.0, 0.0

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            out = model(pts, sex, mask=mask)
            loss, l_skel, l_soft = criterion(out, ctr, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_skel += l_skel.item()
        total_soft += l_soft.item()

    n = max(len(loader), 1)
    return total_loss / n, total_skel / n, total_soft / n


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_errs, skel_errs, soft_errs = [], [], []

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)

        out = model(pts, sex, mask=mask)
        loss, _, _ = criterion(out, ctr, mask)
        total_loss += loss.item()

        preds = out["full_coords"]
        # DE-NORMALIZE TO PHYSICAL MILLIMETERS
        preds_mm = preds * scale.unsqueeze(1) + center.unsqueeze(1)
        ctr_mm = ctr * scale.unsqueeze(1) + center.unsqueeze(1)
        dist_mm = torch.norm(preds_mm - ctr_mm, dim=-1) # (B, 121)

        for b in range(pts.shape[0]):
            for i, name in enumerate(ORGAN_NAMES):
                if mask[b, i] > 0.5:
                    e = dist_mm[b, i].item()
                    all_errs.append(e)
                    if i in SKELETAL_INDICES:
                        skel_errs.append(e)
                    else:
                        soft_errs.append(e)

    mean_loss = total_loss / max(len(loader), 1)
    overall_mean = float(np.mean(all_errs)) if all_errs else 0.0
    overall_sd = float(np.std(all_errs)) if all_errs else 0.0
    overall_median = float(np.median(all_errs)) if all_errs else 0.0
    skel_mean = float(np.mean(skel_errs)) if skel_errs else 0.0
    soft_mean = float(np.mean(soft_errs)) if soft_errs else 0.0

    return mean_loss, overall_mean, overall_sd, overall_median, skel_mean, soft_mean


def main():
    parser = argparse.ArgumentParser(description="SAMe Skeleton-Conditioned Probabilistic GNN Training")
    parser.add_argument("--data_file", type=str, default="sharon/dataset/pointclouds_450.pt", help="Dataset path")
    parser.add_argument("--splits_file", type=str, default="sharon/outputs/splits_pointcloud.json", help="Splits file")
    parser.add_argument("--output_dir", type=str, default="sharon/outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for SAMe Training")

    data_path = Path(args.data_file)
    print(f"[Dataset] Loading {data_path}...")
    full_ds = PointCloudOrganDataset(pt_path=str(data_path), augment=True)

    with open(args.splits_file) as f:
        splits = json.load(f)

    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    train_sub = Subset(full_ds, train_idx)
    val_sub = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    model = SAMeOrganGNN(pointnet_feat_dim=1024, sex_emb_dim=128, latent_dim=512, hidden_dim=256, gnn_layers=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = SAMeJointLoss(beta=0.01, lambda_skel=1.0, lambda_soft=0.5)

    out_path = Path(args.output_dir)
    ckpt_dir = out_path / "checkpoints"
    logs_dir = out_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = logs_dir / "same_model_training_log.csv"
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "skel_loss", "soft_nll", "val_loss", "val_mean_mm", "val_sd_mm", "skel_mean_mm", "soft_mean_mm", "lr", "epoch_time_s"])

    best_val_err = float("inf")
    best_stats = {}

    print("\n" + "=" * 75)
    print(f" TRAINING SAMe ARCHITECTURE (SKELETON-CONDITIONED PROBABILISTIC GNN)")
    print(f" Skeletal Classes: {NUM_SKELETAL} | Soft-Tissue Classes: {NUM_SOFT_TISSUE} | Epochs: {args.epochs}")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_skel, tr_soft = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_mean, val_sd, val_med, val_skel, val_soft = evaluate_metrics(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        with open(csv_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_skel:.4f}", f"{tr_soft:.4f}", f"{val_loss:.4f}", f"{val_mean:.2f}", f"{val_sd:.2f}", f"{val_skel:.2f}", f"{val_soft:.2f}", f"{cur_lr:.6f}", f"{elapsed:.1f}"])

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs or val_mean < best_val_err:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} [{elapsed:.1f}s] - Train: {tr_loss:.4f} (Skel: {tr_skel:.4f}, NLL: {tr_soft:.4f}) | Val Error: {val_mean:.2f} ± {val_sd:.2f} mm (Skel: {val_skel:.2f} mm, Soft: {val_soft:.2f} mm, Med: {val_med:.2f} mm)")

        if val_mean < best_val_err:
            best_val_err = val_mean
            best_stats = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mean_error_mm": val_mean,
                "val_sd_error_mm": val_sd,
                "val_median_error_mm": val_med,
                "val_skel_mean_mm": val_skel,
                "val_soft_mean_mm": val_soft,
            }
            ckpt_file = ckpt_dir / "same_model_best.pth"
            torch.save({
                "model_name": "same_model",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_stats": best_stats,
                "num_organs": NUM_ORGANS,
                "num_skeletal": NUM_SKELETAL,
                "num_soft": NUM_SOFT_TISSUE,
            }, str(ckpt_file))

    print(f"\n✓ SAMe Model Best Validation Physical Error: {best_stats.get('val_mean_error_mm', 0.0):.2f} ± {best_stats.get('val_sd_error_mm', 0.0):.2f} mm at Epoch {best_stats.get('epoch', 0)}")


if __name__ == "__main__":
    main()
