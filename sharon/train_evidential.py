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
    NUM_ORGANS, ORGAN_NAMES, get_sex_organ_mask
)
from model_evidential import EvidentialOrganGNN
from losses import EvidentialJointLoss
from dataset import PointCloudOrganDataset


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss, total_edl, total_nll, total_bio = 0.0, 0.0, 0.0, 0.0

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            out = model(pts, sex, mask=mask)
            loss, l_edl, l_nll, l_bio = criterion(out, ctr, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_edl += l_edl.item()
        total_nll += l_nll.item()
        total_bio += l_bio.item()

    n = max(len(loader), 1)
    return total_loss / n, total_edl / n, total_nll / n, total_bio / n


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_errs = []
    all_ale_sd = []
    all_epi_sd = []

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)

        out = model(pts, sex, mask=mask)
        loss, _, _, _ = criterion(out, ctr, mask)
        total_loss += loss.item()

        preds = out["gamma"]
        ale_var = out["aleatoric_var"]  # (B, 121, 3)
        epi_var = out["epistemic_var"]  # (B, 121, 3)

        # DE-NORMALIZE TO PHYSICAL MILLIMETERS
        preds_mm = preds * scale.unsqueeze(1) + center.unsqueeze(1)
        ctr_mm = ctr * scale.unsqueeze(1) + center.unsqueeze(1)
        dist_mm = torch.norm(preds_mm - ctr_mm, dim=-1) # (B, 121)

        # De-normalize uncertainties: sqrt(mean(var_k * scale_k^2))
        scale_sq = (scale ** 2).unsqueeze(1) # (B, 1, 3)
        phys_ale_sd = torch.sqrt(torch.mean(ale_var * scale_sq, dim=-1)) # (B, 121) in mm
        phys_epi_sd = torch.sqrt(torch.mean(epi_var * scale_sq, dim=-1)) # (B, 121) in mm

        for b in range(pts.shape[0]):
            for i in range(NUM_ORGANS):
                if mask[b, i] > 0.5:
                    all_errs.append(dist_mm[b, i].item())
                    all_ale_sd.append(phys_ale_sd[b, i].item())
                    all_epi_sd.append(phys_epi_sd[b, i].item())

    mean_loss = total_loss / max(len(loader), 1)
    overall_mean = float(np.mean(all_errs)) if all_errs else 0.0
    overall_sd = float(np.std(all_errs)) if all_errs else 0.0
    overall_median = float(np.median(all_errs)) if all_errs else 0.0
    mean_ale = float(np.mean(all_ale_sd)) if all_ale_sd else 0.0
    mean_epi = float(np.mean(all_epi_sd)) if all_epi_sd else 0.0

    return mean_loss, overall_mean, overall_sd, overall_median, mean_ale, mean_epi


def main():
    parser = argparse.ArgumentParser(description="Evidential Dynamic GNN & Biomechanical Collision Training")
    parser.add_argument("--data_file", type=str, default="sharon/dataset/pointclouds_450.pt", help="Dataset path")
    parser.add_argument("--splits_file", type=str, default="sharon/outputs/splits_pointcloud.json", help="Splits file")
    parser.add_argument("--output_dir", type=str, default="sharon/outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="EDL regularizer weight")
    parser.add_argument("--lambda_bio", type=float, default=5.0, help="Biomechanical collision penalty weight")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for Evidential GNN Training")

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

    model = EvidentialOrganGNN(num_organs=NUM_ORGANS, pointnet_feat_dim=1024, sex_emb_dim=128, latent_dim=512, hidden_dim=256, gnn_layers=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = EvidentialJointLoss(lambda_reg=args.lambda_reg, lambda_bio=args.lambda_bio)

    out_path = Path(args.output_dir)
    ckpt_dir = out_path / "checkpoints"
    logs_dir = out_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / "evidential_model_best.pth"
    start_epoch = 1
    best_val_err = float("inf")
    best_stats = {}

    if args.resume and ckpt_file.exists():
        ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        start_epoch = ckpt_data.get("epoch", 1) + 1
        best_stats = ckpt_data.get("best_stats", {})
        best_val_err = best_stats.get("val_mean_error_mm", float("inf"))
        print(f"[Resume] Loaded checkpoint from {ckpt_file}. Resuming from epoch {start_epoch} (Best: {best_val_err:.2f} mm)")

    csv_log_path = logs_dir / "evidential_model_training_log.csv"
    if not args.resume or not csv_log_path.exists():
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "edl_loss", "nll_loss", "bio_loss", "val_loss", "val_mean_mm", "val_sd_mm", "val_ale_mm", "val_epi_mm", "lr", "epoch_time_s"])

    print("\n" + "=" * 80)
    print(f" TRAINING EVIDENTIAL DYNAMIC GNN & BIOMECHANICAL NON-INTERPENETRATION MODEL")
    print(f" Organs: {NUM_ORGANS} | Lambda EDL Reg: {args.lambda_reg} | Lambda Bio Collision: {args.lambda_bio} | Epochs: {start_epoch}..{args.epochs}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_edl, tr_nll, tr_bio = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_mean, val_sd, val_med, val_ale, val_epi = evaluate_metrics(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        with open(csv_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_edl:.4f}", f"{tr_nll:.4f}", f"{tr_bio:.4f}", f"{val_loss:.4f}", f"{val_mean:.2f}", f"{val_sd:.2f}", f"{val_ale:.2f}", f"{val_epi:.2f}", f"{cur_lr:.6f}", f"{elapsed:.1f}"])

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs or val_mean < best_val_err:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} [{elapsed:.1f}s] - Train: {tr_loss:.4f} (NLL: {tr_nll:.3f}, Bio: {tr_bio:.4f}) | Val: {val_mean:.2f} ± {val_sd:.2f} mm (Median: {val_med:.2f} mm, Aleatoric: {val_ale:.2f} mm, Epistemic: {val_epi:.2f} mm)")

        if val_mean < best_val_err:
            best_val_err = val_mean
            best_stats = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mean_error_mm": val_mean,
                "val_sd_error_mm": val_sd,
                "val_median_error_mm": val_med,
                "val_aleatoric_mm": val_ale,
                "val_epistemic_mm": val_epi,
            }
            ckpt_file = ckpt_dir / "evidential_model_best.pth"
            torch.save({
                "model_name": "evidential_model",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_stats": best_stats,
                "num_organs": NUM_ORGANS,
            }, str(ckpt_file))

    print(f"\n✓ Evidential GNN Best Validation Physical Error: {best_stats.get('val_mean_error_mm', 0.0):.2f} ± {best_stats.get('val_sd_error_mm', 0.0):.2f} mm at Epoch {best_stats.get('epoch', 0)}")


if __name__ == "__main__":
    main()
