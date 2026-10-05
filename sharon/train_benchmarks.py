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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import pipeline components
sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

from labels import (
    NUM_ORGANS, NUM_REGIONS, ORGAN_NAMES, TOTAL_CLASSES_121,
    FEMALE_SPECIFIC_INDICES, MALE_SPECIFIC_INDICES,
    get_sex_organ_mask, get_organ_region_indices
)
from model_gnn import HierarchicalSexConditionedGNN
from models_baseline import PointNet2DirectRegressor, DGCNNRegressor
from losses import HierarchicalDynamicMaskedLoss
from dataset import PointCloudOrganDataset


def get_model(model_name: str, num_organs: int = 121, device: torch.device = None) -> nn.Module:
    name = model_name.lower()
    if name in ("sex_gnn", "gnn", "proposed", "hierarchical_gnn"):
        model = HierarchicalSexConditionedGNN(
            num_organs=num_organs,
            num_regions=NUM_REGIONS,
            pointnet_feat_dim=1024,
            sex_emb_dim=128,
            hidden_dim=256,
            gnn_layers=4,
        )
    elif name in ("pointnet2", "pointnet++", "pointnet"):
        model = PointNet2DirectRegressor(num_organs=num_organs)
    elif name in ("dgcnn", "edgeconv"):
        model = DGCNNRegressor(num_organs=num_organs, k=20)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model.to(device)


def build_stratified_splits(
    dataset_file: str,
    output_split_file: str = "sharon/outputs/splits_pointcloud.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    data = torch.load(dataset_file, map_location="cpu", weights_only=False)
    sex_vals = data["sex_vals"].numpy() if "sex_vals" in data else np.argmax(data["sex_priors"].numpy(), axis=-1)
    indices = np.arange(len(sex_vals))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, sex_vals, test_size=(val_ratio + test_ratio), random_state=seed, stratify=sex_vals
    )

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_cases": [data["case_ids"][i] for i in train_idx],
        "val_cases": [data["case_ids"][i] for i in val_idx],
        "test_cases": [data["case_ids"][i] for i in test_idx],
        "total_cases": len(indices),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
    }

    out_p = Path(output_split_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"[Splits] Saved 80/10/10 stratified split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} -> {out_p}")
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    is_hierarchical: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        reg_gt = batch["regional_gt"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            if is_hierarchical:
                preds, pred_reg = model(pts, sex, mask=mask, return_regional=True)
                loss = criterion(preds, ctr, mask=mask, pred_regions=pred_reg, gt_regions=reg_gt)
            else:
                preds = model(pts, sex, mask=mask)
                loss = criterion(preds, ctr, mask=mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_hierarchical: bool = True,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    organ_errors = {name: [] for name in ORGAN_NAMES}

    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        ctr = batch["centroids"].to(device, non_blocking=True)
        reg_gt = batch["regional_gt"].to(device, non_blocking=True)
        sex = batch["sex_prior"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)

        if is_hierarchical:
            preds, pred_reg = model(pts, sex, mask=mask, return_regional=True)
            loss = criterion(preds, ctr, mask=mask, pred_regions=pred_reg, gt_regions=reg_gt)
        else:
            preds = model(pts, sex, mask=mask)
            loss = criterion(preds, ctr, mask=mask)

        total_loss += loss.item()

        # DE-NORMALIZE TO PHYSICAL MILLIMETERS
        # preds: (B, 121, 3), scale: (B, 3) -> scale.unsqueeze(1): (B, 1, 3)
        preds_mm = preds * scale.unsqueeze(1) + center.unsqueeze(1)
        ctr_mm = ctr * scale.unsqueeze(1) + center.unsqueeze(1)

        dist_mm = torch.norm(preds_mm - ctr_mm, dim=-1) # (B, 121) in physical mm

        for b in range(pts.shape[0]):
            for i, name in enumerate(ORGAN_NAMES):
                if mask[b, i] > 0.5:
                    organ_errors[name].append(dist_mm[b, i].item())

    mean_loss = total_loss / max(len(loader), 1)
    all_errs = []
    mean_per_organ = {}
    for name, errs in organ_errors.items():
        if len(errs) > 0:
            m_val = float(np.mean(errs))
            mean_per_organ[name] = m_val
            all_errs.extend(errs)

    overall_mean = float(np.mean(all_errs)) if len(all_errs) > 0 else 0.0
    overall_sd = float(np.std(all_errs)) if len(all_errs) > 0 else 0.0
    overall_median = float(np.median(all_errs)) if len(all_errs) > 0 else 0.0

    return mean_loss, overall_mean, overall_sd, overall_median, mean_per_organ


def train_single_model(
    model_name: str,
    full_dataset: PointCloudOrganDataset,
    train_idx: List[int],
    val_idx: List[int],
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    device: torch.device = None,
) -> Dict:
    is_hierarchical = model_name.lower() in ("sex_gnn", "gnn", "proposed", "hierarchical_gnn")
    print(f"\n{'='*70}")
    print(f" TRAINING MODEL: {model_name.upper()} (Hierarchical: {is_hierarchical}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr})")
    print(f"{'='*70}")

    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    model = get_model(model_name, num_organs=NUM_ORGANS, device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = HierarchicalDynamicMaskedLoss(beta=0.01, lambda_regional=1.0, lambda_offset=2.0)

    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    logs_dir = out_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = logs_dir / f"{model_name}_training_log.csv"
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mean_error_mm", "val_sd_error_mm", "lr", "epoch_time_s"])

    best_val_err = float("inf")
    best_stats = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, is_hierarchical=is_hierarchical)
        val_loss, val_mean, val_sd, val_med, organ_errs = evaluate_metrics(model, val_loader, criterion, device, is_hierarchical=is_hierarchical)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        with open(csv_log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_mean:.2f}", f"{val_sd:.2f}", f"{cur_lr:.6f}", f"{elapsed:.1f}"])

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or val_mean < best_val_err:
            print(f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] - Train: {train_loss:.4f} | Val: {val_loss:.4f} | Physical Error: {val_mean:.2f} ± {val_sd:.2f} mm (Median: {val_med:.2f} mm)")

        if val_mean < best_val_err:
            best_val_err = val_mean
            best_stats = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mean_error_mm": val_mean,
                "val_sd_error_mm": val_sd,
                "val_median_error_mm": val_med,
                "organ_errors": organ_errs,
            }
            ckpt_file = ckpt_dir / f"{model_name}_best.pth"
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_stats": best_stats,
                "num_organs": NUM_ORGANS,
                "is_hierarchical": is_hierarchical,
            }, str(ckpt_file))

    print(f"✓ {model_name.upper()} Best Validation Physical Error: {best_stats.get('val_mean_error_mm', 0.0):.2f} ± {best_stats.get('val_sd_error_mm', 0.0):.2f} mm at Epoch {best_stats.get('epoch', 0)}")
    return best_stats


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Training Suite with Unit Normalization")
    parser.add_argument("--data_file", type=str, default="sharon/dataset/pointclouds_450.pt", help="Path to point cloud dataset")
    parser.add_argument("--output_dir", type=str, default="sharon/outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--models", nargs="+", default=["pointnet2", "dgcnn", "sex_gnn"], help="Models to train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for Benchmark Training Suite")

    data_path = Path(args.data_file)
    print(f"[Dataset] Loading {data_path}...")
    full_ds = PointCloudOrganDataset(pt_path=str(data_path), augment=True)

    split_file = Path(args.output_dir) / "splits_pointcloud.json"
    train_idx, val_idx, test_idx = build_stratified_splits(str(data_path), output_split_file=str(split_file))

    results = {}
    for m_name in args.models:
        stats = train_single_model(
            model_name=m_name,
            full_dataset=full_ds,
            train_idx=train_idx,
            val_idx=val_idx,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        results[m_name] = stats

    print("\n" + "=" * 70)
    print(" ALL BENCHMARK MODELS TRAINED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
