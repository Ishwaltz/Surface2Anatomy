import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# Import pipeline components
sharon_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sharon_dir))
sys.path.insert(0, str(sharon_dir.parent))

from labels import NUM_ORGANS, ORGAN_NAMES, get_sex_organ_mask
from model_gnn import SexConditionedOrganGNN
from losses import DynamicMaskedCentroidLoss, MultiTaskLoss
from dataset import PointCloudOrganDataset
from pointcloud_sampler import PointCloudSampler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        pts = batch["points"].to(device)
        ctr = batch["centroids"].to(device)
        sex = batch["sex_prior"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        preds = model(pts, sex, mask=mask)
        loss, _ = criterion(preds, ctr, mask=mask) if isinstance(criterion, MultiTaskLoss) else (criterion(preds, ctr, mask=mask), {})
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    organ_errors = {name: [] for name in ORGAN_NAMES}

    for batch in loader:
        pts = batch["points"].to(device)
        ctr = batch["centroids"].to(device)
        sex = batch["sex_prior"].to(device)
        mask = batch["mask"].to(device)

        preds = model(pts, sex, mask=mask)
        loss, _ = criterion(preds, ctr, mask=mask) if isinstance(criterion, MultiTaskLoss) else (criterion(preds, ctr, mask=mask), {})
        total_loss += loss.item()

        # Euclidean distance error in mm per organ
        dist = torch.norm(preds - ctr, dim=-1)  # (B, 121)
        for b in range(pts.shape[0]):
            for i, name in enumerate(ORGAN_NAMES):
                if mask[b, i] > 0.5:
                    organ_errors[name].append(dist[b, i].item())

    mean_loss = total_loss / max(len(loader), 1)
    mean_organ_err = {}
    all_errs = []
    for name, err_list in organ_errors.items():
        if len(err_list) > 0:
            m_err = sum(err_list) / len(err_list)
            mean_organ_err[name] = m_err
            all_errs.extend(err_list)

    avg_physical_error = sum(all_errs) / max(len(all_errs), 1)
    return mean_loss, avg_physical_error, mean_organ_err


def main():
    parser = argparse.ArgumentParser(description="Metadata-Conditioned Dynamic GNN Training")
    parser.add_argument("--data_file", type=str, default="sharon/dataset/pointclouds.pt", help="Path to point cloud dataset .pt")
    parser.add_argument("--output_dir", type=str, default="sharon/outputs/gnn", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--gnn_layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split ratio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Check if pointclouds.pt exists; if not, sample it
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"[Dataset] {data_path} not found. Sampling from dataset directory...")
        sampler = PointCloudSampler(
            dataset_dir="sharon/dataset",
            metadata_file="sharon/dataset/metadata.json",
            mesh_dir="sharon/outputs/meshes",
        )
        sampler.sample_and_export_dataset(str(data_path))

    full_dataset = PointCloudOrganDataset(pt_path=str(data_path), augment=True)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[Dataset] Total: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = SexConditionedOrganGNN(
        num_organs=NUM_ORGANS,
        pointnet_feat_dim=1024,
        sex_emb_dim=128,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DynamicMaskedCentroidLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print("\n" + "=" * 60)
    print(f" Starting Metadata-Conditioned Dynamic GNN Training (K={NUM_ORGANS})")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_err_mm, organ_errs = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs:03d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean Error: {val_err_mm:.2f} mm")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = out_dir / "best_gnn_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_err_mm": val_err_mm,
                "num_organs": NUM_ORGANS,
            }, str(ckpt_path))
            print(f"  ✓ Saved new best model -> {ckpt_path} (Val Loss: {val_loss:.4f})")

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()
