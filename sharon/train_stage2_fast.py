import os
import sys
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from model_cascade import LocalPatchRefiner3D
from labels import ORGAN_NAMES, NUM_ORGANS
from train_stage2_refiner import extract_patch_3d


def train_stage2_fast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2 Fast Refiner] Starting in-memory trainer on {device}...")

    cfg = Config()
    splits_file = SHARON_DIR / "splits.json"
    with open(splits_file) as f:
        splits = json.load(f)

    train_cases = set(splits["train_cases"])
    val_cases = set(splits["val_cases"])

    full_ds = CTLocalizationDataset(str(SHARON_DIR / "dataset"), cfg.target_shape, cfg.heatmap_size, augment=False)
    train_indices = [i for i, c in enumerate(full_ds.cases) if c.name in train_cases][:50]
    val_indices = [i for i, c in enumerate(full_ds.cases) if c.name in val_cases][:15]

    print(f"[Stage 2 Fast Refiner] Extracting patches from {len(train_indices)} training CT scans...")
    all_patches = []
    all_deltas = []
    all_organ_ids = []

    for idx in train_indices:
        sample = full_ds[idx]
        img = sample["image"].unsqueeze(0) # (1, 1, 128, 128, 128)
        gt_c = sample["gt_centroids"]      # (117, 3)
        found = sample["found_mask"]       # (117,)

        present_idx = torch.where(found)[0]
        if len(present_idx) == 0:
            continue

        # Extract up to 12 patches per scan
        sel_idx = present_idx[torch.randperm(len(present_idx))[:12]]
        for o_idx in sel_idx:
            c_star = gt_c[o_idx] # (3,)
            # Simulate 10-25 mm Stage 1 error (jitter ~0.08 in normalized 128-voxel space)
            jitter = (torch.randn(3) * 0.07).clamp(-0.15, 0.15)
            c_coarse = (c_star + jitter).clamp(0.05, 0.95)

            patch = extract_patch_3d(img, c_coarse, patch_size=32)
            delta_target = (c_star - c_coarse)

            all_patches.append(patch[0])
            all_deltas.append(delta_target)
            all_organ_ids.append(o_idx)

    train_p = torch.stack(all_patches, dim=0)   # (N, 1, 32, 32, 32)
    train_d = torch.stack(all_deltas, dim=0)    # (N, 3)
    train_o = torch.stack(all_organ_ids, dim=0) # (N,)
    print(f"[Stage 2 Fast Refiner] Pre-cached {len(train_p)} training patches in memory.")

    # Extract validation patches
    val_patches, val_deltas, val_organ_ids = [], [], []
    for idx in val_indices:
        sample = full_ds[idx]
        img = sample["image"].unsqueeze(0)
        gt_c = sample["gt_centroids"]
        found = sample["found_mask"]
        present_idx = torch.where(found)[0]
        if len(present_idx) == 0:
            continue
        for o_idx in present_idx[:8]:
            c_star = gt_c[o_idx]
            jitter = (torch.randn(3) * 0.07).clamp(-0.15, 0.15)
            c_coarse = (c_star + jitter).clamp(0.05, 0.95)
            patch = extract_patch_3d(img, c_coarse, patch_size=32)
            val_patches.append(patch[0])
            val_deltas.append(c_star - c_coarse)
            val_organ_ids.append(o_idx)

    val_p = torch.stack(val_patches, dim=0).to(device)
    val_d = torch.stack(val_deltas, dim=0).to(device)
    val_o = torch.stack(val_organ_ids, dim=0).to(device)

    # Train Model
    train_loader = DataLoader(TensorDataset(train_p, train_d, train_o), batch_size=32, shuffle=True)
    model = LocalPatchRefiner3D(num_organs=NUM_ORGANS, patch_size=32, embed_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    ckpt_dir = SHARON_DIR / "outputs" / "checkpoints" / "stage2_refiner"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"
    best_val = float("inf")

    print("[Stage 2 Fast Refiner] Training for 25 epochs on GPU...")
    t0 = time.time()
    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for pb, db, ob in train_loader:
            pb, db, ob = pb.to(device), db.to(device), ob.to(device)
            optimizer.zero_grad()
            pred = model(pb, ob)
            loss = criterion(pred, db)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(pb)

        scheduler.step()
        train_loss = total_loss / len(train_p)

        model.eval()
        with torch.inference_mode():
            val_pred = model(val_p, val_o)
            val_loss = criterion(val_pred, val_d).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 5 == 0 or epoch == 25:
            print(f"  Epoch {epoch:02d}/25 | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    print(f"[SUCCESS] Trained Stage 2 Refiner in {time.time()-t0:.1f}s! Best weights saved to: {best_ckpt}")


if __name__ == "__main__":
    train_stage2_fast()
