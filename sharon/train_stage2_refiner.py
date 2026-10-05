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
from torch.utils.data import DataLoader, Subset

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from model_cascade import LocalPatchRefiner3D
from labels import ORGAN_NAMES, NUM_ORGANS


def extract_patch_3d(img: torch.Tensor, center_norm: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """
    Extracts a 3D patch of shape (1, patch_size, patch_size, patch_size) from img (1, D, H, W)
    centered around center_norm (z, y, x) in [0, 1].
    """
    D, H, W = img.shape[-3:]
    cz = int(round(center_norm[0].item() * D))
    cy = int(round(center_norm[1].item() * H))
    cx = int(round(center_norm[2].item() * W))

    pad = patch_size // 2
    padded = nn.functional.pad(img, (pad, pad, pad, pad, pad, pad), mode="replicate")

    # Shift indices due to padding
    sz, sy, sx = cz, cy, cx
    ez, ey, ex = sz + patch_size, sy + patch_size, sx + patch_size

    patch = padded[:, :, sz:ez, sy:ey, sx:ex]
    return patch


def train_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2 Refiner] Starting training on {device}...")

    cfg = Config()
    splits_file = SHARON_DIR / "splits.json"
    with open(splits_file) as f:
        splits = json.load(f)

    train_cases = set(splits["train_cases"])
    val_cases = set(splits["val_cases"])

    full_ds = CTLocalizationDataset(str(SHARON_DIR / "dataset"), cfg.target_shape, cfg.heatmap_size, augment=False)
    train_indices = [i for i, c in enumerate(full_ds.cases) if c.name in train_cases]
    val_indices = [i for i, c in enumerate(full_ds.cases) if c.name in val_cases]

    train_sub = Subset(full_ds, train_indices)
    val_sub = Subset(full_ds, val_indices)

    model = LocalPatchRefiner3D(num_organs=NUM_ORGANS, patch_size=32, embed_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    ckpt_dir = SHARON_DIR / "outputs" / "checkpoints" / "stage2_refiner"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pth"

    epochs = 15
    print(f"[Stage 2 Refiner] Training for {epochs} epochs over {len(train_sub)} scans...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0

        for s_idx in range(len(train_sub)):
            sample = train_sub[s_idx]
            img = sample["image"].unsqueeze(0).to(device) # (1, 1, 128, 128, 128)
            gt_centroids = sample["gt_centroids"].to(device) # (117, 3)
            found_mask = sample["found_mask"].to(device)     # (117,)

            present_organs = torch.where(found_mask)[0]
            if len(present_organs) == 0:
                continue

            # Pick up to 8 random present organs per scan for fast convergence
            sel_organs = present_organs[torch.randperm(len(present_organs))[:8]]

            patches = []
            target_deltas = []
            organ_ids = []

            for o_idx in sel_organs:
                gt_c = gt_centroids[o_idx] # (3,) in [0, 1]
                # Simulate Stage 1 coarse prediction error with realistic Gaussian jitter (std ~ 0.08 in normalized space)
                jitter = (torch.randn(3, device=device) * 0.06).clamp(-0.15, 0.15)
                coarse_c = (gt_c + jitter).clamp(0.05, 0.95)

                patch = extract_patch_3d(img, coarse_c, patch_size=32)
                delta_target = (gt_c - coarse_c) # Ground truth correction vector

                patches.append(patch[0]) # (1, 32, 32, 32)
                target_deltas.append(delta_target)
                organ_ids.append(o_idx)

            patch_batch = torch.stack(patches, dim=0) # (B, 1, 32, 32, 32)
            target_batch = torch.stack(target_deltas, dim=0) # (B, 3)
            organ_id_batch = torch.stack(organ_ids, dim=0)   # (B,)

            optimizer.zero_grad()
            pred_deltas = model(patch_batch, organ_id_batch)
            loss = criterion(pred_deltas, target_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(sel_organs)
            n_samples += len(sel_organs)

            del img, patch_batch, target_batch, organ_id_batch

        scheduler.step()
        train_loss /= max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.inference_mode():
            for s_idx in range(min(len(val_sub), 20)): # Quick val pass
                sample = val_sub[s_idx]
                img = sample["image"].unsqueeze(0).to(device)
                gt_centroids = sample["gt_centroids"].to(device)
                found_mask = sample["found_mask"].to(device)

                present_organs = torch.where(found_mask)[0]
                if len(present_organs) == 0:
                    continue

                sel_organs = present_organs[:6]
                patches, target_deltas, organ_ids = [], [], []

                for o_idx in sel_organs:
                    gt_c = gt_centroids[o_idx]
                    jitter = (torch.randn(3, device=device) * 0.06).clamp(-0.15, 0.15)
                    coarse_c = (gt_c + jitter).clamp(0.05, 0.95)
                    patch = extract_patch_3d(img, coarse_c, patch_size=32)
                    patches.append(patch[0])
                    target_deltas.append(gt_c - coarse_c)
                    organ_ids.append(o_idx)

                p_b = torch.stack(patches, dim=0)
                t_b = torch.stack(target_deltas, dim=0)
                o_b = torch.stack(organ_ids, dim=0)

                pred_d = model(p_b, o_b)
                v_l = criterion(pred_d, t_b)
                val_loss += v_l.item() * len(sel_organs)
                val_samples += len(sel_organs)

                del img, p_b, t_b, o_b

        val_loss /= max(val_samples, 1)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train SmoothL1: {train_loss:.5f} | Val SmoothL1: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    ⭐ Saved new best Stage 2 Refiner weights -> {best_ckpt_path}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Stage 2 Refiner training complete! Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    train_stage2()
