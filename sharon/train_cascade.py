import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from config import Config
from dataset import CTLocalizationDataset
from model import Modular3DOrganPredictor
from model_cascade import LocalPatchRefiner3D
from losses import soft_argmax
from labels import NUM_ORGANS
from utils.metrics import calculate_physical_error, OrganErrorAccumulator


def extract_3d_patch(image: torch.Tensor, center_norm: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """
    Extracts a 3D patch of spatial size (patch_size, patch_size, patch_size) from full volume.
    image: (1, D, H, W)
    center_norm: (3,) in range [0, 1] (z, y, x)
    returns: (1, patch_size, patch_size, patch_size)
    """
    _, D, H, W = image.shape
    cz = int(torch.round(center_norm[0] * D).item())
    cy = int(torch.round(center_norm[1] * H).item())
    cx = int(torch.round(center_norm[2] * W).item())

    half = patch_size // 2
    z_min, z_max = cz - half, cz + half
    y_min, y_max = cy - half, cy + half
    x_min, x_max = cx - half, cx + half

    pad_d = max(0, -z_min, z_max - D)
    pad_h = max(0, -y_min, y_max - H)
    pad_w = max(0, -x_min, x_max - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        padded = F.pad(image.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="replicate").squeeze(0)
        cz += pad_d
        cy += pad_h
        cx += pad_w
        patch = padded[:, cz - half:cz + half, cy - half:cy + half, cx - half:cx + half]
    else:
        patch = image[:, z_min:z_max, y_min:y_max, x_min:x_max]

    return patch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Fast Cascade Refiner] Training on {device}...")

    cfg = Config()
    splits_file = SHARON_DIR / "splits.json"
    with open(splits_file) as f:
        splits = json.load(f)

    train_cases = set(splits["train_cases"])
    val_cases = set(splits["val_cases"])

    full_ds = CTLocalizationDataset(str(SHARON_DIR / "dataset"), cfg.target_shape, cfg.heatmap_size, augment=False)
    train_indices = [i for i, c in enumerate(full_ds.cases) if c.name in train_cases]
    val_indices = [i for i, c in enumerate(full_ds.cases) if c.name in val_cases]

    print(f"Dataset: Train = {len(train_indices)} scans | Val = {len(val_indices)} scans")

    # 1. Load Pretrained Stage 1 ResNet50
    stage1_model = Modular3DOrganPredictor(
        num_organs=cfg.num_organs,
        heatmap_size=cfg.heatmap_size,
        freeze_blocks=0,
        dropout=0.0,
        backbone="resnet50",
    )
    ckpt_path = SHARON_DIR / "outputs" / "checkpoints" / "resnet50" / "best_model.pth"
    print(f"[Cascade] Loading Stage 1 ResNet50 from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stage1_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    stage1_model.to(device)
    stage1_model.eval()

    # Pre-extract patch samples from training cases (Fast 1-pass extraction)
    print(f"[Cascade] Pre-extracting training patches across 308 cases...")
    t0 = time.time()
    all_patches = []
    all_targets = []
    all_organs = []

    train_sub = torch.utils.data.Subset(full_ds, train_indices)
    train_loader = DataLoader(train_sub, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in train_loader:
            img = batch["image"].to(device)
            gt_centroids = batch["gt_centroids"].to(device)
            found_mask = batch["found_mask"].to(device)

            stage1_hm = stage1_model(img)
            coarse_c = soft_argmax(stage1_hm, temperature=1000.0)

            present = torch.where(found_mask[0])[0]
            for org_idx in present:
                p_c = coarse_c[0, org_idx]
                gt_c = gt_centroids[0, org_idx]
                delta = gt_c - p_c # (3,)

                # Only include patches within reasonable search radius (<20mm in voxel units)
                if torch.norm(delta) < 0.20:
                    patch = extract_3d_patch(img[0].cpu(), p_c.cpu(), patch_size=32)
                    all_patches.append(patch)
                    all_targets.append(delta.cpu())
                    all_organs.append(org_idx.cpu())

    print(f"[Cascade] Extracted {len(all_patches)} high-quality training patches in {time.time()-t0:.2f}s!")

    patch_tensor = torch.stack(all_patches, dim=0) # (N_p, 1, 32, 32, 32)
    target_tensor = torch.stack(all_targets, dim=0) # (N_p, 3)
    organ_tensor = torch.tensor(all_organs, dtype=torch.long) # (N_p,)

    patch_dataset = TensorDataset(patch_tensor, target_tensor, organ_tensor)
    patch_loader = DataLoader(patch_dataset, batch_size=64, shuffle=True, pin_memory=True)

    # 2. Train Stage 2 Patch Refiner
    refiner = LocalPatchRefiner3D(num_organs=cfg.num_organs, patch_size=32).to(device)
    optimizer = AdamW(refiner.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    print(f"\n==================================================")
    print(f" Fast Training Stage 2 Patch Refiner (30 Epochs)")
    print(f"==================================================")

    for epoch in range(1, 31):
        refiner.train()
        losses = []
        for p_b, t_b, o_b in patch_loader:
            p_b, t_b, o_b = p_b.to(device), t_b.to(device), o_b.to(device)
            optimizer.zero_grad()
            pred_delta = refiner(p_b, o_b)
            loss = criterion(pred_delta, t_b)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        if epoch % 5 == 0 or epoch == 30:
            print(f"Epoch {epoch:02d}/30 | Patch SmoothL1 Loss = {np.mean(losses):.6f}")

    # 3. Validation on Unseen Validation Scans
    print(f"\n[Cascade] Evaluating Stage 1 + Stage 2 Cascade on 66 Validation Scans...")
    refiner.eval()
    val_sub = torch.utils.data.Subset(full_ds, val_indices)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=2)

    accumulator_stage1 = OrganErrorAccumulator(num_organs=cfg.num_organs)
    accumulator_cascade = OrganErrorAccumulator(num_organs=cfg.num_organs)

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            gt_centroids = batch["gt_centroids"].to(device)
            found_mask = batch["found_mask"].to(device)
            spacing = batch["spacing"].to(device)

            stage1_hm = stage1_model(img)
            coarse_c = soft_argmax(stage1_hm, temperature=1000.0)

            # Stage 1 Error
            err_stage1 = calculate_physical_error(coarse_c, gt_centroids, spacing, tuple(cfg.target_shape))
            accumulator_stage1.update(err_stage1, found_mask)

            # Stage 2 Refinement
            refined_c = coarse_c.clone()
            present = torch.where(found_mask[0])[0]
            if len(present) > 0:
                patches = [extract_3d_patch(img[0], coarse_c[0, o], patch_size=32) for o in present]
                patch_t = torch.stack(patches, dim=0).to(device)
                org_t = present.clone().to(device)
                pred_delta = refiner(patch_t, org_t)
                refined_c[0, present] += pred_delta

            # Cascade Error
            err_cascade = calculate_physical_error(refined_c, gt_centroids, spacing, tuple(cfg.target_shape))
            accumulator_cascade.update(err_cascade, found_mask)

    err_s1 = accumulator_stage1.overall()
    err_cas = accumulator_cascade.overall()

    print(f"\n==================================================")
    print(f" Stage 1 Baseline Error (ResNet50): {err_s1:.2f} mm")
    print(f" Stage 2 Cascade Refined Error:     {err_cas:.2f} mm")
    print(f" Physical Error Reduction:         {err_s1 - err_cas:+.2f} mm")
    print(f"==================================================")

    out_ckpt_dir = SHARON_DIR / "outputs" / "checkpoints" / "cascade_refiner"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": 30,
        "model_state_dict": refiner.state_dict(),
        "stage1_err_mm": err_s1,
        "cascade_err_mm": err_cas,
    }, out_ckpt_dir / "best_model.pth")
    print(f"[SUCCESS] Saved Cascade Refiner checkpoint to {out_ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
