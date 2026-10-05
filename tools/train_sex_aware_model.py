#!/usr/bin/env python3
"""
tools/train_sex_aware_model.py
==============================
Trains a 3-seed ensemble (Seeds 42, 43, 44) of SAMeOrganGNN on sharon/dataset/pointclouds_450.pt
incorporating biological sex conditioning for all 121 organs:
- Slot 21: Male Prostate
- Slot 117: Female Uterus
- Slot 118: Female Left Ovary
- Slot 119: Female Right Ovary
- Slot 120: Female Vagina

Also trains a PointNet geometric sex classifier to auto-detect biological sex
directly from 3D surface point clouds.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_same import SAMeOrganGNN
from sharon.losses import SAMeJointLoss
from sharon.dataset import PointCloudOrganDataset
from sharon.labels import (
    NUM_ORGANS, ORGAN_NAMES, MALE_SPECIFIC_INDICES, FEMALE_SPECIFIC_INDICES,
    get_sex_organ_mask
)
from sharon.model_gnn import PointNet2Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = repo_root / "experiments" / "sex_aware" / "checkpoints"
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. PointNet Geometric Biological Sex Classifier
# -------------------------------------------------------------
class SurfaceSexClassifier(nn.Module):
    """
    Predicts biological sex P(Female) vs P(Male) directly from 4096 surface points.
    Leverages pelvic dimorphism (subpubic angle, iliac flare, pelvic inlet width).
    """
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder(in_channel=3, out_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) # [logit_female, logit_male]
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(pts)
        return self.classifier(feat)

def train_sex_classifier(ds: Dataset, epochs: int = 30) -> SurfaceSexClassifier:
    print("\n--- Training Geometric Biological Sex Classifier ---")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = SurfaceSexClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            pts = batch["points"].to(device)
            labels = batch["sex_val"].to(device) # 0 = F, 1 = M

            optimizer.zero_grad()
            logits = model(pts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pts.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += pts.size(0)

        acc = (correct / total) * 100.0
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"Sex Classifier Epoch {ep+1:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Accuracy: {acc:.2f}%")

    ckpt_path = out_dir / "S2A_SexClassifier.pt"
    torch.save({"model_state_dict": model.state_dict(), "accuracy": acc}, ckpt_path)
    print(f"Saved Sex Classifier -> {ckpt_path.name} (Accuracy: {acc:.1f}%)")
    return model

# -------------------------------------------------------------
# 2. Train 3-Seed Sex-Conditioned 121-Organ SAMe Ensemble
# -------------------------------------------------------------
def train_seed(seed: int, ds: Dataset, epochs: int = 30):
    print(f"\n=======================================================")
    print(f"Training S2A-Net Sex-Aware Ensemble: Seed {seed} ({epochs} epochs)")
    print(f"=======================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = SAMeOrganGNN(
        pointnet_feat_dim=1024,
        sex_emb_dim=128,
        latent_dim=512,
        hidden_dim=256,
        gnn_layers=4
    ).to(device)

    # Initialize from pre-trained weights if available for faster convergence
    init_ckpt = repo_root / "sharon" / "outputs" / "checkpoints" / "same_model_best.pth"
    if init_ckpt.exists():
        try:
            saved = torch.load(init_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"], strict=False)
            print(f"Initialized Seed {seed} from existing baseline weights {init_ckpt.name}")
        except Exception as e:
            print(f"Init notice: {e}")

    criterion = SAMeJointLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            pts = batch["points"].to(device)
            ctr = batch["centroids"].to(device)
            sex = batch["sex_prior"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            out = model(pts, sex, mask=mask)
            loss, _, _ = criterion(out, ctr, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * pts.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(ds)

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"Seed {seed} | Epoch {ep+1:02d}/{epochs:02d} | Joint Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    dt = time.time() - t0
    print(f"Seed {seed} completed in {dt:.1f}s")

    ckpt_path = out_dir / f"S2A_SexAware_seed{seed}.pt"
    torch.save({
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "num_organs": NUM_ORGANS,
        "organ_names": ORGAN_NAMES,
        "epochs": epochs,
        "final_loss": avg_loss,
    }, ckpt_path)
    print(f"Saved Seed {seed} Checkpoint -> {ckpt_path.name}")
    return model

def main():
    ds_path = repo_root / "sharon" / "dataset" / "pointclouds_450.pt"
    print(f"Loading dataset from {ds_path}...")
    ds = PointCloudOrganDataset(str(ds_path))
    print(f"Loaded {len(ds)} patient point clouds.")

    # 1. Train Sex Classifier Head
    train_sex_classifier(ds, epochs=25)

    # 2. Train 3 Ensemble Seeds
    for seed in [42, 43, 44]:
        train_seed(seed, ds, epochs=25)

    print("\nAll Sex-Aware models successfully trained and checkpoints generated!")

if __name__ == "__main__":
    main()
