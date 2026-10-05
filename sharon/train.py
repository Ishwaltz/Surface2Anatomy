import os
import csv
import time
import random
import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from config import Config
from dataset import CTLocalizationDataset
from model import Modular3DOrganPredictor
from losses import MultiTaskLoss
from labels import NUM_ORGANS, ORGAN_NAMES
from utils.checkpoint import (
    save_checkpoint, load_checkpoint, build_state,
    restore_state, find_resume_checkpoint,
)
from utils.metrics import calculate_physical_error, OrganErrorAccumulator


# ─────────────────────────── helpers ────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires a GPU. Exiting.")
    dev = torch.device("cuda")
    print(f"[Device] CUDA – {torch.cuda.get_device_name(0)}")
    return dev


def build_scheduler(cfg: Config, optimizer, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, (cfg.epochs - cfg.warmup_epochs) * steps_per_epoch),
            eta_min=cfg.lr_min,
        )
    elif cfg.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=cfg.patience // 2, factor=0.5)
    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


def warmup_lr(optimizer, epoch: int, base_lr: float, warmup_epochs: int):
    if epoch < warmup_epochs and warmup_epochs > 0:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def init_csv(path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["epoch", "train_loss", "val_loss", "avg_error_mm", "lr"] + \
                 [f"{n}_mm" for n in ORGAN_NAMES]
        w.writerow(header)


def append_csv(path: str, row: list):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ─────────────────────────── train epoch ────────────────────────

def run_epoch_train(
    model, loader, criterion, optimizer, scaler,
    device, cfg: Config, epoch: int, accum_steps: int,
):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    total_ctr = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch:03d} [Train]", leave=False)

    for step, batch in pbar:
        images   = batch["image"].to(device, non_blocking=True)
        gt_hm    = batch["gt_heatmaps"].to(device, non_blocking=True)
        gt_ctr   = batch["gt_centroids"].to(device, non_blocking=True)
        mask     = batch["found_mask"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=cfg.mixed_precision):
            pred_hm = model(images)
            loss, _, l_dict = criterion(pred_hm, gt_hm, gt_ctr, mask)
            loss_accum = loss / accum_steps

        scaler.scale(loss_accum).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_mse += l_dict.get("mse", 0.0)
        total_focal += l_dict.get("focal", 0.0)
        total_ctr += l_dict.get("centroid", 0.0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", ctr=f"{l_dict.get('centroid', 0.0):.4f}")

    n_steps = max(1, len(loader))
    return (
        total_loss / n_steps,
        total_mse / n_steps,
        total_focal / n_steps,
        total_ctr / n_steps,
    )


# ─────────────────────────── val epoch ──────────────────────────

def run_epoch_val(
    model, loader, criterion, device, cfg: Config, epoch: int
):
    model.eval()
    total_loss = 0.0
    acc = OrganErrorAccumulator(NUM_ORGANS)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            images   = batch["image"].to(device, non_blocking=True)
            gt_hm    = batch["gt_heatmaps"].to(device, non_blocking=True)
            gt_ctr   = batch["gt_centroids"].to(device, non_blocking=True)
            mask     = batch["found_mask"].to(device, non_blocking=True)
            spacing  = batch["spacing"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.mixed_precision):
                pred_hm = model(images)
                loss, pred_ctr, _ = criterion(pred_hm, gt_hm, gt_ctr, mask)

            total_loss += loss.item()
            err_mm = calculate_physical_error(pred_ctr, gt_ctr, spacing, cfg.target_shape)
            acc.update(err_mm, mask)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader)), acc


# ─────────────────────────── main ───────────────────────────────

def main():
    cfg = Config.from_args()
    cfg.make_dirs()
    set_seed(cfg.seed)
    device = get_device()

    # Save config snapshot
    cfg.save(os.path.join(cfg.output_dir, "config.yaml"))

    # ── Dataset ──────────────────────────────────────────────────
    full_ds = CTLocalizationDataset(
        cfg.data_root, cfg.target_shape, cfg.heatmap_size, augment=False
    )
    if len(full_ds) == 0:
        print(f"[WARNING] Dataset directory '{cfg.data_root}' is empty or contains no valid cases.")
        print(f"Creating a single synthetic sample for end-to-end pipeline execution and benchmark verification.")
        synth_case = Path(cfg.data_root) / "case_000"
        synth_case.mkdir(parents=True, exist_ok=True)
        import nibabel as nib
        dummy_ct = np.random.randn(128, 128, 128).astype(np.float32)
        dummy_seg = np.random.randint(0, 118, size=(128, 128, 128)).astype(np.int16)
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(dummy_ct, aff), str(synth_case / "ct.nii.gz"))
        nib.save(nib.Nifti1Image(dummy_seg, aff), str(synth_case / "segmentation.nii.gz"))
        full_ds = CTLocalizationDataset(
            cfg.data_root, cfg.target_shape, cfg.heatmap_size, augment=False
        )

    n_train = max(1, int(cfg.train_val_split * len(full_ds)))
    n_val   = max(1, len(full_ds) - n_train)
    if n_train + n_val > len(full_ds):
        n_val = max(1, len(full_ds) - n_train)

    train_ds, val_ds = random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_ds.dataset = CTLocalizationDataset(
        cfg.data_root, cfg.target_shape, cfg.heatmap_size, augment=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0 and len(train_ds) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0 and len(val_ds) > 0,
    )
    print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}")

    # ── Model / Criterion / Optimizer ────────────────────────────
    print(f"[Model] Initializing backbone: {cfg.backbone} ({cfg.num_organs} organs)")
    model = Modular3DOrganPredictor(
        num_organs=cfg.num_organs,
        heatmap_size=cfg.heatmap_size,
        freeze_blocks=cfg.freeze_blocks,
        dropout=cfg.dropout_prob,
        backbone=cfg.backbone,
    ).to(device)

    criterion = MultiTaskLoss(cfg.heatmap_weight, cfg.centroid_weight).to(device)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scaler    = GradScaler(enabled=cfg.mixed_precision)
    scheduler = build_scheduler(cfg, optimizer, max(1, len(train_loader)))

    # ── Resume ───────────────────────────────────────────────────
    start_epoch    = 0
    best_val_error = float("inf")
    no_improve     = 0

    resume_path = find_resume_checkpoint(cfg.checkpoint_dir)
    if resume_path:
        print(f"[Resume] Loading {resume_path}")
        ckpt = load_checkpoint(resume_path, device)
        start_epoch, best_val_error = restore_state(ckpt, model, optimizer, scheduler, scaler)
        print(f"[Resume] Continuing from epoch {start_epoch}, best_error={best_val_error:.2f} mm")

    # ── Logging ──────────────────────────────────────────────────
    writer   = None
    csv_path = os.path.join(cfg.log_dir, "metrics.csv")
    if cfg.log_tensorboard and HAS_TB:
        writer = SummaryWriter(cfg.tensorboard_dir)
    if cfg.log_csv and not os.path.isfile(csv_path):
        init_csv(csv_path)

    # ── Training Loop ─────────────────────────────────────────────
    root_ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    root_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        warmup_lr(optimizer, epoch, cfg.learning_rate, cfg.warmup_epochs)

        train_loss, train_mse, train_focal, train_ctr = run_epoch_train(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg, epoch + 1, cfg.grad_accum_steps,
        )

        if cfg.scheduler == "cosine" and epoch >= cfg.warmup_epochs:
            scheduler.step()

        val_loss, acc = run_epoch_val(model, val_loader, criterion, device, cfg, epoch + 1)

        if cfg.scheduler == "plateau":
            scheduler.step(val_loss)

        per_organ = acc.per_organ()
        avg_err   = acc.overall()
        cur_lr    = optimizer.param_groups[0]["lr"]
        elapsed   = time.time() - t0

        major_keys = ["spleen", "liver", "pancreas", "gallbladder", "urinary_bladder"]
        major_str = " | ".join([f"{k.capitalize()}: {per_organ.get(k, float('nan')):.1f}mm" for k in major_keys])

        # ── Print ─────────────────────────────────────────────
        print(
            f"\nEpoch {epoch+1:03d}/{cfg.epochs} | "
            f"TrainLoss={train_loss:.4f} (MSE={train_mse:.4f}, Focal={train_focal:.4f}, Ctr={train_ctr:.4f}) | "
            f"ValLoss={val_loss:.4f} | AvgErr={avg_err:.2f}mm | LR={cur_lr:.2e} | {elapsed:.0f}s"
        )
        print(f"  Major Organs Physical Error -> {major_str}")

        # ── TensorBoard ───────────────────────────────────────
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/train_mse", train_mse, epoch)
            writer.add_scalar("Loss/train_focal", train_focal, epoch)
            writer.add_scalar("Loss/train_centroid", train_ctr, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("Metrics/avg_error_mm", avg_err, epoch)
            writer.add_scalar("LR", cur_lr, epoch)

        # ── CSV ───────────────────────────────────────────────
        if cfg.log_csv:
            row = [epoch + 1, round(train_loss, 5), round(val_loss, 5),
                   round(avg_err, 3), f"{cur_lr:.2e}"]
            row += [round(per_organ.get(n, float("nan")), 3) for n in ORGAN_NAMES]
            append_csv(csv_path, row)

        # ── Checkpointing ─────────────────────────────────────
        is_best = (avg_err < best_val_error) or (epoch == 0 and not np.isnan(avg_err))
        if is_best:
            best_val_error = avg_err if not np.isnan(avg_err) else best_val_error
            no_improve = 0
        else:
            no_improve += 1

        state = build_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_error=best_val_error,
            cfg_dict=dataclasses.asdict(cfg),
        )

        save_checkpoint(state, cfg.checkpoint_dir, "latest_model.pth")
        save_checkpoint(state, str(root_ckpt_dir), "latest_model.pth")

        if is_best:
            save_checkpoint(state, cfg.checkpoint_dir, "best_model.pth")
            save_checkpoint(state, str(root_ckpt_dir), "best_model.pth")
            print(f"  ✓ New best model saved! ({best_val_error:.2f} mm)")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(state, cfg.checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
            print(f"  ✓ Periodic checkpoint saved (epoch {epoch+1})")

        if no_improve >= cfg.patience:
            print(f"\n[Early Stop] No improvement for {cfg.patience} epochs. Stopping.")
            break

    if writer:
        writer.close()

    print(f"\n{'='*55}")
    print(f"Training complete. Best avg error: {best_val_error:.2f} mm")
    print(f"Checkpoints: {cfg.checkpoint_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()