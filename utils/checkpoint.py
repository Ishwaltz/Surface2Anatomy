import os
import torch
from pathlib import Path
 
 
def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str,
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path
 
 
def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)
 
 
def build_state(
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_error: float,
    cfg_dict: dict,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_error": best_val_error,
        "config": cfg_dict,
    }
 
 
def restore_state(ckpt: dict, model, optimizer, scheduler, scaler) -> tuple:
    """Returns (start_epoch, best_val_error)."""
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt.get("best_val_error", float("inf"))
 
 
def find_resume_checkpoint(checkpoint_dir: str) -> str | None:
    latest = os.path.join(checkpoint_dir, "latest_model.pth")
    if os.path.isfile(latest):
        return latest
    return None