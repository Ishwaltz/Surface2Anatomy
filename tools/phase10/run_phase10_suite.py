import os
import sys
import json
import csv
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup repository root
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "sharon"))

from sharon.model_target_query import TargetQueryTransformerDecoder, MultiScaleSurfacePointNet2Encoder
from sharon.model_gnn import PointNetSetAbstraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_log(msg, log_file=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

class Phase10Dataset(Dataset):
    def __init__(self, pts, tgts, masks, indices, S_global=500.0, augment=False, normals=None):
        self.indices = indices
        self.S_global = S_global
        self.augment = augment
        self.has_normals = normals is not None
        
        self.pts = torch.from_numpy(pts[indices] / S_global).float()
        self.targets = torch.from_numpy(tgts[indices] / S_global).float()
        self.masks = torch.from_numpy(masks[indices]).float()
        if self.has_normals:
            self.normals = torch.from_numpy(normals[indices]).float()
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        pts = self.pts[idx]
        if self.augment:
            pts = pts + torch.randn_like(pts) * 0.001 # 0.5 mm jitter
        item = {
            "pts": pts,
            "targets": self.targets[idx],
            "masks": self.masks[idx],
            "dataset_index": self.indices[idx]
        }
        if self.has_normals:
            item["normals"] = self.normals[idx]
        return item

def compute_detailed_metrics(pred_c_mm, tgt_c_mm, masks, benchmark_indices):
    pred_sub = pred_c_mm[:, benchmark_indices, :]
    tgt_sub = tgt_c_mm[:, benchmark_indices, :]
    mask_sub = masks[:, benchmark_indices]
    
    errors = np.linalg.norm(pred_sub - tgt_sub, axis=-1) # (N, num_bench)
    valid = (mask_sub == 1)
    
    if np.sum(valid) == 0:
        return {}
        
    valid_errors = errors[valid]
    
    # Target-level statistics
    target_stats = []
    target_mres = []
    for t_i in range(len(benchmark_indices)):
        v = (mask_sub[:, t_i] == 1)
        if np.sum(v) > 0:
            e_t = errors[v, t_i]
            mre_t = float(np.mean(e_t))
            target_mres.append(mre_t)
            target_stats.append({
                "target_idx": benchmark_indices[t_i],
                "support": int(np.sum(v)),
                "mre": mre_t,
                "median": float(np.median(e_t)),
                "std": float(np.std(e_t)),
                "p75": float(np.percentile(e_t, 75)),
                "p90": float(np.percentile(e_t, 90)),
                "p95": float(np.percentile(e_t, 95)),
                "sdr5": float(np.mean(e_t <= 5.0) * 100.0),
                "sdr10": float(np.mean(e_t <= 10.0) * 100.0),
                "sdr15": float(np.mean(e_t <= 15.0) * 100.0),
                "errors": e_t
            })
            
    macro_mre = float(np.mean(target_mres)) if len(target_mres) > 0 else float(np.mean(valid_errors))
    micro_mre = float(np.mean(valid_errors))
    median_err = float(np.median(valid_errors))
    p75_err = float(np.percentile(valid_errors, 75))
    p90_err = float(np.percentile(valid_errors, 90))
    p95_err = float(np.percentile(valid_errors, 95))
    sdr5 = float(np.mean(valid_errors <= 5.0) * 100.0)
    sdr10 = float(np.mean(valid_errors <= 10.0) * 100.0)
    sdr15 = float(np.mean(valid_errors <= 15.0) * 100.0)
    sdr20 = float(np.mean(valid_errors <= 20.0) * 100.0)
    
    # Patient-level statistics
    patient_mres = []
    for i in range(len(pred_c_mm)):
        v = (mask_sub[i] == 1)
        if np.sum(v) > 0:
            patient_mres.append(float(np.mean(errors[i, v])))
    macro_patient_mre = float(np.mean(patient_mres)) if len(patient_mres) > 0 else micro_mre
    
    return {
        "macro_mre": macro_mre,
        "micro_mre": micro_mre,
        "macro_patient_mre": macro_patient_mre,
        "median": median_err,
        "p75": p75_err,
        "p90": p90_err,
        "p95": p95_err,
        "sdr5": sdr5,
        "sdr10": sdr10,
        "sdr15": sdr15,
        "sdr20": sdr20,
        "target_stats": target_stats,
        "patient_mres": patient_mres,
        "errors_matrix": errors,
        "valid_matrix": mask_sub
    }

def evaluate_model(model, val_loader, S_global=500.0, return_attention=False):
    model.eval()
    preds = []
    tgts = []
    masks = []
    attns = []
    
    with torch.no_grad():
        for batch in val_loader:
            pts = batch["pts"].to(device)
            target = batch["targets"].numpy() * S_global
            mask = batch["masks"].numpy()
            
            pred_m, attn_w = model(pts)
            pred_mm = pred_m.cpu().numpy() * S_global
            
            preds.append(pred_mm)
            tgts.append(target)
            masks.append(mask)
            if return_attention and attn_w is not None:
                attns.append(attn_w.cpu().numpy())
                
    preds_mm = np.concatenate(preds, axis=0)
    tgts_mm = np.concatenate(tgts, axis=0)
    masks_arr = np.concatenate(masks, axis=0)
    attns_arr = np.concatenate(attns, axis=0) if return_attention and len(attns) > 0 else None
    return preds_mm, tgts_mm, masks_arr, attns_arr

print("Phase 10 Suite Module Imported Successfully.")
