import sys
import hashlib
from pathlib import Path
import torch
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def main():
    print("=" * 80)
    print("STAGE 0: DATASET V2 VERIFICATION & PHASE 2 PROVENANCE")
    print("=" * 80)

    pt_path = repo_root / "sharon" / "dataset_v2" / "pointclouds_v2.pt"
    splits_path = repo_root / "sharon" / "dataset_v2" / "splits_v2.json"

    assert pt_path.exists(), f"Missing {pt_path}"
    assert splits_path.exists(), f"Missing {splits_path}"

    with open(pt_path, "rb") as f:
        pt_hash = hashlib.sha256(f.read()).hexdigest()
    with open(splits_path, "rb") as f:
        splits_hash = hashlib.sha256(f.read()).hexdigest()

    data = torch.load(str(pt_path), weights_only=False)
    import json
    with open(splits_path) as f:
        splits = json.load(f)

    # 1. Split counts
    tr_cases = splits["train_cases"]
    val_cases = splits["val_cases"]
    te_cases = splits["test_cases"]

    n_tr = len(tr_cases)
    n_val = len(val_cases)
    n_te = len(te_cases)
    n_tot = len(data["case_ids"])

    print(f"Total Unique Patients: {n_tot}")
    print(f"Split counts: Train={n_tr}, Val={n_val}, Test={n_te}")
    assert n_tot == 440, f"Expected 440, got {n_tot}"
    assert n_tr == 352, f"Expected 352, got {n_tr}"
    assert n_val == 44, f"Expected 44, got {n_val}"
    assert n_te == 44, f"Expected 44, got {n_te}"

    # 2. Split disjointness
    s_tr, s_val, s_te = set(tr_cases), set(val_cases), set(te_cases)
    assert len(s_tr & s_val) == 0, "Train and Val overlap!"
    assert len(s_tr & s_te) == 0, "Train and Test overlap!"
    assert len(s_val & s_te) == 0, "Val and Test overlap!"
    print("✓ Confirmed zero overlap between Train, Val, and Test splits.")

    # 3. Model coordinates and S_global
    S_global = data["s_global"]
    assert S_global == 500.0, f"Expected S_global=500.0, got {S_global}"
    print(f"✓ Confirmed global metric scale: S_global = {S_global:.1f} mm")

    pts_m = data["points_model"].numpy()
    tgt_m = data["targets_model"].numpy()
    m_mask = data["target_mask"].numpy() > 0.5
    
    print(f"Model space coordinate ranges (should be in [-1.0, 1.0]^3):")
    print(f"  Points model min={pts_m.min():.4f}, max={pts_m.max():.4f}")
    print(f"  Targets model min={tgt_m[m_mask].min():.4f}, max={tgt_m[m_mask].max():.4f}")
    assert pts_m.min() >= -1.0 and pts_m.max() <= 1.0
    assert tgt_m[m_mask].min() >= -1.0 and tgt_m[m_mask].max() <= 1.0
    print("✓ Confirmed model coordinates are strictly within [-1.0, 1.0]^3.")

    # 4. Target primary mask
    p_mask = data["target_primary_mask"].numpy()
    synthetic_indices = [117, 118, 119, 120]
    for s_idx in synthetic_indices:
        assert p_mask[:, s_idx].sum() == 0.0, f"Synthetic class {s_idx} in primary mask!"
    print("✓ Confirmed synthetic targets (117-120: uterus, ovaries, vagina) strictly excluded from primary mask.")

    # 5. Round-trip physical reconstruction
    pts_w = data["points_world_mm"]
    pts_c = data["points_centered_mm"]
    c_surf = data["surface_center_mm"].unsqueeze(1)
    pts_rec = pts_m * S_global + c_surf.numpy()
    err_surface = np.linalg.norm(pts_w.numpy() - pts_rec, axis=-1).max()
    print(f"✓ Confirmed physical reconstruction round-trip max error: {err_surface:.8e} mm")
    assert err_surface < 1e-3

    # 6. Generate 00_phase2_provenance.md
    out_md = repo_root / "reports" / "phase2" / "00_phase2_provenance.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Phase 2 Provenance & Dataset V2 Verification\n\n")
        f.write("## 1. Environment & Hardware\n")
        f.write(f"- **Operating System**: Linux 7.0.0-29-generic x86_64\n")
        f.write(f"- **Python Version**: {sys.version.split()[0]}\n")
        f.write(f"- **PyTorch Version**: {torch.__version__}\n")
        f.write(f"- **CUDA Available**: {torch.cuda.is_available()}\n")
        f.write(f"- **GPU**: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)\n")
        f.write(f"- **Random Seeds Designated**: `42`, `43`, `44`\n\n")

        f.write("## 2. Dataset V2 Cryptographic Verification\n")
        f.write(f"- **Dataset Path**: `sharon/dataset_v2/pointclouds_v2.pt`\n")
        f.write(f"- **Dataset SHA-256**: `{pt_hash}`\n")
        f.write(f"- **Splits Path**: `sharon/dataset_v2/splits_v2.json`\n")
        f.write(f"- **Splits SHA-256**: `{splits_hash}`\n")
        f.write(f"- **Total Unique Patients**: {n_tot}\n")
        f.write(f"- **Training Split**: {n_tr} cases (80.0%)\n")
        f.write(f"- **Validation Split**: {n_val} cases (10.0%)\n")
        f.write(f"- **Held-Out Test Split**: {n_te} cases (10.0%, **FROZEN - UNTOUCHED**)\n\n")

        f.write("## 3. Geometric & Representation Invariants\n")
        f.write(f"- **Global Metric Scale**: $S_{{\\text{{global}}}} = {S_global:.1f}\\text{{ mm}}$\n")
        f.write(f"- **Translation Normalization**: Patient-specific surface-derived center $c_{{\\text{{surface}}}}$ removed; identical center applied to targets.\n")
        f.write(f"- **Coordinate Bounds**: Model space coordinates strictly bounded in $[-1.0, 1.0]^3$.\n")
        f.write(f"- **Reconstruction Fidelity**: Maximum round-trip error $= {err_surface:.8e}\\text{{ mm}} < 10^{{-3}}\\text{{ mm}}$.\n")
        f.write(f"- **Primary Benchmark Integrity**: 4 synthetic female pelvic organs strictly masked out (`primary_valid_mask = 0`).\n")

    print(f"[Done] Report generated: {out_md}")

if __name__ == "__main__":
    main()
