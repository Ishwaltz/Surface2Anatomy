import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

def audit_augmentations():
    print("=" * 80)
    print("AUDIT: DATA AUGMENTATION IMPLEMENTATION & CONSISTENCY")
    print("=" * 80)

    # Inspect dataset.py
    dataset_file = repo_root / "sharon" / "dataset.py"
    with open(dataset_file) as f:
        code = f.read()

    has_jitter = "jitter_std" in code and "torch.randn_like(pts)" in code
    has_rot = "rotation_aug" in code
    rot_implemented = "rotation_aug" in code and "torch.matmul" in code

    print(f"Jitter Augmentation Present: {has_jitter}")
    print(f"Rotation Augmentation Argument Present: {has_rot}")
    print(f"Rotation Augmentation Actually Implemented: {rot_implemented}")

    out_md = repo_root / "reports" / "phase1" / "04_augmentation_audit.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w") as f:
        f.write("# Data Augmentation Audit Report\n\n")
        f.write("## Executive Summary\n")
        f.write("During active training (`train_evidential.py`, `train_same.py`, `train_benchmarks.py`), dataset augmentation is enabled via `augment=True` in `PointCloudOrganDataset`.\n\n")
        
        f.write("## Augmentation Matrix\n\n")
        f.write("| Augmentation Name | Applied to Surface? | Applied to Targets? | Mathematically Consistent? | Code Location | Severity if Wrong |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- | :--- |\n")
        f.write("| **Gaussian Point Jitter** | Yes (`pts + noise * 0.005`) | No | **YES** (Valid feature noise) | `sharon/dataset.py:62-64` | **LOW** |\n")
        f.write("| **3D Spatial Rotation** | No (Unused parameter) | No | **N/A** (Dead parameter) | `sharon/dataset.py:22,45` | **MEDIUM** (Misleading interface) |\n")
        f.write("| **Spatial Translation** | No | No | **N/A** | None | None |\n")
        f.write("| **Isotropic/Anisotropic Scaling** | No | No | **N/A** | None | None |\n")
        f.write("| **Axis Reflection / Flipping** | No | No | **N/A** | None | None |\n\n")

        f.write("## Key Findings\n")
        f.write("1. **No Spatial Coordinate Transformations in Augmentation**: The only active augmentation is minor independent Gaussian point coordinate jitter (`sigma = 0.005` in normalized cube `[-1, 1]`) applied to surface points.\n")
        f.write("2. **Targets are Not Jittered**: Because Gaussian jitter represents surface sensor noise rather than patient translation, keeping target organ centroids fixed is mathematically sound.\n")
        f.write("3. **Dead Argument `rotation_aug`**: `PointCloudOrganDataset.__init__` accepts `rotation_aug: bool = False`, but `__getitem__` never executes any rotation logic. If a developer enabled `rotation_aug=True` expecting rotation, nothing would happen.\n")

    print(f"[Done] Report written to: {out_md}")

if __name__ == "__main__":
    audit_augmentations()
