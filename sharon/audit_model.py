import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import center_of_mass

from dataset import CTLocalizationDataset
from model import Modular3DOrganPredictor
from losses import soft_argmax
from labels import get_organ_names, ORGAN_IDS


def audit_model(organ_id_to_audit: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Audit] Using device: {device}")

    # 1. Paths
    sharon_dir = Path(__file__).resolve().parent
    data_root = sharon_dir / "dataset"
    checkpoint_path = sharon_dir / "outputs/checkpoints/swin_unetr/best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = sharon_dir / "outputs/checkpoints/best_model.pth"
    output_dir = sharon_dir / "outputs/audits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset / Case Load
    dataset = CTLocalizationDataset(str(data_root), target_shape=(128, 128, 128), augment=False)
    if len(dataset) == 0:
        raise RuntimeError("No valid cases found in dataset directory!")

    case_data = dataset[0]
    case_id = case_data["case_id"]
    print(f"[Audit] Auditing case: {case_id}")

    # 3. Model Loading
    model = Modular3DOrganPredictor(num_organs=117, backbone="swin_unetr").to(device)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"[Audit] Successfully loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[WARNING] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.eval()

    # 4. Inference
    image = case_data["image"].unsqueeze(0).to(device)  # (1, 1, 128, 128, 128)
    gt_ctr = case_data["gt_centroids"].unsqueeze(0).to(device)  # (1, 117, 3) (z,y,x) normalized
    spacing = case_data["spacing"].unsqueeze(0).to(device)  # (1, 3) [dx, dy, dz]

    with torch.no_grad():
        pred_hm = model(image)
        pred_ctr = soft_argmax(pred_hm, temperature=1000.0)  # (1, 117, 3) (z,y,x) normalized

    # Index Mapping Alignment
    channel_index = ORGAN_IDS.index(organ_id_to_audit)
    organ_name = get_organ_names()[channel_index]
    print(f"[Audit Index Alignment] Organ ID {organ_id_to_audit} ('{organ_name}') maps to 0-based Tensor Channel Index {channel_index}")
    if organ_id_to_audit == 5:
        assert organ_name == "liver", f"Indexing Error: Expected 'liver' for ID 5, got '{organ_name}'"
        assert channel_index == 4, f"Indexing Error: Expected Channel Index 4 for Liver ID 5, got {channel_index}"

    # Ground Truth Centroid & Model Prediction Centroid for Channel Index
    gt_norm_zyx = gt_ctr[0, channel_index].cpu().numpy()
    pred_norm_zyx = pred_ctr[0, channel_index].cpu().numpy()

    target_shape = (128, 128, 128)  # (D, H, W)
    gt_vox_zyx = gt_norm_zyx * np.array(target_shape)
    pred_vox_zyx = pred_norm_zyx * np.array(target_shape)

    # Convert (z, y, x) to physical space mm using spacing (dx, dy, dz)
    dx, dy, dz = case_data["spacing"].numpy()
    diff_vox_xyz = np.array([
        (pred_norm_zyx[2] - gt_norm_zyx[2]) * target_shape[2],  # x
        (pred_norm_zyx[1] - gt_norm_zyx[1]) * target_shape[1],  # y
        (pred_norm_zyx[0] - gt_norm_zyx[0]) * target_shape[0],  # z
    ])
    diff_mm = diff_vox_xyz * np.array([dx, dy, dz])
    euclidean_error_mm = np.linalg.norm(diff_mm)

    print(f"\n================ AUDIT RESULTS: {organ_name.upper()} (ID {organ_id_to_audit}, Channel Index {channel_index}) ================")
    print(f"GT Voxel (z,y,x):         ({gt_vox_zyx[0]:.2f}, {gt_vox_zyx[1]:.2f}, {gt_vox_zyx[2]:.2f})")
    print(f"Pred Voxel (z,y,x):       ({pred_vox_zyx[0]:.2f}, {pred_vox_zyx[1]:.2f}, {pred_vox_zyx[2]:.2f})")
    print(f"Voxel Spacing (dx,dy,dz): ({dx:.2f}, {dy:.2f}, {dz:.2f}) mm")
    print(f"3D Physical Error:        {euclidean_error_mm:.2f} mm")
    print("=====================================================================================================\n")

    # 5. Orthogonal Visual Audit Plot (Axial, Coronal, Sagittal)
    ct_vol = case_data["image"].squeeze().numpy()  # (128, 128, 128) (D, H, W)

    # Slice Indices along GT centroid
    z_slice = int(np.clip(gt_vox_zyx[0], 0, target_shape[0] - 1))
    y_slice = int(np.clip(gt_vox_zyx[1], 0, target_shape[1] - 1))
    x_slice = int(np.clip(gt_vox_zyx[2], 0, target_shape[2] - 1))

    # Predicted slice coords
    pred_z = pred_vox_zyx[0]
    pred_y = pred_vox_zyx[1]
    pred_x = pred_vox_zyx[2]

    gt_z = gt_vox_zyx[0]
    gt_y = gt_vox_zyx[1]
    gt_x = gt_vox_zyx[2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial Slice (XY plane at Z=z_slice) -> axes[0]
    axes[0].imshow(ct_vol[z_slice, :, :], cmap="gray", origin="lower")
    axes[0].scatter(gt_x, gt_y, c="lime", s=120, marker="o", edgecolors="black", label="Ground Truth (O)")
    axes[0].scatter(pred_x, pred_y, c="red", s=150, marker="x", linewidths=3, label="Model Prediction (X)")
    axes[0].set_title(f"Axial Slice (Z={z_slice})")
    axes[0].set_xlabel("X (Width)")
    axes[0].set_ylabel("Y (Height)")
    axes[0].legend(loc="upper right")

    # Coronal Slice (XZ plane at Y=y_slice) -> axes[1]
    axes[1].imshow(ct_vol[:, y_slice, :], cmap="gray", origin="lower")
    axes[1].scatter(gt_x, gt_z, c="lime", s=120, marker="o", edgecolors="black", label="Ground Truth (O)")
    axes[1].scatter(pred_x, pred_z, c="red", s=150, marker="x", linewidths=3, label="Model Prediction (X)")
    axes[1].set_title(f"Coronal Slice (Y={y_slice})")
    axes[1].set_xlabel("X (Width)")
    axes[1].set_ylabel("Z (Depth)")
    axes[1].legend(loc="upper right")

    # Sagittal Slice (YZ plane at X=x_slice) -> axes[2]
    axes[2].imshow(ct_vol[:, :, x_slice], cmap="gray", origin="lower")
    axes[2].scatter(gt_y, gt_z, c="lime", s=120, marker="o", edgecolors="black", label="Ground Truth (O)")
    axes[2].scatter(pred_y, pred_z, c="red", s=150, marker="x", linewidths=3, label="Model Prediction (X)")
    axes[2].set_title(f"Sagittal Slice (X={x_slice})")
    axes[2].set_xlabel("Y (Height)")
    axes[2].set_ylabel("Z (Depth)")
    axes[2].legend(loc="upper right")

    plt.suptitle(
        f"Spatial Audit: Organ '{organ_name}' (ID {organ_id_to_audit}, Channel {channel_index}) | Case: {case_id}\nPhysical Euclidean Error: {euclidean_error_mm:.2f} mm",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    out_file = output_dir / f"audit_organ_{organ_id_to_audit}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Audit] Plot successfully saved to: {out_file.resolve()}")


if __name__ == "__main__":
    audit_model(organ_id_to_audit=5)
