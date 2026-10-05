import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Ensure local sharon modules are imported
SHARON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHARON_DIR))

from dataset import CTLocalizationDataset
from model import Modular3DOrganPredictor
from losses import soft_argmax
from labels import ORGAN_NAMES, ORGAN_IDS, NUM_ORGANS, ORGAN_INDEX

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[117 Audits] Device: {device}")

    # Paths
    data_root = SHARON_DIR / "dataset"
    ckpt_path = SHARON_DIR / "outputs/checkpoints/swin_unetr/best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = SHARON_DIR / "outputs/checkpoints/best_model.pth"
    
    audit_img_dir = SHARON_DIR / "outputs/audits"
    audit_img_dir.mkdir(parents=True, exist_ok=True)
    pdf_out_path = Path("/home/sharon/Desktop/3D_Organ_Location_Prediction_117_Organs_Visual_Audit.pdf")

    # Load Model
    print(f"[117 Audits] Loading weights from {ckpt_path}...")
    model = Modular3DOrganPredictor(num_organs=NUM_ORGANS, backbone="swin_unetr").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Dataset
    print("[117 Audits] Initializing Dataset...")
    dataset = CTLocalizationDataset(str(data_root), target_shape=(128, 128, 128), augment=False)
    print(f"[117 Audits] Total dataset cases: {len(dataset)}")

    audited_organs = {}
    target_shape = (128, 128, 128)

    print("\n[117 Audits] Generating visual 3-plane validation plots for all 117 organs...")
    
    # Iterate through cases to find and audit every organ in its true anatomical volume
    for case_idx in range(len(dataset)):
        if len(audited_organs) == NUM_ORGANS:
            break
            
        case_data = dataset[case_idx]
        case_id = case_data["case_id"]
        found_mask = case_data["found_mask"]
        
        # Check if this case contains any un-audited organs
        needed_organs_in_case = [i for i in range(NUM_ORGANS) if found_mask[i] and i not in audited_organs]
        if not needed_organs_in_case:
            continue
            
        image = case_data["image"].unsqueeze(0).to(device)
        gt_ctr = case_data["gt_centroids"].unsqueeze(0).to(device)
        spacing = case_data["spacing"].numpy()
        ct_vol = case_data["image"].squeeze().numpy()

        with torch.no_grad():
            pred_hm = model(image)
            pred_ctr = soft_argmax(pred_hm, temperature=1000.0)

        for channel_idx in needed_organs_in_case:
            org_id = ORGAN_IDS[channel_idx]
            org_name = ORGAN_NAMES[channel_idx]

            gt_norm = gt_ctr[0, channel_idx].cpu().numpy()
            pred_norm = pred_ctr[0, channel_idx].cpu().numpy()

            gt_vox = gt_norm * np.array(target_shape)
            pred_vox = pred_norm * np.array(target_shape)

            diff_xyz = np.array([
                (pred_norm[2] - gt_norm[2]) * target_shape[2],
                (pred_norm[1] - gt_norm[1]) * target_shape[1],
                (pred_norm[0] - gt_norm[0]) * target_shape[0],
            ])
            diff_mm = diff_xyz * np.array([spacing[0], spacing[1], spacing[2]])
            err_mm = float(np.linalg.norm(diff_mm))

            # Slice indices
            z_slice = int(np.clip(round(gt_vox[0]), 0, target_shape[0] - 1))
            y_slice = int(np.clip(round(gt_vox[1]), 0, target_shape[1] - 1))
            x_slice = int(np.clip(round(gt_vox[2]), 0, target_shape[2] - 1))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Axial (XY)
            axes[0].imshow(ct_vol[z_slice, :, :], cmap="gray", origin="lower")
            axes[0].scatter(gt_vox[2], gt_vox[1], c="lime", s=130, marker="o", edgecolors="black", label="Ground Truth (O)")
            axes[0].scatter(pred_vox[2], pred_vox[1], c="red", s=160, marker="x", linewidths=3, label="Prediction (X)")
            axes[0].set_title(f"Axial Slice (Z={z_slice})", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("X (Width)")
            axes[0].set_ylabel("Y (Height)")
            axes[0].legend(loc="upper right", fontsize=8)

            # Coronal (XZ)
            axes[1].imshow(ct_vol[:, y_slice, :], cmap="gray", origin="lower")
            axes[1].scatter(gt_vox[2], gt_vox[0], c="lime", s=130, marker="o", edgecolors="black", label="Ground Truth (O)")
            axes[1].scatter(pred_vox[2], pred_vox[0], c="red", s=160, marker="x", linewidths=3, label="Prediction (X)")
            axes[1].set_title(f"Coronal Slice (Y={y_slice})", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("X (Width)")
            axes[1].set_ylabel("Z (Depth)")
            axes[1].legend(loc="upper right", fontsize=8)

            # Sagittal (YZ)
            axes[2].imshow(ct_vol[:, :, x_slice], cmap="gray", origin="lower")
            axes[2].scatter(gt_vox[1], gt_vox[0], c="lime", s=130, marker="o", edgecolors="black", label="Ground Truth (O)")
            axes[2].scatter(pred_vox[1], pred_vox[0], c="red", s=160, marker="x", linewidths=3, label="Prediction (X)")
            axes[2].set_title(f"Sagittal Slice (X={x_slice})", fontsize=11, fontweight="bold")
            axes[2].set_xlabel("Y (Height)")
            axes[2].set_ylabel("Z (Depth)")
            axes[2].legend(loc="upper right", fontsize=8)

            plt.suptitle(
                f"Organ #{org_id:03d}: {org_name.upper()} | Case: {case_id}\nPhysical Euclidean Error: {err_mm:.2f} mm | Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm",
                fontsize=13, fontweight="bold"
            )
            plt.tight_layout()

            img_path = audit_img_dir / f"audit_organ_{org_id}.png"
            plt.savefig(img_path, dpi=130)
            plt.close()

            audited_organs[channel_idx] = {
                "org_id": org_id,
                "org_name": org_name,
                "case_id": case_id,
                "error_mm": err_mm,
                "img_path": img_path,
            }
            print(f"  [✓ {len(audited_organs):03d}/117] Organ #{org_id:03d} '{org_name}': Error = {err_mm:.2f} mm (Case: {case_id})")

    # If any organs were completely absent in the scanned dataset subset, generate fallback audit
    if len(audited_organs) < NUM_ORGANS:
        case_data = dataset[0]
        case_id = case_data["case_id"]
        ct_vol = case_data["image"].squeeze().numpy()
        image = case_data["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_hm = model(image)
            pred_ctr = soft_argmax(pred_hm, temperature=1000.0)
            
        for channel_idx in range(NUM_ORGANS):
            if channel_idx not in audited_organs:
                org_id = ORGAN_IDS[channel_idx]
                org_name = ORGAN_NAMES[channel_idx]
                pred_norm = pred_ctr[0, channel_idx].cpu().numpy()
                pred_vox = pred_norm * np.array(target_shape)
                z_slice = int(np.clip(round(pred_vox[0]), 0, target_shape[0] - 1))
                y_slice = int(np.clip(round(pred_vox[1]), 0, target_shape[1] - 1))
                x_slice = int(np.clip(round(pred_vox[2]), 0, target_shape[2] - 1))
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(ct_vol[z_slice, :, :], cmap="gray", origin="lower")
                axes[0].scatter(pred_vox[2], pred_vox[1], c="red", s=160, marker="x", linewidths=3, label="Model Spatial Prediction (X)")
                axes[0].set_title(f"Axial Slice (Z={z_slice})", fontsize=11, fontweight="bold")
                axes[0].legend(loc="upper right", fontsize=8)

                axes[1].imshow(ct_vol[:, y_slice, :], cmap="gray", origin="lower")
                axes[1].scatter(pred_vox[2], pred_vox[0], c="red", s=160, marker="x", linewidths=3, label="Model Spatial Prediction (X)")
                axes[1].set_title(f"Coronal Slice (Y={y_slice})", fontsize=11, fontweight="bold")
                axes[1].legend(loc="upper right", fontsize=8)

                axes[2].imshow(ct_vol[:, :, x_slice], cmap="gray", origin="lower")
                axes[2].scatter(pred_vox[1], pred_vox[0], c="red", s=160, marker="x", linewidths=3, label="Model Spatial Prediction (X)")
                axes[2].set_title(f"Sagittal Slice (X={x_slice})", fontsize=11, fontweight="bold")
                axes[2].legend(loc="upper right", fontsize=8)

                plt.suptitle(
                    f"Organ #{org_id:03d}: {org_name.upper()} (Field of View Prediction) | Case: {case_id}\n[Note: Outside Torso FOV for this scan - Spatial Expectation Plotted]",
                    fontsize=12, fontweight="bold"
                )
                plt.tight_layout()
                img_path = audit_img_dir / f"audit_organ_{org_id}.png"
                plt.savefig(img_path, dpi=130)
                plt.close()
                audited_organs[channel_idx] = {
                    "org_id": org_id,
                    "org_name": org_name,
                    "case_id": case_id,
                    "error_mm": float("nan"),
                    "img_path": img_path,
                }
                print(f"  [✓ {len(audited_organs):03d}/117] Organ #{org_id:03d} '{org_name}': FOV Spatial Expectation Plotted")

    # Compile all 117 images into a multi-page PDF
    print(f"\n[117 Audits] Compiling all 117 images into PDF: {pdf_out_path}...")
    pdf_pages = []
    for channel_idx in range(NUM_ORGANS):
        org_id = ORGAN_IDS[channel_idx]
        img_path = audit_img_dir / f"audit_organ_{org_id}.png"
        if img_path.exists():
            im = Image.open(img_path).convert("RGB")
            pdf_pages.append(im)

    if pdf_pages:
        pdf_pages[0].save(
            str(pdf_out_path),
            save_all=True,
            append_images=pdf_pages[1:],
            resolution=150.0,
        )
        print(f"\n==================================================")
        print(f" [SUCCESS] Successfully compiled {len(pdf_pages)} organ audit diagrams!")
        print(f" PDF Location: {pdf_out_path}")
        print(f" File Size: {pdf_out_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"==================================================")

if __name__ == "__main__":
    main()
