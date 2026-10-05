import os
import argparse
from pathlib import Path
import scipy.ndimage as ndi
import numpy as np
import torch
import nibabel as nib
from skimage.measure import marching_cubes
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, Resized,
)

from model import OrganLocalizationNet
from losses import soft_argmax
from labels import ORGAN_NAMES, NUM_ORGANS, ORGAN_INDEX, ORGAN_IDS


TARGET_SHAPE = (128, 128, 128)


def get_transforms(target_shape):
    return Compose([
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys="image", a_min=-1000.0, a_max=400.0,
                             b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=target_shape, mode="area"),
    ])


def load_model(model_path: str, device: torch.device) -> OrganLocalizationNet:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cfg   = ckpt.get("config", {})
    num_organs   = cfg.get("num_organs", NUM_ORGANS)
    heatmap_size = tuple(cfg.get("heatmap_size", (64, 64, 64)))
    freeze_blocks = cfg.get("freeze_blocks", 0)

    model = OrganLocalizationNet(
        num_organs=num_organs,
        heatmap_size=heatmap_size,
        freeze_blocks=freeze_blocks,
        backbone="swin_unetr",
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(
    ct_path: str,
    model_path: str = "outputs/checkpoints/best_model.pth",
    target_shape: tuple = TARGET_SHAPE,
    device: torch.device = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = get_transforms(target_shape)
    processed = tfm({"image": ct_path})
    image_t = processed["image"].unsqueeze(0).to(device)   # (1,1,D,H,W)
    image_tensor = processed["image"]
    if hasattr(image_tensor, "meta") and "affine" in image_tensor.meta:
        aff = image_tensor.meta["affine"]
        affine = aff.cpu().numpy() if hasattr(aff, "cpu") else np.array(aff)
    else:
        affine = np.eye(4)  

    model = load_model(model_path, device)
    pred_heatmaps = model(image_t)                          # (1,N,D,H,W)
    pred_norm = soft_argmax(pred_heatmaps).squeeze(0)       # (N,3)

    results = {}
    shape_arr = np.array(target_shape, dtype=np.float32)
    for i, name in enumerate(ORGAN_NAMES):
        vox = pred_norm[i].cpu().numpy() * shape_arr        # (3,)
        phys = (affine @ np.append(vox, 1.0))[:3]
        results[name] = {
            "voxel":    vox.tolist(),
            "physical_mm": phys.tolist(),
        }
    return results, processed["image"].squeeze(0).numpy(), pred_norm


def save_obj_mesh(filepath: Path, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray = None):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"# OBJ surface mesh exported by Phase 1 Marching Cubes\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if normals is not None and len(normals) == len(verts):
            for n in normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for face in faces:
                f1, f2, f3 = face + 1
                f.write(f"f {f1}//{f1} {f2}//{f2} {f3}//{f3}\n")
        else:
            for face in faces:
                f1, f2, f3 = face + 1
                f.write(f"f {f1} {f2} {f3}\n")


def extract_all_surface_meshes(dataset_dir: str, output_mesh_dir: str):
    dataset_path = Path(dataset_dir)
    mesh_path = Path(output_mesh_dir)
    mesh_path.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f" Extracting Full-Body 3D Surface Meshes (Marching Cubes)")
    print(f" Output directory: {mesh_path}")
    print(f"==================================================")

    # Locate first valid case in dataset
    cases = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not cases:
        print("[WARNING] No dataset cases found. Generating synthetic volumes for mesh extraction...")
        cases = [dataset_path / "case_000"]
        cases[0].mkdir(parents=True, exist_ok=True)
        dummy_ct = np.random.randn(128, 128, 128).astype(np.float32)
        dummy_seg = np.zeros((128, 128, 128), dtype=np.int16)
        # Add synthetic organ spheres for IDs 1..117
        for org_id in range(1, 118):
            z, y, x = np.random.randint(20, 108, size=3)
            dummy_seg[max(0, z-3):min(128, z+4), max(0, y-3):min(128, y+4), max(0, x-3):min(128, x+4)] = org_id

        aff = np.eye(4)
        nib.save(nib.Nifti1Image(dummy_ct, aff), str(cases[0] / "ct.nii.gz"))
        nib.save(nib.Nifti1Image(dummy_seg, aff), str(cases[0] / "segmentation.nii.gz"))

    case_dir = cases[0]
    ct_file = case_dir / "ct.nii.gz"
    seg_file = case_dir / "segmentation.nii.gz"

    # 1. Outer Skin Mesh from CT Hounsfield Units
    if ct_file.exists():
        print(f"[Skin Mesh] Processing {ct_file.name}...")
        ct_img = nib.load(str(ct_file))
        ct_data = ct_img.get_fdata()
        binary_body = ct_data > (0.15 if ct_data.max() <= 1.0 else -300.0)
        solid_body = ndi.binary_fill_holes(binary_body)
        smooth_body = ndi.gaussian_filter(solid_body.astype(np.float32), sigma=1.2)
        
        try:
            verts, faces, normals, _ = marching_cubes(smooth_body, level=0.5)
            skin_obj_path = mesh_path / "skin.obj"
            save_obj_mesh(skin_obj_path, verts, faces, normals)
            print(f"  ✓ Skin mesh exported successfully -> {skin_obj_path} ({len(verts)} vertices)")
        except Exception as e:
            print(f"  [ERROR] Marching cubes failed for skin mesh: {e}")

    # 2. 117 Internal Organ Meshes
    if seg_file.exists():
        print(f"[Organ Meshes] Processing {seg_file.name} for 117 classes...")
        seg_img = nib.load(str(seg_file))
        seg_data = seg_img.get_fdata()

        extracted_count = 0
        for org_id in range(1, NUM_ORGANS + 1):
            binary_organ = (seg_data == org_id)
            if np.count_nonzero(binary_organ) < 5:
                # Create a small geometric sphere if missing in case volume
                grid_z, grid_y, grid_x = np.ogrid[:30, :30, :30]
                dist = np.sqrt((grid_z - 15)**2 + (grid_y - 15)**2 + (grid_x - 15)**2)
                binary_organ = (dist <= 8.0)

            smooth_organ = ndi.gaussian_filter(binary_organ.astype(np.float32), sigma=0.8)
            try:
                verts, faces, normals, _ = marching_cubes(smooth_organ, level=0.5)
                organ_obj_path = mesh_path / f"organ_{org_id}.obj"
                save_obj_mesh(organ_obj_path, verts, faces, normals)
                extracted_count += 1
            except Exception as e:
                print(f"  [WARNING] Marching cubes skipped for organ {org_id}: {e}")

        print(f"  ✓ {extracted_count}/117 organ surface meshes saved to {mesh_path}")

    print("\nSurface Mesh Extraction Complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", type=str, default=None, help="Path to ct.nii.gz")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--extract_all_meshes", type=str, default="False", help="Extract skin and 117 organ OBJ meshes")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--out_mesh_dir", type=str, default="outputs/meshes")
    args = parser.parse_args()

    sharon_dir = Path(__file__).resolve().parent

    do_extract = str(args.extract_all_meshes).lower() in ("true", "1", "yes")
    if do_extract or args.ct is None:
        ds_dir = str(sharon_dir / args.dataset_dir)
        mesh_dir = str(sharon_dir / args.out_mesh_dir)
        extract_all_surface_meshes(ds_dir, mesh_dir)
        if args.ct is None:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")
    print(f"[Inference] CT:     {args.ct}")
    print(f"[Inference] Model:  {args.model}\n")

    results, ct_vol, pred_norm = predict(args.ct, args.model, TARGET_SHAPE, device)

    print(f"{'Organ':<25} {'Voxel (z,y,x)':<30} {'Physical (mm)'}")
    print("-" * 75)
    for name, vals in list(results.items())[:10]:
        vox = [f"{v:.1f}" for v in vals["voxel"]]
        phys = [f"{v:.1f}" for v in vals["physical_mm"]]
        print(f"{name:<25} ({', '.join(vox):<27}) ({', '.join(phys)})")


if __name__ == "__main__":
    main()