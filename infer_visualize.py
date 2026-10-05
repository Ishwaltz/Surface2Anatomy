import argparse
from pathlib import Path
import scipy.ndimage as ndi
 
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, Resized,
)
 
from model import OrganLocalizationNet
from losses import soft_argmax
from labels import ORGAN_NAMES, NUM_ORGANS
 
 
TARGET_SHAPE = (128, 128, 128)
 
 
def get_transforms(target_shape):
    return Compose([
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=-1000.0, a_max=400.0,
                             b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=target_shape, mode="area"),
    ])
 
 
def load_model(model_path: str, device: torch.device) -> OrganLocalizationNet:
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    cfg   = ckpt.get("config", {})
    num_organs   = cfg.get("num_organs", NUM_ORGANS)
    heatmap_size = tuple(cfg.get("heatmap_size", (64, 64, 64)))
    freeze_blocks = cfg.get("freeze_blocks", 0)
 
    model = OrganLocalizationNet(
        num_organs=num_organs,
        heatmap_size=heatmap_size,
        freeze_blocks=freeze_blocks,
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
 
 
def visualize(ct_volume: np.ndarray, pred_norm: torch.Tensor, target_shape: tuple):
    try:
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        from skimage.measure import marching_cubes
    except ImportError:
        print("[Visualize] open3d / scikit-image not installed. Skipping.")
        return

    COLORS = {
        "spleen":       [0.8, 0.2, 0.2],
        "kidney_right": [0.2, 0.8, 0.2],
        "kidney_left":  [0.2, 0.8, 0.8],
        "gallbladder":  [0.8, 0.8, 0.2],
        "liver":        [0.8, 0.2, 0.8],
    }

    # Initialize the modern GUI application
    gui.Application.instance.initialize()
    window = o3d.visualization.O3DVisualizer("Organ Localization", 1024, 768)
    window.set_background([1.0, 1.0, 1.0, 1.0], None)  # White background
    window.show_settings = False

    # 1. Mesh Generation
    binary_body = ct_volume > 0.15
    solid_body = ndi.binary_fill_holes(binary_body)
    smooth_body = ndi.gaussian_filter(solid_body.astype(np.float32), sigma=1.5)
    verts, faces, *_ = marching_cubes(smooth_body, level=0.5)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.88, 0.82, 0.78]) # Skin tone
    
    # Standard material to properly render colors
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    
    window.add_geometry("Torso", mesh, mat)
    skin_verts = np.asarray(mesh.vertices)
    shape_arr = np.array(target_shape, dtype=np.float32)
    
    # 2. Add Markers and 2D Screen-Space Labels
    for i, name in enumerate(ORGAN_NAMES):
        vox = pred_norm[i].cpu().numpy() * shape_arr
        color = COLORS.get(name, [1.0, 1.0, 1.0])
        
        # Raycasting to surface
        xz_dist = np.linalg.norm(skin_verts[:, [0, 2]] - vox[[0, 2]], axis=1)
        cylinder_indices = np.where(xz_dist < 3.0)[0]
        
        if len(cylinder_indices) > 0:
            candidates = skin_verts[cylinder_indices]
            max_y = np.max(candidates[:, 1])
            skin_pt = vox.copy()
            skin_pt[1] = max_y + 1.0
        else:
            skin_pt = vox.copy()
            skin_pt[1] += 20.0
            
        # Outer surface sphere (marker)
        sphere_out = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        sphere_out.translate(skin_pt)
        sphere_out.paint_uniform_color(color)
        window.add_geometry(f"{name}_marker", sphere_out, mat)

        # Clean 2D Screen-Space Text Label
        clean_name = name.replace("_", " ").title()
        
        # Offset the label slightly above the point so it doesn't overlap the sphere
        label_pos = skin_pt + np.array([0, 5, 0]) 
        window.add_3d_label(label_pos, clean_name)

    window.reset_camera_to_default()
    gui.Application.instance.add_window(window)
    gui.Application.instance.run()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct",    type=str, required=True, help="Path to ct.nii.gz")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out",   type=str, default=None, help="Save predictions JSON")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")
    print(f"[Inference] CT:     {args.ct}")
    print(f"[Inference] Model:  {args.model}\n")
 
    results, ct_vol, pred_norm = predict(args.ct, args.model, TARGET_SHAPE, device)
 
    print(f"{'Organ':<16} {'Voxel (z,y,x)':<30} {'Physical (mm)'}")
    print("-" * 65)
    for name, vals in results.items():
        vox = [f"{v:.1f}" for v in vals["voxel"]]
        phys = [f"{v:.1f}" for v in vals["physical_mm"]]
        print(f"{name:<16} ({', '.join(vox):<27}) ({', '.join(phys)})")
 
    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to {args.out}")
 
    if args.visualize:
        visualize(ct_vol, pred_norm, TARGET_SHAPE)
 
 
if __name__ == "__main__":
    main()
    
    