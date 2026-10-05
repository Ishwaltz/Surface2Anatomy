import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent

def generate_pipeline_map():
    out_md = repo_root / "reports" / "phase1" / "01_data_pipeline_map.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w") as f:
        f.write("# Forensic Mapping of the Complete Data, Geometry & Coordinate Pipeline\n\n")
        f.write("## 1. Complete End-to-End Pipeline Map\n\n")
        f.write("```text\n")
        f.write("Raw Patient DICOM / NIfTI Files (raw_data/*.nii.gz)\n")
        f.write("        ↓  [SimpleITK / TotalSegmentator API: dataset_preprocessing.py]\n")
        f.write("Preprocessed CT (ct.nii.gz) & Multiclass Segmentation (segmentation.nii.gz)\n")
        f.write("        ↓  [Geometric Ellipsoidal Insertion: _extract_female_pelvic_anatomy()]\n")
        f.write("121-Class Organ Mask Volume (segmentation.nii.gz: shape (Nx, Ny, Nz))\n")
        f.write("        ↓  [scipy.ndimage.center_of_mass: pointcloud_sampler.py]\n")
        f.write("Voxel Centers of Mass com = (i, j, k)\n")
        f.write("        ↓  [CRITICAL BUG: cz, cy, cx = com -> affine @ [cx, cy, cz, 1.0]]\n")
        f.write("Corrupted Physical Target Centroids (raw_centroids: X & Z swapped, Z ~ 2.5m)\n")
        f.write("        ↓  [Bounding-Box Rescaling of Static Template: load_canonical_surface_points()]\n")
        f.write("Synthetic Torso Surface Points (raw_points: shape (4096, 3), stretched to warped bbox)\n")
        f.write("        ↓  [Anisotropic Bounding-Box Normalization: [-1, 1]^3]\n")
        f.write("Normalized Tensors: points (B, 4096, 3), centroids (B, 121, 3), scales (B, 3), centers (B, 3)\n")
        f.write("        ↓  [torch.save: pointcloud_sampler.py]\n")
        f.write("Consolidated Binary Dataset (sharon/dataset/pointclouds_450.pt)\n")
        f.write("        ↓  [Dataset Loader + Point Jitter (0.005): dataset.py: PointCloudOrganDataset]\n")
        f.write("Batch Tensors: pts, ctr, sex, mask, center, scale\n")
        f.write("        ↓  [Neural Network Forward Pass: model_evidential.py / model_same.py / model_gnn.py]\n")
        f.write("Predicted Normalized Coordinates (gamma / full_coords in [-1, 1]^3)\n")
        f.write("        ↓  [Physical De-normalization: preds_mm = preds * scale + center]\n")
        f.write("Predicted Physical Coordinates (in warped mm coordinate space)\n")
        f.write("        ↓  [Euclidean Distance: torch.norm(preds_mm - ctr_mm, dim=-1)]\n")
        f.write("Physical Mean Radial Error Metric (MRE in warped mm)\n")
        f.write("```\n\n")

        f.write("## 2. Stage-by-Stage Forensic Breakdown\n\n")
        
        stages = [
            {
                "stage": "Stage 1: Raw CT & Segmentation Ingestion",
                "file": "sharon/dataset_preprocessing.py",
                "class_func": "DatasetPreprocessor.process_case()",
                "inputs": "Raw CT DICOM directories or NIfTI files in raw_data/",
                "outputs": "sharon/dataset/case_XXX/ct.nii.gz, segmentation.nii.gz",
                "units": "Hounsfield Units (HU) for CT; Integer class IDs (1..117) for segmentation",
                "frame": "LAS (Left, Anterior, Superior) from NIfTI affine",
                "shape": "(768, 768, 90) or (512, 512, 90..132)",
                "transform": "SimpleITK DICOM series reader -> NIfTI-1 file export"
            },
            {
                "stage": "Stage 2: Female Pelvic Organ Augmentation (K=121)",
                "file": "sharon/dataset_preprocessing.py",
                "class_func": "DatasetPreprocessor._extract_female_pelvic_anatomy()",
                "inputs": "Base 117-class segmentation.nii.gz, patient sex prior",
                "outputs": "121-class segmentation with labels 118 (uterus), 119 (ovary_l), 120 (ovary_r), 121 (vagina)",
                "units": "Voxel indices / binary spatial ellipsoidal masks",
                "frame": "Voxel grid space (D, H, W) assumed to be Z, Y, X (actually X, Y, Z in nibabel)",
                "shape": "(Nx, Ny, Nz)",
                "transform": "Synthetic geometric ellipsoid rasterization relative to bladder (21) and sacrum (23)"
            },
            {
                "stage": "Stage 3: Target Centroid Extraction & Physical Conversion",
                "file": "sharon/pointcloud_sampler.py",
                "class_func": "_worker_normalized_case()",
                "inputs": "segmentation.nii.gz, NIfTI affine matrix",
                "outputs": "raw_centroids: float32 array (121, 3)",
                "units": "Millimeters (mm), but corrupted by axis swap",
                "frame": "CORRUPTED: X and Z axes swapped before affine multiplication",
                "shape": "(121, 3)",
                "transform": "CRITICAL BUG: com = ndi.center_of_mass(seg); cz, cy, cx = com; phys = (affine @ [cx, cy, cz, 1])[:3]. In nibabel, com is (x, y, z). Setting cz=x, cx=z swapped in-plane X and slice Z."
            },
            {
                "stage": "Stage 4: Skin Surface Generation",
                "file": "sharon/pointcloud_sampler.py",
                "class_func": "load_canonical_surface_points() & _worker_normalized_case()",
                "inputs": "Canonical template mesh (sharon/outputs/meshes/skin.obj) or random ellipsoid",
                "outputs": "raw_points: float32 array (4096, 3)",
                "units": "Millimeters (mm), stretched to match warped target bounds",
                "frame": "Artificial: Fitted to min/max of corrupted target coordinates",
                "shape": "(4096, 3)",
                "transform": "raw_points = (base_skin - ref_center)/ref_scale * scale_ras + center_ras + N(0, 0.5)"
            },
            {
                "stage": "Stage 5: Spatial Normalization",
                "file": "sharon/pointcloud_sampler.py",
                "class_func": "_worker_normalized_case()",
                "inputs": "raw_points (4096, 3), raw_centroids (121, 3)",
                "outputs": "norm_points (4096, 3) in [-1, 1], norm_centroids (121, 3) in [-1, 1], center_c (3,), scale_s (3,)",
                "units": "Dimensionless normalized coordinates",
                "frame": "Unit bounding box [-1, 1]^3 centered at torso midpoint",
                "shape": "points: (4096, 3), centroids: (121, 3), center: (3,), scale: (3,)",
                "transform": "c = (min + max)/2; s = (max - min)/2; norm = (raw - c) / s"
            },
            {
                "stage": "Stage 6: Batch Loading & Training Augmentation",
                "file": "sharon/dataset.py",
                "class_func": "PointCloudOrganDataset.__getitem__()",
                "inputs": "sharon/dataset/pointclouds_450.pt",
                "outputs": "Dict: points, centroids, sex_prior, mask, center, scale",
                "units": "Dimensionless for points/centroids; mm for center/scale",
                "frame": "Unit bounding box [-1, 1]^3",
                "shape": "points: (4096, 3), centroids: (121, 3)",
                "transform": "Gaussian coordinate jitter: pts = clamp(pts + N(0, 0.005), -1.2, 1.2)"
            },
            {
                "stage": "Stage 7: Neural Inference & Physical De-normalization",
                "file": "sharon/evaluate_evidential.py (and evaluate_same.py / evaluate_gnn.py)",
                "class_func": "evaluate_evidential_test_set()",
                "inputs": "Model normalized predictions gamma in [-1, 1]^3, patient center, patient scale",
                "outputs": "preds_mm, ctr_mm, dist_mm",
                "units": "Millimeters (in warped coordinate system)",
                "frame": "De-normalized space: preds_mm = gamma * scale + center",
                "shape": "preds_mm: (B, 121, 3), dist_mm: (B, 121)",
                "transform": "Linear affine reversal: preds * scale + center; dist_mm = norm(preds_mm - ctr_mm, dim=-1)"
            }
        ]

        for s in stages:
            f.write(f"### {s['stage']}\n")
            f.write(f"- **File Path**: `{s['file']}`\n")
            f.write(f"- **Function / Class**: `{s['class_func']}`\n")
            f.write(f"- **Inputs**: {s['inputs']}\n")
            f.write(f"- **Outputs**: {s['outputs']}\n")
            f.write(f"- **Coordinate Units**: {s['units']}\n")
            f.write(f"- **Coordinate Frame**: {s['frame']}\n")
            f.write(f"- **Tensor Shape**: `{s['shape']}`\n")
            f.write(f"- **Transformation Applied**: {s['transform']}\n\n")

    print(f"[Done] Generated pipeline map: {out_md}")

if __name__ == "__main__":
    generate_pipeline_map()
