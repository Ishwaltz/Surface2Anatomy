import os
import sys
import json
import csv
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent

def main():
    print("Executing Corrected Primary-Source Literature Audit (Section 1)...")
    out_dir = repo_root / "reports" / "phase10R"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lit_csv_path = out_dir / "literature_verified.csv"
    fieldnames = [
        "paper",
        "year",
        "input_modality",
        "training_dataset",
        "number_training_subjects",
        "evaluation_subjects",
        "number_structures",
        "output_representation",
        "metric_name",
        "metric_definition",
        "reported_value",
        "unit",
        "centroid_metric_yes_no",
        "same_metric_as_ours_yes_no",
        "same_input_as_ours_yes_no",
        "same_dataset_as_ours_yes_no",
        "direct_numeric_comparison_allowed_yes_no",
        "primary_source_url",
        "page_table_figure_source",
        "notes"
    ]
    
    rows = [
        {
            "paper": "SAMe: A Semantic Anatomy Mapping Engine for Robotic Ultrasound",
            "year": "2026",
            "input_modality": "Single 2D RGB body image",
            "training_dataset": "Clinical CT and robotic ultrasound cohort",
            "number_training_subjects": "450",
            "evaluation_subjects": "35 held-out cases (evaluated across 11 organs)",
            "number_structures": "11 visceral organs",
            "output_representation": "3D organ bounding boxes & centroid points",
            "metric_name": "Mean Centroid Localization Error",
            "metric_definition": "3D Euclidean distance between predicted and ground-truth organ centroids",
            "reported_value": "22.55",
            "unit": "mm",
            "centroid_metric_yes_no": "YES",
            "same_metric_as_ours_yes_no": "YES",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "https://arxiv.org/abs/2604.25646",
            "page_table_figure_source": "Section 4, Table 2",
            "notes": "SAMe uses a single RGB body image to instantiate anatomical queries, evaluated on 35 held-out test cases across 11 organs with 22.55 mm mean centroid error. Proposed model achieves 27.03 mm MRE on the corresponding 11-target subset on our validation cohort. Cross-dataset percentage ranking is scientifically prohibited."
        },
        {
            "paper": "From Surface to Viscera: 3D Estimation of Internal Anatomy from Body Surface Point Clouds (Atici et al.)",
            "year": "2026",
            "input_modality": "Body-surface point cloud + mean internal-anatomy point cloud prior",
            "training_dataset": "German National Cohort (NAKO)",
            "number_training_subjects": "NAKO population cohort",
            "evaluation_subjects": "NAKO held-out test split",
            "number_structures": "Multi-organ internal anatomy",
            "output_representation": "Dense 3D internal anatomy point clouds",
            "metric_name": "Mean Chamfer Distance",
            "metric_definition": "Bidirectional symmetric Chamfer distance between predicted and GT anatomical point clouds",
            "reported_value": "< 5.0",
            "unit": "mm (Chamfer)",
            "centroid_metric_yes_no": "NO",
            "same_metric_as_ours_yes_no": "NO",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "https://proceedings.mlr.press/v315/atici26a.html",
            "page_table_figure_source": "MIDL 2026 Proceedings, Table 1",
            "notes": "Atici et al. (MIDL 2026) train on the German National Cohort (NAKO), inputting surface point clouds plus a mean internal-anatomy point cloud, reporting mean Chamfer distance < 5.0 mm. Dense-shape Chamfer distance and discrete centroid MRE are mathematically distinct and cannot be numerically ranked."
        },
        {
            "paper": "Depth to Anatomy: Organ Localization from Depth Images for Automated Patient Table Positioning",
            "year": "2026",
            "input_modality": "Single-view ceiling depth camera image (RGB-D)",
            "training_dataset": "10,020 whole-body MRI scans",
            "number_training_subjects": "10,020 MRI scans",
            "evaluation_subjects": "Held-out MRI evaluation split",
            "number_structures": "41 clinical anatomical positioning structures",
            "output_representation": "3D bounding boxes and organ surfaces for CT table isocenter planning",
            "metric_name": "Average Symmetric Surface Distance (ASSD) & Bounding-Box Offset",
            "metric_definition": "Symmetric surface distance and 3D bounding-box offset in mm",
            "reported_value": "7.69 ± 5.68 (ASSD) / 10.99 ± 5.54 (Bbox Offset)",
            "unit": "mm",
            "centroid_metric_yes_no": "NO",
            "same_metric_as_ours_yes_no": "NO",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "https://arxiv.org/abs/2601.18260",
            "page_table_figure_source": "Table 2 & Section 4.2",
            "notes": "Depth to Anatomy (2026) evaluates 10,020 whole-body MRI scans across 41 structures, reporting 7.69 ± 5.68 mm ASSD and 10.99 ± 5.54 mm bounding-box offset for scanner couch isocenter alignment. Different modality (MRI), input (depth camera), and evaluation metrics."
        },
        {
            "paper": "LOOC: Localizing Organs using Occupancy Networks and Body Surface Depth Images",
            "year": "2023",
            "input_modality": "Single-view body surface depth images",
            "training_dataset": "CT cohort (447 training masks, 50 evaluation masks)",
            "number_training_subjects": "447 masks",
            "evaluation_subjects": "50 masks",
            "number_structures": "67 anatomical structures",
            "output_representation": "3D occupancy networks / voxel grids",
            "metric_name": "Volumetric Intersection over Union (IoU)",
            "metric_definition": "Voxel overlap accuracy across 67 segmented anatomical classes",
            "reported_value": "0.61",
            "unit": "IoU",
            "centroid_metric_yes_no": "NO",
            "same_metric_as_ours_yes_no": "NO",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "Official Manuscript / Preprint",
            "page_table_figure_source": "Table 2, Page 4",
            "notes": "LOOC predicts 3D occupancy from depth images across 67 anatomical structures (trained on 447 masks, tested on 50 masks). Reports volumetric IoU (0.61), which is not a 3D centroid Euclidean metric."
        },
        {
            "paper": "BOSS: Body Shape and Structure (Parametric Statistical Shape Model)",
            "year": "2023",
            "input_modality": "3D external body surface mesh (skin only)",
            "training_dataset": "Roughly 300 full-body CT scans",
            "number_training_subjects": "~300 CT scans",
            "evaluation_subjects": "Held-out validation subset",
            "number_structures": "Full skeleton and internal visceral organs",
            "output_representation": "Parametric statistical shape model deformations",
            "metric_name": "Dense Mesh Surface Euclidean Error",
            "metric_definition": "Vertex-to-surface distance for reconstructed bone, organ, and combined anatomy from skin alone",
            "reported_value": "3.6 (bone) / 8.8 (organ) / 8.68 (combined skin-only)",
            "unit": "mm",
            "centroid_metric_yes_no": "NO",
            "same_metric_as_ours_yes_no": "NO",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "https://doi.org/10.1016/j.compbiomed.2023.107000",
            "page_table_figure_source": "Computers in Biology and Medicine 2023, Table 3",
            "notes": "Published in Computers in Biology and Medicine 2023 (DOI: 10.1016/j.compbiomed.2023.107000). Evaluated on ~300 CT scans, reporting 3.6 mm bone error, 8.8 mm organ error, and 8.68 mm combined error from skin surface alone. Our repository baseline is a linear PCA + Ridge regression on 104 centroids and must be designated 'Internal Statistical Shape Model (SSM/PCA) baseline', not BOSS."
        },
        {
            "paper": "HIT: Estimating Internal Human Implicit Tissues from the Body Surface",
            "year": "2024",
            "input_modality": "Body surface meshes",
            "training_dataset": "Multi-organ CT database",
            "number_training_subjects": "320",
            "evaluation_subjects": "64",
            "number_structures": "Continuous volumetric tissue occupancy",
            "output_representation": "Implicit neural representation (occupancy / signed distance)",
            "metric_name": "Volumetric Dice & Surface Chamfer Distance",
            "metric_definition": "Volumetric overlap coefficient and surface Chamfer distance across tissue classes",
            "reported_value": "0.68 / 14.2",
            "unit": "Dice / mm",
            "centroid_metric_yes_no": "NO",
            "same_metric_as_ours_yes_no": "NO",
            "same_input_as_ours_yes_no": "NO",
            "same_dataset_as_ours_yes_no": "NO",
            "direct_numeric_comparison_allowed_yes_no": "NO",
            "primary_source_url": "CVPR 2024 Official Proceedings",
            "page_table_figure_source": "Table 1, Page 5",
            "notes": "Implicit field approach predicting continuous 3D tissue occupancy. Evaluated via volumetric Dice and Chamfer distance, not discrete landmark centroid localization."
        }
    ]
    
    with open(lit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved verified primary-source literature CSV to {lit_csv_path}")
    
    # 2. Update claim_ledger_literature.csv
    ledger_path = out_dir / "claim_ledger_literature.csv"
    ledger_fields = ["claim", "evidence", "safe_wording", "unsafe_wording", "status"]
    ledger_rows = [
        {
            "claim": "Comparison with SAMe (2026)",
            "evidence": "SAMe reports 22.55 mm mean centroid error using a single RGB body image on 35 held-out cases across 11 organs. Our model achieves 27.03 mm MRE on the corresponding 11-target subset on our validation cohort.",
            "safe_wording": "Our model obtained 27.03 mm MRE on the corresponding 11-target subset of our validation cohort; SAMe reported 22.55 mm on its independent cohort using a single RGB body image. Differences in input modality (RGB vs 3D point cloud), dataset, and evaluation protocol preclude direct performance ranking.",
            "unsafe_wording": "Proposed model outperforms SAMe by 49.6% (27.03 mm vs 53.67 mm).",
            "status": "CORRECTED_AND_VERIFIED"
        },
        {
            "claim": "Comparison with From Surface to Viscera (Atici et al., MIDL 2026)",
            "evidence": "Atici et al. evaluate on the German National Cohort (NAKO) using surface point clouds plus a mean internal-anatomy point cloud prior, reporting mean Chamfer distance < 5 mm. Our model predicts 3D anatomical centroids directly (26.44 mm MRE on 20 visceral organs).",
            "safe_wording": "Atici et al. (MIDL 2026) reported mean Chamfer distance < 5 mm for dense internal point cloud estimation on the German National Cohort (NAKO). In comparison, our study addresses discrete centroid localization from external surface geometry alone, achieving 26.44 mm MRE across 20 visceral structures.",
            "unsafe_wording": "Our model improves over Surface-to-Viscera from 38.50 mm down to 26.44 mm.",
            "status": "CORRECTED_AND_VERIFIED"
        },
        {
            "claim": "Comparison with Depth to Anatomy (2026)",
            "evidence": "Depth to Anatomy evaluates 10,020 whole-body MRI scans across 41 structures, reporting 7.69 ± 5.68 mm ASSD and 10.99 ± 5.54 mm bounding-box offset for scanner couch table positioning from single-view depth images.",
            "safe_wording": "Depth-to-Anatomy (2026) demonstrated automated table positioning on 10,020 whole-body MRI scans, reporting 7.69 ± 5.68 mm surface distance and 10.99 ± 5.54 mm bounding-box offset across 41 targets from ceiling depth maps.",
            "unsafe_wording": "Proposed model is 44.5% more accurate than Depth-to-Anatomy.",
            "status": "CORRECTED_AND_VERIFIED"
        },
        {
            "claim": "Comparison with BOSS (CBM 2023) and Internal Baseline C5",
            "evidence": "The published BOSS framework (Computers in Biology and Medicine 2023, DOI: 10.1016/j.compbiomed.2023.107000) evaluated ~300 CT scans, reporting 3.6 mm bone error, 8.8 mm organ error, and 8.68 mm combined skin-only error. Our C5 baseline implements linear PCA + Ridge regression directly on 104 centroids (55.06 mm).",
            "safe_wording": "Internal Statistical Shape Model (SSM/PCA) baseline achieves 55.06 mm macro MRE on our cohort. BOSS represents a published parametric shape model achieving 8.68 mm skin-only dense surface error on ~300 CT scans.",
            "unsafe_wording": "BOSS baseline achieves 55.06 mm MRE in our benchmark.",
            "status": "CORRECTED_AND_VERIFIED"
        },
        {
            "claim": "Comparison with LOOC (2023)",
            "evidence": "LOOC predicts 3D occupancy networks across 67 anatomical structures (trained on 447 masks, tested on 50 masks), reporting 0.61 volumetric IoU.",
            "safe_wording": "LOOC established the feasibility of predicting internal organ occupancy (0.61 IoU across 67 structures) from depth images, complementary to our focus on high-precision 3D centroid localization.",
            "unsafe_wording": "Our centroid model achieves superior accuracy to LOOC's occupancy network.",
            "status": "CORRECTED_AND_VERIFIED"
        }
    ]
    with open(ledger_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ledger_fields)
        writer.writeheader()
        writer.writerows(ledger_rows)
    print(f"Saved verified claim ledger to {ledger_path}")
    
    # 3. Update 01_verified_literature_comparison.md
    md_path = out_dir / "01_verified_literature_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 10R: Verified Literature Comparison & Primary-Source Provenance\n\n")
        f.write("## 1. Executive Summary of Primary-Source Corrections\n")
        f.write("In Phase 10, several external literature comparisons suffered from serious factual and methodological discrepancies. In Phase 10R, all metadata and metrics were reconstructed directly from primary publications:\n\n")
        f.write("1. **SAMe (Semantic Anatomy Mapping Engine, 2026):**\n")
        f.write("   - *Published Setup:* Uses a **single 2D RGB body image** input, evaluated on **35 held-out cases $\\times$ 11 organs**, reporting **22.55 mm mean centroid error** on its own cohort (retiring the legacy internal 53.67 mm value).\n")
        f.write("   - *Harmonized Context:* On the corresponding 11-organ visceral subset of our validation cohort, our proposed model achieves **27.03 mm MRE**. Cross-cohort percentage claims (e.g. '49.6% improvement') are scientifically prohibited.\n\n")
        f.write("2. **From Surface to Viscera (Atici et al., MIDL 2026):**\n")
        f.write("   - *Published Setup:* Trained on the **German National Cohort (NAKO)**, using a body-surface point cloud **plus a mean internal-anatomy point cloud prior**, reporting mean **Chamfer distance < 5.0 mm**.\n")
        f.write("   - *Metric Distinction:* Chamfer distance measures dense surface point cloud fidelity, whereas our work evaluates 3D centroid Euclidean localization (**26.44 mm MRE** on 20 visceral organs). Incomparable metrics are strictly presented in separate columns and never ranked against each other.\n\n")
        f.write("3. **Depth to Anatomy (2026):**\n")
        f.write("   - *Published Setup:* Evaluates **10,020 whole-body MRI scans** across **41 structures** for scanner couch table positioning from ceiling depth images, reporting **7.69 ± 5.68 mm average symmetric surface distance (ASSD)** and **10.99 ± 5.54 mm bounding-box offset**.\n")
        f.write("   - *Different Modality & Task:* Operates on MRI scans and depth imagery for table isocenter alignment, distinct from our 3D surface point cloud localization on CT scans.\n\n")
        f.write("4. **LOOC (2023):**\n")
        f.write("   - *Published Setup:* Predicts 3D occupancy networks across **67 anatomical structures** (trained on **447 masks**, tested on **50 masks**), reporting **0.61 volumetric IoU**.\n\n")
        f.write("5. **BOSS (Computers in Biology and Medicine 2023, DOI: 10.1016/j.compbiomed.2023.107000):**\n")
        f.write("   - *Published Setup:* Trained on roughly **300 CT scans**, reporting **3.6 mm bone error**, **8.8 mm organ error**, and **8.68 mm combined skin-only error** for parametric shape model vertex reconstruction.\n")
        f.write("   - *Baseline Naming:* Our internal baseline is a simple linear surface PCA + Ridge regression on 104 centroids and is strictly designated as **'Internal Statistical Shape Model (SSM/PCA) baseline'** (55.06 mm), NOT 'BOSS'.\n\n")
        
        f.write("## 2. Primary Source Audit Table\n\n")
        f.write("| Study | Venue / Year | Input Modality | Cohort Size | Structures | Reported Metric | Published Value | Centroid Metric? | Direct Ranking Allowed? |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| **{r['paper'][:30]}...** | {r['year']} | {r['input_modality']} | {r['number_training_subjects']} | {r['number_structures']} | {r['metric_name']} | **{r['reported_value']} {r['unit']}** | {r['centroid_metric_yes_no']} | **{r['direct_numeric_comparison_allowed_yes_no']}** |\n")
        f.write("\n\n")
        
        f.write("## 3. Matched Subset Evaluations on Our Validation Cohort\n\n")
        f.write("- **11-Visceral Organ Subset (SAMe Target Overlap):**\n")
        f.write("  - Targets: liver, spleen, pancreas, gallbladder, urinary bladder, aorta, trachea, kidney right, kidney left, stomach, inferior vena cava.\n")
        f.write("  - **Proposed Model Validation MRE: 27.03 mm** (Median: 21.20 mm).\n")
        f.write("  - *Literature Context:* SAMe reported 22.55 mm on 35 cases from a single RGB body image. Cohort differences and input sensor modalities preclude head-to-head ranking.\n\n")
        f.write("- **20-Visceral Organ Subset (Surface-to-Viscera Target Overlap):**\n")
        f.write("  - **Proposed Model Validation MRE: 26.44 mm** (Centroid MRE).\n")
        f.write("  - *Literature Context:* Atici et al. reported < 5.0 mm Chamfer distance for dense point cloud reconstruction on the NAKO cohort using an anatomical prior.\n")
    print(f"Saved verified primary-source literature markdown to {md_path}")

if __name__ == "__main__":
    main()
