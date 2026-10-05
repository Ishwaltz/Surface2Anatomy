# AMOS-22 Surface Extraction & Ground Truth Visual Quality Control

## Overview
Visual quality control was performed across 40 randomly stratified cases:
- **20 CT Cases:** `amos_0001` through `amos_0025`
- **20 MRI Cases:** `amos_0507` through `amos_0538`

Each visual panel inspects:
1. **Axial Slice (Mid-Z):** Intensity image overlaid with extracted exterior skin surface contour (red for CT, cyan for MRI).
2. **Coronal Slice (Mid-Y):** Craniocaudal extent and anterior/posterior surface boundary tracking.
3. **Centered 3D Surface Point Cloud & GT Organ Centroids:** Verification of the 4,096 sampled exterior body surface points (colored by depth) against the 3D ground-truth organ centroids (red 'x' marks).

## Quality Audit Summary
- **Surface Integrity Gate:** 40 / 40 cases passed inspection without body clipping artifacts, boundary collapse, or bed contamination.
- **Centroid In-Body Verification:** 100% of ground-truth organ centroids lie strictly within the interior volume defined by the exterior body mesh.
- **Anatomical Consistency:** CT thresholding at $HU > -500$ and MRI background-adaptive Otsu/closing accurately captured physical skin boundaries without inclusion of external scanning apparatus or clothing.
- **QC Panels Directory:** `reports/phase11/qc/AMOS/`
