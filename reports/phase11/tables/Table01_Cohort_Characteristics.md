# Table 1: Phase 11 Evaluation Cohort Characteristics

| Feature / Attribute | Dataset V3 (Internal Locked Test) | AMOS-22 (External Clinical Validation) | HuMMan (Real Physical Depth Validation) |
|---|---|---|---|
| **Primary Scientific Role** | In-distribution baseline & benchmark | Zero-shot anatomical generalization | Real-sensor stability & optical realism |
| **Acquisition Modality** | Multi-vendor CT (TotalSegmentator & KiTS) | Multi-vendor clinical CT & MRI | Apple TrueDepth (iPhone Depth Sensor) |
| **Clinical Institutions** | Univ. Hospital Basel & Univ. of Minnesota | Multi-center hospitals, Shenzhen, China | Multi-subject laboratory capture |
| **Geographic Provenance** | Switzerland & United States | Shenzhen, China | Singapore |
| **Subject Count (N)** | 143 held-out test subjects | 259 labeled subjects (200 CT, 59 MRI) | 35 subjects (390 evaluated frames) |
| **Internal Ground Truth** | 104 primary organ/bone centroids | 15 voxel-level abdominal organs | **NONE (Zero Internal Ground Truth Rule)** |
| **Surface Derivation** | Offline marching cubes on CT HU | Offline marching cubes on CT/MRI | Real-time optical depth back-projection |
| **Torso Completeness** | Whole-body & broad torso | Diagnostic abdominal coverage (FOV-A, B, C) | Partial anterior sensor viewpoint |
| **Evaluation Metrics** | Macro MRE, Median, SDR@5–40 mm | Macro MRE, Median, SDR@5–40 mm | NaN rate, Temporal Jitter, View Disagreement |
