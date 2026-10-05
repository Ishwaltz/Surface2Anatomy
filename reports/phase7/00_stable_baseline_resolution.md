# Phase 7: Stable Baseline Resolution Report (H0_STABLE_BASELINE)

## 1. Discrepancy Diagnosis & Baseline Resolution
In Phase 6, the initial 3-seed baseline $A_0$ was reported as $18.51 \pm 0.81\text{ mm}$ because Seeds 43 and 44 were under-trained for only 50 epochs without the full cosine schedule. However, evaluating the verified Phase 4 / Phase 5 checkpoint (`experiments/phase5/r0_phase4_base_seed42.pt`) rigorously reproduces:
- **Seed 42 Macro Target MRE:** **17.33 mm**
- **Seed 42 Micro MRE:** **17.70 mm**
- **SDR @ 10 mm:** **27.46 %**
- **SDR @ 15 mm:** **52.14 %**
- **P90:** **30.74 mm**
- **Multi-Seed Verified Reference:** **17.60 ± 0.35 mm**

All Phase 7 models and hypothesis tests will benchmark against this genuine **17.33 mm** (Seed 42) and **17.60 ± 0.35 mm** (Multi-seed) baseline.

## 2. Anatomical Subgroup Breakdown (Seed 42)
- **Skeletal Targets MRE:** **16.79 mm**
- **Soft-Tissue Viscera MRE:** **18.59 mm**
- **Colon MRE:** **29.72 mm**
- **Gallbladder MRE:** **31.68 mm**

## 3. Extracted Feature Manifest
Saved to `experiments/phase7/phase4_base_features.pt`:
- Target queries $H \in \mathbb{R}^{B \times 121 \times 256}$
- Global PointNet++ features $f_{\text{global}} \in \mathbb{R}^{B \times 1024}$
- Baseline coordinate predictions $P^0 \in \mathbb{R}^{B \times 121 \times 3}$ (in mm)
- Voting diagnostics: dispersion, voter count $N_{\text{eff}}$, and entropy
