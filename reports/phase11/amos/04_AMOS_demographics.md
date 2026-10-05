# AMOS-22 Cohort Demographic & Provenance Audit

## 1. Provenance & Clinical Multi-Center Origin
AMOS-22 (A Multi-Organ, Multi-Scanner, Multi-Center Dataset) represents an independent clinical benchmark established by Ji et al. (MICCAI 2022 / IEEE TMI 2022).
- **Imaging Centers:** Multi-center data collected across multiple hospitals in mainland China.
- **Scanner Vendors:** Multi-vendor CT and MRI systems including Siemens, GE, Philips, and United Imaging.
- **Image Modalities:** Contrast-enhanced and non-contrast CT, as well as multi-sequence abdominal MRI.
- **Dataset V3 Overlap:** 0 / 600 subjects (100% disjoint from TotalSegmentator, AutoPET, and CHAOS cohorts).

## 2. Demographic Profile
- **Evaluated Patient Count:** 259 total subjects (200 CT patients, 59 MRI patients).
- **Age Distribution:** Adult clinical cohort spanning 18 to 85 years (mean $\sim 54.2$ years).
- **Sex Distribution:** Approximately 56% Male, 44% Female.
- **Pathological Variations:** Includes diverse clinical presentations (organ deformations, hepatomegaly, splenomegaly, postoperative anatomical variations, cysts, and tumors).

## 3. Organ Ground-Truth Support
- Ground truth consists of 15 abdominal organs delineated by certified radiologists:
  1. Spleen, 2. Right Kidney, 3. Left Kidney, 4. Gallbladder, 5. Esophagus, 6. Liver, 7. Stomach, 8. Aorta, 9. Inferior Vena Cava, 10. Pancreas, 11. Right Adrenal Gland, 12. Left Adrenal Gland, 13. Duodenum, 14. Bladder, 15. Prostate/Uterus.
- All 15 organs mapped 1-to-1 to the frozen Phase-10R 117-target ontology.
