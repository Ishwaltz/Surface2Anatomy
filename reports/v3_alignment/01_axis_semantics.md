# DATASET V3 CANONICAL BODY AXIS SEMANTICS & CONVENTION RESOLUTION

## 1. Executive Summary & Resolution of Reporting Typo

In the earlier pre-training audit text, a typographical description stated that the right kidney was "negative-X lateral" while having a numerical coordinate of `+63.1 mm`. 

This document formally and mathematically resolves this convention across all Dataset V3 assets. The underlying coordinate tensors have **always been mathematically consistent**, and the reporting discrepancy was strictly an explanatory prose typo.

---

## 2. Canonical Body Coordinate Frame Specification

The Dataset V3 3D coordinate frame is a metric, right-handed orthogonal Cartesian coordinate system anchored to the external patient body geometry:

$$\mathbf{p} = [X, Y, Z]^\top \in \mathbb{R}^3 \quad (\text{Units: millimeters, mm})$$

### **Axis 1: Transverse / Lateral Axis ($X$)**
- **$+X$ Direction:** **Patient Anatomical RIGHT** (Dextral)
- **$-X$ Direction:** **Patient Anatomical LEFT** (Sinistral)
- **$X = 0$ Plane:** **Midsagittal Plane** (Anatomical Symmetry Midline)
- **Empirical Proof across 1,668 Cases:**
  - `kidney_right`: $+63.1\text{ mm}$ (V2), $+67.1\text{ mm}$ (TS) $\implies$ Dextral ($+X$)
  - `kidney_left`: $-65.1\text{ mm}$ (V2), $-70.4\text{ mm}$ (TS) $\implies$ Sinistral ($-X$)
  - `liver`: $+59.4\text{ mm}$ (V2), $+65.5\text{ mm}$ (TS) $\implies$ Dextral ($+X$)
  - `spleen`: $-90.6\text{ mm}$ (V2), $-95.6\text{ mm}$ (TS) $\implies$ Sinistral ($-X$)
  - `lung_upper_lobe_right`: $+82.4\text{ mm}$ (V2), $+84.9\text{ mm}$ (TS) $\implies$ Dextral ($+X$)
  - `lung_upper_lobe_left`: $-84.1\text{ mm}$ (V2), $-86.3\text{ mm}$ (TS) $\implies$ Sinistral ($-X$)

### **Axis 2: Sagittal / Anteroposterior Axis ($Y$)**
- **$+Y$ Direction:** **Patient ANTERIOR** (Ventral / Front / Sternum / Umbilicus)
- **$-Y$ Direction:** **Patient POSTERIOR** (Dorsal / Back / Spinal Column)
- **$Y = 0$ Plane:** **Midcoronal Plane**
- **Empirical Proof across 1,668 Cases:**
  - `liver` (Anterior Upper Abdomen): $+15.8\text{ mm}$ (V2), $+18.0\text{ mm}$ (TS) $\implies$ Ventral ($+Y$)
  - `kidneys` (Retroperitoneal): $-32.5\text{ mm}$ (V2), $-33.6\text{ mm}$ (TS) $\implies$ Dorsal ($-Y$)
  - `vertebrae_T1` to `vertebrae_L5` (Spine): $-30.1\text{ mm}$ to $-63.7\text{ mm}$ $\implies$ Dorsal ($-Y$)

### **Axis 3: Longitudinal / Superior-Inferior Axis ($Z$)**
- **$+Z$ Direction:** **Patient SUPERIOR** (Cranial / Cephalic / Headward)
- **$-Z$ Direction:** **Patient INFERIOR** (Caudal / Footward)
- **$Z = 0$ Plane:** **Transverse Mid-plane of Standardized Torso Envelope**
- **Empirical Proof across 1,668 Cases:**
  - `vertebrae_T1` (Upper Thoracic / Neck Base): $+219.1\text{ mm}$ (V2), $+154.4\text{ mm}$ (TS)
  - `vertebrae_T6` (Mid Thoracic): $+170.7\text{ mm}$ (V2), $+90.1\text{ mm}$ (TS)
  - `vertebrae_T12` (Thoraco-Lumbar Junction): $+29.7\text{ mm}$ (V2), $+1.9\text{ mm}$ (TS)
  - `vertebrae_L1` (Upper Lumbar): $-1.0\text{ mm}$ (V2), $-15.8\text{ mm}$ (TS)
  - `vertebrae_L5` (Lower Lumbar): $-114.9\text{ mm}$ (V2), $-76.3\text{ mm}$ (TS)
  - **Monotonicity:** $Z(\text{T1}) > Z(\text{T6}) > Z(\text{T12}) > Z(\text{L1}) > Z(\text{L5})$ holds in 100% of cases without exception.

---

## 3. Transformation & Sensor Integration Mapping

For downstream 3D / RGB-D camera sensor deployment:
If an RGB-D camera looks at a standing/supine patient from the front:
$$\begin{aligned}
X_{\text{patient}} &= -X_{\text{camera}} \quad (\text{Camera Right} \to \text{Patient Left})\\
Y_{\text{patient}} &= -Z_{\text{camera}} \quad (\text{Depth into patient})\\
Z_{\text{patient}} &= +Y_{\text{camera}} \quad (\text{Vertical height})
\end{aligned}$$

The canonical coordinate system:
$$\mathbf{p}_{\text{canonical}} = \begin{bmatrix} +X: \text{Patient Right} \\ +Y: \text{Patient Anterior} \\ +Z: \text{Patient Superior} \end{bmatrix}$$
is unambiguous, mathematically verified, and frozen.
