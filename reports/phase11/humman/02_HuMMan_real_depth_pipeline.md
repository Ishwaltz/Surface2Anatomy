# HuMMan Real-Sensor Depth Pipeline & Mathematical Formulations

## 1. Sensor Specifications & Coordinate Back-Projection
The HuMMan dataset provides real physical depth measurements acquired from Apple TrueDepth sensors (iPhone front-facing sensor suite):
- **Sensor Resolution:** $192 \times 256$ pixels.
- **Physical Depth Units:** Verified `uint16` encoding in **millimetres (mm)**.
- **Intrinsic Camera Matrix:**
  $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  where $f_x \approx 1437.45$, $f_y \approx 1437.45$, $c_x \approx 964.96$, $c_y \approx 718.91$.

### Mathematical Back-Projection
For each valid depth pixel $(u, v)$ with depth value $Z = \text{depth}(u, v) \in [300, 3000]\text{ mm}$:
$$\begin{aligned}
X_{\text{cam}} &= \frac{(u - c_x) \cdot Z}{f_x} \\
Y_{\text{cam}} &= \frac{(v - c_y) \cdot Z}{f_y} \\
Z_{\text{cam}} &= Z
\end{aligned}$$

## 2. Anatomical Coordinate Canonicalization
The camera coordinate frame (+X right, +Y down, +Z away into scene) is transformed into the standard anatomical reference frame (+X Right, +Y Anterior, +Z Superior) required by the frozen Phase-10R model:
$$\begin{bmatrix} X_{\text{anat}} \\ Y_{\text{anat}} \\ Z_{\text{anat}} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{bmatrix} \begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \end{bmatrix}$$

## 3. Surface Outlier Removal & Sampling
1. **Distance Filtering:** Points outside $[300, 3000]\text{ mm}$ are rejected to eliminate background walls and near-lens artifacts.
2. **Statistical Outlier Rejection:** Points exceeding the 95th percentile Euclidean distance from the torso median are pruned.
3. **Body Centering:** The 3D bounding box midpoint $[X_{\text{mid}}, Y_{\text{mid}}, Z_{\text{mid}}]$ is subtracted.
4. **4096 Sampling & Normalization:** Exactly 4,096 points are sampled and normalized by $S_{\text{global}} = 500\text{ mm}$ matching the exact frozen training specification.
