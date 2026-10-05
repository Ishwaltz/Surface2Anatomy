# Exact Preprocessing Trace Audit (5 V2 vs 5 AMOS Cases)

> [!IMPORTANT]
> **HARD GATE RESOLUTION: WHERE DOES THE +56.6 MM AP REFERENCE ORIGINATE?**
> **Verdict: STORED POINT-CLOUD CANONICALIZATION DISCREPANCY (Different Body Reference Definition).**
> - In Dataset V3 (`apply_canonical_alignment.py`), the stored point clouds were centered relative to the **dorsal vertebral column (spine, T12)**, NOT the bounding box midpoint. Because the human spine is posterior to the body centroid, the resulting point clouds had an intrinsic **$+56.60\text{ mm}$ mean bounding box midpoint** along the $Y$ (anterior-posterior) axis.
> - The model training loader (`FastPointDataset`) performed **NO additional centering**—it directly divided the stored coordinates by $S_{\text{global}} = 500\text{ mm}$. Thus, the neural network was trained on inputs with an intrinsic $+0.1132$ normalized anterior midpoint!
> - When AMOS was processed in Phase 11 (`04_amos_surface_and_gt.py`), it subtracted the **raw bounding box midpoint** ($C_{\text{body}} = (\min + \max)/2$), forcing the normalized input midpoint to **$0.0$**!
> - The model, expecting a dorsal reference frame where the abdomen extends forward into $+Y$, perceived the zero-centered AMOS body as being shifted posterior by $56.6\text{ mm}$, and accordingly predicted all internal organs displaced anteriorly by **$+46.22\text{ mm}$**.

---

## 1. Trace Walkthrough: 5 In-Domain V2 Cases

### Case `v3_v2_case_000` (Dataset V2 Source)
- **Raw World Bounding Box Midpoint:** `[19.61, -169.34, 246.50] mm`
- **Raw World Surface Centroid:** `[4.42, -195.81, 246.04] mm`
- **$C_{\text{body}}$ Stored / Used by Loader:** `[13.54, -210.73, 420.13] mm`
- **Stored Point Cloud Bbox Midpoint:** `[6.08, 41.39, -173.63] mm` $\leftarrow$ **[Y = +41.39 mm!]**
- **Stored Point Cloud Centroid:** `[-9.12, 14.92, -174.09] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0122, 0.0828, -0.3473]` $\leftarrow$ **[Y = +0.0828]**
- **Network Input Centroid (Normalized):** `[-0.0182, 0.0298, -0.3482]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1831, 0.0138, -0.0663]`
- **Denormalized Spleen World Coordinate:** `[-78.03, -203.81, 387.00] mm`

### Case `v3_v2_case_001` (Dataset V2 Source)
- **Raw World Bounding Box Midpoint:** `[17.97, 6.11, 1798.00] mm`
- **Raw World Surface Centroid:** `[18.48, -6.60, 1785.56] mm`
- **$C_{\text{body}}$ Stored / Used by Loader:** `[24.60, -44.82, 1669.72] mm`
- **Stored Point Cloud Bbox Midpoint:** `[-6.63, 50.94, 128.28] mm` $\leftarrow$ **[Y = +50.94 mm!]**
- **Stored Point Cloud Centroid:** `[-6.12, 38.22, 115.84] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[-0.0133, 0.1019, 0.2566]` $\leftarrow$ **[Y = +0.1019]**
- **Network Input Centroid (Normalized):** `[-0.0122, 0.0764, 0.2317]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1922, 0.0384, 0.0306]`
- **Denormalized Spleen World Coordinate:** `[-71.49, -25.63, 1685.01] mm`

### Case `v3_v2_case_002` (Dataset V2 Source)
- **Raw World Bounding Box Midpoint:** `[3.01, -6.93, 1463.99] mm`
- **Raw World Surface Centroid:** `[2.35, -20.72, 1462.04] mm`
- **$C_{\text{body}}$ Stored / Used by Loader:** `[5.42, -62.72, 1600.75] mm`
- **Stored Point Cloud Bbox Midpoint:** `[-2.42, 55.79, -136.77] mm` $\leftarrow$ **[Y = +55.79 mm!]**
- **Stored Point Cloud Centroid:** `[-3.08, 42.01, -138.71] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[-0.0048, 0.1116, -0.2735]` $\leftarrow$ **[Y = +0.1116]**
- **Network Input Centroid (Normalized):** `[-0.0062, 0.0840, -0.2774]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1907, 0.0296, 0.0068]`
- **Denormalized Spleen World Coordinate:** `[-89.94, -47.92, 1604.17] mm`

### Case `v3_v2_case_003` (Dataset V2 Source)
- **Raw World Bounding Box Midpoint:** `[-10.00, -12.90, 1307.51] mm`
- **Raw World Surface Centroid:** `[-10.09, -20.22, 1298.31] mm`
- **$C_{\text{body}}$ Stored / Used by Loader:** `[-10.47, -44.56, 1406.86] mm`
- **Stored Point Cloud Bbox Midpoint:** `[0.47, 31.66, -99.35] mm` $\leftarrow$ **[Y = +31.66 mm!]**
- **Stored Point Cloud Centroid:** `[0.39, 24.34, -108.55] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0009, 0.0633, -0.1987]` $\leftarrow$ **[Y = +0.0633]**
- **Network Input Centroid (Normalized):** `[0.0008, 0.0487, -0.2171]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1709, 0.0208, 0.0520]`
- **Denormalized Spleen World Coordinate:** `[-95.94, -34.15, 1432.84] mm`

### Case `v3_v2_case_004` (Dataset V2 Source)
- **Raw World Bounding Box Midpoint:** `[12.37, -175.36, 200.94] mm`
- **Raw World Surface Centroid:** `[3.86, -206.99, 209.16] mm`
- **$C_{\text{body}}$ Stored / Used by Loader:** `[21.18, -215.38, 384.85] mm`
- **Stored Point Cloud Bbox Midpoint:** `[-8.81, 40.02, -183.91] mm` $\leftarrow$ **[Y = +40.02 mm!]**
- **Stored Point Cloud Centroid:** `[-17.32, 8.39, -175.69] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[-0.0176, 0.0800, -0.3678]` $\leftarrow$ **[Y = +0.0800]**
- **Network Input Centroid (Normalized):** `[-0.0346, 0.0168, -0.3514]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1732, 0.0149, -0.0934]`
- **Denormalized Spleen World Coordinate:** `[-65.44, -207.92, 338.14] mm`

---

## 2. Trace Walkthrough: 5 External AMOS Cases (Phase 11 Preprocessing)

### Case `amos_0004` (AMOS-22 External Source)
- **Raw World Bounding Box Midpoint:** `[-10.16, -14.46, 1310.00] mm`
- **Raw World Surface Centroid:** `[-11.63, -46.54, 1300.56] mm`
- **$C_{\text{body}}$ Subtracted in Phase 11:** `[-10.16, -14.46, 1310.00] mm` (Forced Bbox Midpoint = 0)
- **Stored Point Cloud Bbox Midpoint:** `[0.39, 0.00, 0.07] mm` $\leftarrow$ **[Y = 0.00 mm vs V3's +56.6 mm!]**
- **Stored Point Cloud Centroid:** `[-1.48, -32.07, -9.44] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0008, 0.0000, 0.0001]` $\leftarrow$ **[Y = 0.0000]**
- **Network Input Centroid (Normalized):** `[-0.0030, -0.0641, -0.0189]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1592, 0.0173, 0.2092]`
- **Phase 11 Denormalized Spleen:** `[-89.78, -5.81, 1414.60] mm`
- **True Spleen Ground Truth World:** `[-99.01, -35.55, 1441.18] mm`
- **Raw Error Vector (Pred - GT):** `[+9.23, +29.73, -26.58] mm` $\leftarrow$ **[Notice dy = +29.7 mm!]**

### Case `amos_0011` (AMOS-22 External Source)
- **Raw World Bounding Box Midpoint:** `[-14.27, -14.46, 1425.50] mm`
- **Raw World Surface Centroid:** `[-0.42, -49.52, 1424.45] mm`
- **$C_{\text{body}}$ Subtracted in Phase 11:** `[-14.27, -14.46, 1425.50] mm` (Forced Bbox Midpoint = 0)
- **Stored Point Cloud Bbox Midpoint:** `[0.39, 0.00, 0.01] mm` $\leftarrow$ **[Y = 0.00 mm vs V3's +56.6 mm!]**
- **Stored Point Cloud Centroid:** `[13.86, -35.06, -1.05] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0008, 0.0000, 0.0000]` $\leftarrow$ **[Y = 0.0000]**
- **Network Input Centroid (Normalized):** `[0.0277, -0.0701, -0.0021]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1519, 0.0273, 0.2388]`
- **Phase 11 Denormalized Spleen:** `[-90.23, -0.79, 1544.89] mm`
- **True Spleen Ground Truth World:** `[-92.62, -48.70, 1574.60] mm`
- **Raw Error Vector (Pred - GT):** `[+2.39, +47.91, -29.72] mm` $\leftarrow$ **[Notice dy = +47.9 mm!]**

### Case `amos_0009` (AMOS-22 External Source)
- **Raw World Bounding Box Midpoint:** `[2.93, 0.79, 1708.50] mm`
- **Raw World Surface Centroid:** `[7.95, -9.14, 1720.34] mm`
- **$C_{\text{body}}$ Subtracted in Phase 11:** `[2.93, 0.79, 1708.50] mm` (Forced Bbox Midpoint = 0)
- **Stored Point Cloud Bbox Midpoint:** `[1.16, 0.00, 0.13] mm` $\leftarrow$ **[Y = 0.00 mm vs V3's +56.6 mm!]**
- **Stored Point Cloud Centroid:** `[5.02, -9.92, 11.84] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0023, 0.0000, 0.0003]` $\leftarrow$ **[Y = 0.0000]**
- **Network Input Centroid (Normalized):** `[0.0100, -0.0198, 0.0237]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1915, 0.0231, -0.1151]`
- **Phase 11 Denormalized Spleen:** `[-92.80, 12.35, 1650.93] mm`
- **True Spleen Ground Truth World:** `[-83.63, -28.95, 1671.40] mm`
- **Raw Error Vector (Pred - GT):** `[-9.17, +41.29, -20.47] mm` $\leftarrow$ **[Notice dy = +41.3 mm!]**

### Case `amos_0006` (AMOS-22 External Source)
- **Raw World Bounding Box Midpoint:** `[-7.68, 8.87, 1354.50] mm`
- **Raw World Surface Centroid:** `[1.42, -35.82, 1344.75] mm`
- **$C_{\text{body}}$ Subtracted in Phase 11:** `[-7.68, 8.87, 1354.50] mm` (Forced Bbox Midpoint = 0)
- **Stored Point Cloud Bbox Midpoint:** `[0.42, -0.42, 0.27] mm` $\leftarrow$ **[Y = -0.42 mm vs V3's +56.6 mm!]**
- **Stored Point Cloud Centroid:** `[9.10, -44.70, -9.75] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0008, -0.0008, 0.0005]` $\leftarrow$ **[Y = -0.0008]**
- **Network Input Centroid (Normalized):** `[0.0182, -0.0894, -0.0195]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1858, 0.0341, 0.2094]`
- **Phase 11 Denormalized Spleen:** `[-100.56, 25.91, 1459.20] mm`
- **True Spleen Ground Truth World:** `[-95.36, -35.22, 1491.89] mm`
- **Raw Error Vector (Pred - GT):** `[-5.20, +61.13, -32.69] mm` $\leftarrow$ **[Notice dy = +61.1 mm!]**

### Case `amos_0014` (AMOS-22 External Source)
- **Raw World Bounding Box Midpoint:** `[-20.51, -15.64, 1458.50] mm`
- **Raw World Surface Centroid:** `[-11.94, -51.84, 1454.80] mm`
- **$C_{\text{body}}$ Subtracted in Phase 11:** `[-20.51, -15.64, 1458.50] mm` (Forced Bbox Midpoint = 0)
- **Stored Point Cloud Bbox Midpoint:** `[0.28, 0.00, -0.16] mm` $\leftarrow$ **[Y = 0.00 mm vs V3's +56.6 mm!]**
- **Stored Point Cloud Centroid:** `[8.57, -36.20, -3.70] mm`
- **Loader Centering Applied:** **NONE** (Direct passthrough to $/ 500.0$)
- **Network Input Midpoint (Normalized):** `[0.0006, 0.0000, -0.0003]` $\leftarrow$ **[Y = 0.0000]**
- **Network Input Centroid (Normalized):** `[0.0171, -0.0724, -0.0074]`
- **Atlas Spleen Query (Normalized):** `[-0.1891, 0.0420, 0.0008]`
- **Predicted Spleen (Normalized):** `[-0.1674, 0.0136, 0.2363]`
- **Phase 11 Denormalized Spleen:** `[-104.18, -8.83, 1576.63] mm`
- **True Spleen Ground Truth World:** `[-91.18, -41.43, 1626.28] mm`
- **Raw Error Vector (Pred - GT):** `[-13.00, +32.60, -49.65] mm` $\leftarrow$ **[Notice dy = +32.6 mm!]**

---

## 3. Direct Mathematical Proof of the Coordinate Offset

1. **V3 Stored Definition:**
   $$\mathbf{P}_{\text{stored}} = \mathbf{P}_{\text{world}} - \mathbf{C}_{\text{spine}}$$
   where $\mathbf{C}_{\text{spine}}$ is the vertebral baseline predicted by the Ridge model.
   Because the spine is dorsal to the torso midpoint by $56.6\text{ mm}$, the bounding box midpoint of $\mathbf{P}_{\text{stored}}$ is:
   $$\text{Midpoint}_Y(\mathbf{P}_{\text{stored}}) = +56.6\text{ mm}$$
2. **Phase 11 AMOS Preprocessing:**
   $$\mathbf{P}_{\text{AMOS}} = \mathbf{P}_{\text{world}} - \mathbf{C}_{\text{bbox}}$$
   where $\mathbf{C}_{\text{bbox}} = 0.5(\min + \max)$. Thus:
   $$\text{Midpoint}_Y(\mathbf{P}_{\text{AMOS}}) = 0.0\text{ mm}$$
3. **The Offset:**
   $$\Delta Y = \mathbf{C}_{\text{bbox}} - \mathbf{C}_{\text{spine}} \approx +56.60\text{ mm}$$
   When the network (trained on $\mathbf{P}_{\text{stored}}$) receives $\mathbf{P}_{\text{AMOS}}$, every point is translated along $-Y$ by $56.6\text{ mm}$. The network interprets the entire anatomy as shifted forward relative to the coordinate origin, outputting predictions with a systematic $+46.22\text{ mm}$ anterior displacement.
