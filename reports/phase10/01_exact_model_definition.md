# EXACT BEST TARGET-QUERY MODEL SPECIFICATION (PART A)

**Document:** `reports/phase10/01_exact_model_definition.md`  
**Dataset Version:** `Dataset V3.0.0-FROZEN`  
**Model Class:** `sharon.model_target_query.TargetQueryTransformerDecoder`  
**Encoder Class:** `sharon.model_target_query.MultiScaleSurfacePointNet2Encoder`  
**Inspection Date:** 2026-09-12  
**Verification Method:** Direct python introspection and PyTorch parameter inspection from codebase.

---

## 1. Complete Architectural Blueprint

### A. PointNet++ Multi-Scale Surface Encoder
* **Input Representation:** $4096 \times 3$ pure external skin surface coordinates $(x, y, z)$. No normals, no body dimensions, no scan metadata, zero CT voxel intensities.
* **Normalization:** Fixed global isotropic scaling $S_{\text{global}} = 500.0\text{ mm}$ after rigid body centering $(P - C_{\text{body}})$, where $C_{\text{body}} = \frac{1}{2}(\min P + \max P)$.
* **Hierarchical Set Abstraction (SA) Levels:**
  1. **SA1 (Fine Surface Geometry):**
     * Subsampled Points ($N_1$): $1024$
     * Ball Query Radius ($r_1$): $0.20$ (corresponding to $100.0\text{ mm}$ in physical space)
     * Neighbors per Ball ($K_1$): $32$
     * Channel Mapping: $3 \to 64 \to 64 \to 128$
     * Output Dimensions: $(B, 1024, 3)$ centroids, $(B, 1024, 128)$ feature channels
     * Parameter Count: $13,248$
  2. **SA2 (Mid-Scale Surface Geometry):**
     * Subsampled Points ($N_2$): $256$
     * Ball Query Radius ($r_2$): $0.40$ (corresponding to $200.0\text{ mm}$ in physical space)
     * Neighbors per Ball ($K_2$): $32$
     * Channel Mapping: $(128 + 3) \to 128 \to 128 \to 256$
     * Output Dimensions: $(B, 256, 3)$ centroids, $(B, 256, 256)$ feature channels
     * Parameter Count: $67,456$
  3. **SA3 (Coarse-Scale Torso Topology):**
     * Subsampled Points ($N_3$): $64$
     * Ball Query Radius ($r_3$): $0.80$ (corresponding to $400.0\text{ mm}$ in physical space)
     * Neighbors per Ball ($K_3$): $32$
     * Channel Mapping: $(256 + 3) \to 256 \to 256 \to 512$
     * Output Dimensions: $(B, 64, 3)$ centroids, $(B, 64, 512)$ feature channels
     * Parameter Count: $265,984$
  4. **SA4 (Global Torso Pooling):**
     * Subsampled Points: All $64$ points pooled to $1$ global center
     * Grouping: `group_all = True`
     * Channel Mapping: $(512 + 3) \to 512 \to 512 \to 1024$
     * Output Dimensions: $(B, 1, 3)$ center, $(B, 1024)$ global feature vector
     * Parameter Count: $1,056,256$
* **Total Encoder Parameters:** **1,402,944**

---

### B. Surface Memory Token Matrix ($F_s \in \mathbb{R}^{M \times d}$)
* **Total Memory Tokens ($M$):** **320 tokens**
* **Token Feature Dimension ($d$):** **256**
* **Constituent Encoder Levels:**
  * Level 2 (SA2 Mid-scale): $256$ tokens ($256$-d feature projected via `proj_l2`: Linear($256 \to 256$))
  * Level 3 (SA3 Coarse-scale): $64$ tokens ($512$-d feature projected via `proj_l3`: Linear($512 \to 256$))
* **Scale Disambiguation Embedding:** Learnable scale vector (`scale_embed = nn.Embedding(2, 256)`), where weight[0] is added to $L_2$ and weight[1] is added to $L_3$.
* **Continuous 3D Positional Encoding:** 2-layer MLP operating on concatenated spatial centroids $(x, y, z) \in \mathbb{R}^{320 \times 3}$:
  $$\text{PE}_{\text{surf}}(x, y, z) = \text{Linear}_{128 \to 256}(\text{ReLU}(\text{Linear}_{3 \to 128}(x, y, z)))$$
* **Mathematical Definition of Surface Token Matrix:**
  $$F_s = \begin{bmatrix} W_{L2} f_{L2} + e_{\text{scale}, 0} \\ W_{L3} f_{L3} + e_{\text{scale}, 1} \end{bmatrix} + \text{PE}_{\text{surf}}(X_{\text{centroids}}) \in \mathbb{R}^{B \times 320 \times 256}$$

---

### C. Target-Query Transformer Decoder
* **Number of Anatomical Targets:** 117 model slots (evaluated on the 104 primary benchmark targets).
* **Target Embeddings:** Learnable dictionary `target_embed = nn.Embedding(117, 256)`.
* **Canonical Atlas Coordinate Reference ($a_k$):** Fixed training-set average centroid buffer $C_{\text{atlas}} \in \mathbb{R}^{117 \times 3}$ in model space.
* **Atlas Positional Encoding:** 2-layer MLP matching the surface PE structure:
  $$\text{PE}_{\text{atlas}}(a_k) = \text{Linear}_{128 \to 256}(\text{ReLU}(\text{Linear}_{3 \to 128}(a_k)))$$
* **Initial Query Formulation:**
  $$q_{k, 0} = E_{\text{target}}(k) + \text{PE}_{\text{atlas}}(a_k) \in \mathbb{R}^{B \times 117 \times 256}$$
* **Decoder Layers:** 4 identical stacked Transformer Cross-Attention layers.
* **Attention Heads:** 8 heads ($d_{\text{head}} = 32$).
* **Cross-Attention Mechanism:**
  $$Q_k = W_Q q_k, \quad K = W_K F_s, \quad V = W_V F_s$$
  $$\alpha_{kj} = \text{softmax}_j \left( \frac{Q_k K_j^\top}{\sqrt{d}} + b_{kj}^{\text{geo}} \right)$$
  $$h_k = \sum_{j=1}^{320} \alpha_{kj} V_j$$
* **Feed-Forward Networks (FFN):** Linear($256 \to 1024$) $\to$ ReLU $\to$ Dropout($p=0.05$) $\to$ Linear($1024 \to 256$).
* **Normalization & Residuals:** Pre-LN/Post-LN residual connections with `LayerNorm(256)`.
* **Total Decoder Parameters:** **3,487,235**

---

### D. Coordinate Residual Head & Physical Output
* **Coordinate Head Architecture:**
  $$\Delta p_k = \text{Linear}_{128 \to 3}(\text{ReLU}(\text{LayerNorm}_{128}(\text{Linear}_{256 \to 128}(q_{k, 4}))))$$
* **Final Model Centroid Prediction (Model Space):**
  $$\hat{p}_k = a_k + \Delta p_k$$
* **Physical Coordinate Denormalization (Physical Millimeters):**
  $$\hat{P}_{\text{world}, k} = (\hat{p}_k \times S_{\text{global}}) + C_{\text{body}} = (\hat{p}_k \times 500.0) + C_{\text{body}}$$

---

### E. Parameter Inventory Table (Exact Introspection)

| Component | Sub-Module | Layer Specification | Parameter Count |
| :--- | :--- | :--- | :---: |
| **Encoder** | `sa1` | Conv2d(3, 64) $\to$ Conv2d(64, 64) $\to$ Conv2d(64, 128) | 13,248 |
| **Encoder** | `sa2` | Conv2d(131, 128) $\to$ Conv2d(128, 128) $\to$ Conv2d(128, 256) | 67,456 |
| **Encoder** | `sa3` | Conv2d(259, 256) $\to$ Conv2d(256, 256) $\to$ Conv2d(256, 512) | 265,984 |
| **Encoder** | `sa4` | Conv2d(515, 512) $\to$ Conv2d(512, 512) $\to$ Conv2d(512, 1024) | 1,056,256 |
| **Encoder Total** | — | — | **1,402,944** |
| **Decoder** | `proj_l2` | Linear(256, 256) + bias | 65,792 |
| **Decoder** | `proj_l3` | Linear(512, 256) + bias | 131,328 |
| **Decoder** | `scale_embed` | Embedding(2, 256) | 512 |
| **Decoder** | `pe_mlp` | Linear(3, 128) $\to$ Linear(128, 256) | 33,536 |
| **Decoder** | `target_embed`| Embedding(117, 256) | 29,952 |
| **Decoder** | `atlas_pe_mlp`| Linear(3, 128) $\to$ Linear(128, 256) | 33,536 |
| **Decoder** | `cross_attns` | 4x MultiheadAttention(256, 8 heads) | 1,052,672 |
| **Decoder** | `norms1` | 4x LayerNorm(256) | 2,048 |
| **Decoder** | `ffns` | 4x [Linear(256, 1024) $\to$ Linear(1024, 256)] | 2,102,272 |
| **Decoder** | `norms2` | 4x LayerNorm(256) | 2,048 |
| **Decoder** | `coord_head` | Linear(256, 128) $\to$ LayerNorm(128) $\to$ Linear(128, 3)| 33,539 |
| **Decoder Total** | — | — | **3,487,235** |
| **GRAND TOTAL** | — | — | **4,890,179** |

---

### F. Training Objective & Optimization Recipe
* **Loss Objective:** Masked Smooth L2 Radial MRE loss on valid benchmark targets:
  $$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B \sum_{k \in \mathcal{T}_{\text{bench}}} m_{i, k} \sqrt{\|\hat{p}_{i, k} - p_{i, k}\|^2 + \epsilon^2}}{\sum_{i=1}^B \sum_{k \in \mathcal{T}_{\text{bench}}} m_{i, k}}$$
  where $\epsilon = 1 / S_{\text{global}} = 0.002$ ($1.0\text{ mm}$ in physical units).
* **Missing Target Handling:** Target matrix NaNs replaced with 0.0 prior to gradient computation via `torch.nan_to_num` with zeroed loss mask.
* **Optimizer:** AdamW with split parameter groups:
  * Encoder LR: $2 \times 10^{-4}$
  * Decoder LR: $5 \times 10^{-4}$
  * Weight Decay: $1 \times 10^{-4}$
* **Learning Rate Schedule:** CosineAnnealingLR ($T_{\text{max}} = 65$ epochs)
* **Batch Size:** 16
* **Epochs:** 65
* **Data Augmentation:** Subtle Gaussian surface jitter ($\sigma = 0.5\text{ mm}$, equivalent to $0.001$ in normalized space) applied during training only.
* **Model Selection Criterion:** Lowest Validation Macro Target MRE across the 104 primary benchmark targets on the frozen 166-subject validation split.
