# Zero Internal Reference Audit — System 1 Deployment Pipeline

> [!IMPORTANT]
> **System 1 is confirmed EXTERNAL-ONLY at inference.**
> No CT-derived anatomical landmarks, T12 spine coordinates, vertebral segmentations,
> internal bone segmentations, organ masks, or any scan-internal information enters
> the canonical frame computation at test time.

---

## Component-by-Component Audit

| Pipeline Component | System 0 (Retrospective) | System 1 (Deployable) | Reference Type |
|---|---|---|---|
| **Surface Point Cloud** | ✅ External 3D surface geometry | ✅ External 3D surface geometry | EXTERNAL ONLY |
| **X-Axis Origin ($C_x$)** | Historical canonical alignment (T12/spine-derived) | $\text{median}(P_x)$ — sagittal midline from external surface | EXTERNAL ONLY |
| **Y-Axis Origin ($C_y$)** | Historical canonical alignment (T12/spine-derived) | $0.5 \times (\min P_y + \max P_y) - 56.60\text{ mm}$ — population dorsal offset from external surface bbox | EXTERNAL ONLY |
| **Z-Axis Origin ($C_z$)** | Historical canonical alignment (T12/spine-derived) | Fixed at $0.0\text{ mm}$ (stabilised anchor) | EXTERNAL ONLY |
| **$S_{\text{global}}$** | $500\text{ mm}$ | $500\text{ mm}$ | CONSTANT |
| **Neural Network** | Phase10R C4_Proposed (seeds 42, 43, 44) | Phase10R C4_Proposed (same checkpoints) | FROZEN |
| **Atlas Query Coordinates** | Training population mean (V3 train only) | Same training population mean | FROZEN |
| **Ensemble** | Unweighted coordinate mean of 3 seeds | Unweighted coordinate mean of 3 seeds | FROZEN |

### V2 / V3 System 0 Internal Reference Disclosure

> [!WARNING]
> **System 0 on V2 and V3 uses CT-derived internal reference.**
> The historical canonical alignment (`apply_canonical_alignment.py`) placed the coordinate
> origin at the **dorsal vertebral column (T12 spine level)**. This is an **INTERNAL**
> anatomical landmark derived from CT segmentation. System 0 is therefore a
> **retrospective internally registered condition**, NOT a deployable optical-only system.

### AMOS System 0 Internal Reference Disclosure

> [!WARNING]
> **System 0 on AMOS uses naive bounding box centering.**
> Phase 11 AMOS preprocessing subtracted the raw bounding box midpoint, which is
> **incompatible** with the V3 T12-based coordinate convention. This created the
> $+46.22\text{ mm}$ anterior displacement artifact.

### System 1 Verification Summary

- **T12 coordinates used at inference:** NO
- **Spine centroid used at inference:** NO
- **Vertebral segmentation used at inference:** NO
- **Internal bone segmentation used at inference:** NO
- **Organ masks used at inference:** NO
- **CT anatomical landmarks used at inference:** NO
- **Population offset constant ($-56.60\text{ mm}$):** Derived from V3 training data statistics only, NOT from any individual patient's internal anatomy at inference time.

**VERDICT: System 1 is DEPLOYMENT-COMPATIBLE (EXTERNAL ONLY).**
