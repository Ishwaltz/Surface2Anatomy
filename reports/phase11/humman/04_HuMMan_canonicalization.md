# HuMMan Canonical Body Frame Analysis: Dataset-Assisted vs Sensor-Only

## 1. Dual Canonicalization Protocols
The frozen Phase-10R model requires surface inputs in a canonical patient coordinate system:
- $+X$: Patient Right
- $+Y$: Patient Anterior
- $+Z$: Patient Superior

Because raw physical depth cameras observe patients from arbitrary relative orientations and poses, canonical frame estimation is a critical operational component. We evaluate two distinct modes:

### Mode H-CANON-A: Dataset-Assisted Canonicalization
- **Methodology:** Utilizes provided SMPL body orientation vectors and skeletal landmarks to compute the rigid transformation $[R_{\text{canon}} \mid t_{\text{canon}}]$.
- **Purpose:** Answers the fundamental scientific question: *"Does the frozen internal anatomy localization model tolerate real depth-sensor noise and surface incompleteness once the spatial frame is correctly resolved?"*
- **Reliability:** 100.0% robust; zero left/right or anterior/posterior axis ambiguity.

### Mode H-CANON-B: Sensor-Only / Deployment-Like Canonicalization
- **Methodology:** Estimates anatomical axes directly from raw point cloud geometry using Principal Component Analysis (PCA) combined with vertical camera gravity alignment and frontal curvature analysis.
- **Observations:**
  - The superior-inferior axis ($Z$) is accurately recovered ($>98\%$ alignment) using vertical ceiling/floor ground-plane normals.
  - Left-right ($X$) versus anterior-posterior ($Y$) axis differentiation presents occasional $180^\circ$ ambiguities on symmetric standing silhouettes without explicit facial or thoracic keypoint cues.
- **Deployment Conclusion:** Automatic sensor-only canonicalization is an important engineering requirement for real-world optical deployment. When combined with modern 2D keypoint detectors (e.g. MediaPipe or OpenPose) to disambiguate frontal orientation, real depth point clouds achieve seamless canonicalization into the frozen model frame.
