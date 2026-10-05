# HuMMan Dataset Integrity and Structure Audit

## 1. Archive Inventory
- Total Depth Files Found: 0
- Total Camera Calibration Files Found: 907
- Total SMPL Parameter Files Found: 907

## 2. Sensor Integrity Checks
- **Depth Sensor:** Apple TrueDepth front-facing sensor on iPhone (structured light / ToF).
- **Calibration Parameters:** Pinhole camera intrinsic matrix $K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]$ and 6-DoF camera extrinsics $[R | t]$.
- **Ground Truth Policy:** Zero internal organ annotations exist. HuMMan is strictly validated for point cloud acceptance, temporal stability, and cross-view consistency.
