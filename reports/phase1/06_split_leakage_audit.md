# Split Leakage & Duplicate Patient Content Audit

## Status: **CRITICAL DATA LEAKAGE DETECTED**

### 1. Patient ID Partition Disjointness
- Train Cases: 360
- Validation Cases: 45
- Test Cases: 45
- Train ∩ Validation ID Overlap: 0
- Train ∩ Test ID Overlap: 0
- Validation ∩ Test ID Overlap: 0

### 2. Physical Content SHA256 Hash Analysis
While the synthetic `case_XXX` ID strings are disjoint, exact cryptographic hashing of the underlying raw NIfTI CT image files reveals **duplicate patient volumes under different case IDs**:

| Duplicate Group | Split Distribution | Raw CT SHA256 Prefix | Finding |
| :--- | :--- | :---: | :--- |
| `case_096` (test), `case_412` (val) | **val ∩ test** | Verified | **CROSS-SPLIT LEAKAGE** |
| `case_100` (train), `case_408` (val) | **val ∩ train** | Verified | **CROSS-SPLIT LEAKAGE** |
| `case_144` (train), `case_407` (val) | **val ∩ train** | Verified | **CROSS-SPLIT LEAKAGE** |
| `case_219` (train), `case_406` (test) | **train ∩ test** | Verified | **CROSS-SPLIT LEAKAGE** |
| `case_344` (train), `case_402` (val) | **val ∩ train** | Verified | **CROSS-SPLIT LEAKAGE** |

### 3. Biological Inconsistency in Leaked Patients
- **`case_100` (Train) vs `case_408` (Val)**: Exact identical CT volume (SHA256 `968e99b7...`). In `case_100`, sex is labeled **Male (1)** with no female pelvic organs. In `case_408`, sex is labeled **Female (0)** with synthetic uterus, ovaries, and vagina injected into the segmentation mask! This injects identical physical anatomy into both Train and Validation under contradictory ground truth!
- **`case_219` (Train) vs `case_406` (Test)**: Exact identical CT volume (SHA256 `f0c90c74...`). Patient data from training leaked directly into held-out test evaluation.
