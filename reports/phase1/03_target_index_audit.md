# Target Index Integrity Audit

## Final Status: **TARGET INDEXING CONSISTENT**

### Summary of Verifications
1. **Unique Organ Names**: Exactly 121 unique anatomical names in `sharon/labels.py`.
2. **Continuous 1-to-1 Mapping**: `TOTAL_CLASSES_121` continuously maps keys to 1..121 without gaps.
3. **0-indexed Tensor Alignment**: Array index `i` maps to `TOTAL_CLASSES_121` label `i + 1` across all dataset loaders.
4. **Dataset Alignment**: `pointclouds_450.pt` `organ_names` list is identical to `labels.py` `ORGAN_NAMES`.
5. **Skeletal / Soft-Tissue Disjoint Partition**: 63 skeletal + 58 soft-tissue indices strictly partition `{0..120}`.
6. **Biological Sex Indices**: Male-specific organ `prostate` (index 21) and female-specific organs `uterus` (117), `ovary_left` (118), `ovary_right` (119), and `vagina` (120) strictly match `MALE_SPECIFIC_INDICES` and `FEMALE_SPECIFIC_INDICES`.

### Verified Code Locations
- `sharon/labels.py`: Lines 5-135 (`TOTAL_CLASSES_121`), Lines 137-147 (`ORGAN_NAMES`), Lines 150-175 (`SKELETAL_CLASSES`, `SOFT_TISSUE_CLASSES`).
- `sharon/pointcloud_sampler.py`: Lines 73-76 (`ndi.center_of_mass(..., index=np.arange(1, num_organs + 1))` mapped to `org_idx`).
- `sharon/dataset.py`: Lines 28-40 (`pointclouds_450.pt` indexing).
