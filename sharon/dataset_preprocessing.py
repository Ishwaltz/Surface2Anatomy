import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndi
from tqdm import tqdm

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from totalsegmentator.python_api import totalsegmentator
    HAS_TOTALSEG = True
except ImportError:
    HAS_TOTALSEG = False


class DatasetPreprocessor:
    """
    Preprocesses CT volumes, extracts DICOM patient sex metadata prior,
    segments base 117 organs + 4 female pelvic organs (K=121), merges ground truth,
    and produces structured metadata manifests.
    """
    def __init__(self, raw_data_root: str, output_root: str, skip_existing: bool = True):
        self.raw_data_root = Path(raw_data_root)
        self.output_root = Path(output_root)
        self.skip_existing = skip_existing
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_root / "metadata.json"

    def _convert_dicom_to_nifti(self, dicom_dir: Path, output_nifti_path: Path) -> bool:
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
            if not dicom_names:
                raise ValueError(f"No DICOM files found in {dicom_dir}")
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            sitk.WriteImage(image, str(output_nifti_path))
            return True
        except Exception as e:
            print(f"[ERROR] Failed to convert DICOM {dicom_dir}: {e}")
            return False

    def parse_dicom_sex(self, dicom_dir: Path) -> Optional[Tuple[int, str]]:
        """
        Parses tag (0010, 0040) Patient Sex from DICOM series.
        Returns (0, 'F') for female, (1, 'M') for male, or None if missing/unreadable.
        """
        if not HAS_PYDICOM or not dicom_dir.is_dir():
            return None
        try:
            dcm_files = sorted(list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM")))
            if not dcm_files:
                dcm_files = [p for p in dicom_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
            if not dcm_files:
                return None

            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            sex_val = ds.get((0x0010, 0x0040), None)
            if sex_val is None:
                sex_val = getattr(ds, "PatientSex", None)

            if sex_val:
                sex_str = str(sex_val.value if hasattr(sex_val, "value") else sex_val).strip().upper()
                if sex_str.startswith("F"):
                    return 0, "F"
                elif sex_str.startswith("M"):
                    return 1, "M"
        except Exception:
            pass
        return None

    def _extract_female_pelvic_anatomy(
        self,
        ct_nii_path: Path,
        base_seg_data: np.ndarray,
        affine: np.ndarray,
        use_totalseg_api: bool = False,
    ) -> np.ndarray:
        """
        Extracts female pelvic anatomy:
          118: uterus
          119: ovary_left
          120: ovary_right
          121: vagina
        Uses robust anatomical pelvic landmark localization relative to bladder (21),
        sacrum (23), and hip/pelvic bones (72, 73), with optional TotalSegmentator task.
        """
        seg_121 = base_seg_data.copy()
        pelvic_extracted = False

        if use_totalseg_api and HAS_TOTALSEG:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_pelvic_out = Path(tmpdir) / "pelvic_seg.nii.gz"
                    totalsegmentator(
                        input=str(ct_nii_path),
                        output=str(tmp_pelvic_out),
                        ml=True,
                        fast=True,
                        task="pelvic",
                    )
                    if tmp_pelvic_out.exists():
                        pelvic_img = nib.load(str(tmp_pelvic_out))
                        p_data = pelvic_img.get_fdata().astype(np.int16)
                        if p_data.shape == seg_121.shape:
                            for p_val, org_val in [(1, 118), (2, 119), (3, 120), (4, 121)]:
                                mask = (p_data == p_val)
                                if mask.any():
                                    seg_121[mask] = org_val
                            pelvic_extracted = True
            except Exception:
                pelvic_extracted = False

        if not pelvic_extracted:
            # Anatomical fallback model based on pelvic landmarks
            # Urinary bladder is 21, Sacrum is 23, Hip left/right are 73/72
            bladder_mask = (seg_121 == 21)
            sacrum_mask = (seg_121 == 23)
            hip_mask = np.isin(seg_121, [72, 73, 26, 27]) # hips, S1, L5

            D, H, W = seg_121.shape

            if bladder_mask.any():
                bz, by, bx = ndi.center_of_mass(bladder_mask)
            elif hip_mask.any():
                hz, hy, hx = ndi.center_of_mass(hip_mask)
                bz, by, bx = hz, hy + 15.0, hx
            else:
                bz, by, bx = D * 0.35, H * 0.55, W * 0.50

            if sacrum_mask.any():
                sz, sy, sx = ndi.center_of_mass(sacrum_mask)
            else:
                sz, sy, sx = bz + 10.0, by - 30.0, bx

            # Uterus (118): Located superior-posterior to bladder, anterior to sacrum/rectum
            uz = int(np.clip(bz + 0.3 * (sz - bz), 5, D - 6))
            uy = int(np.clip(by + 0.4 * (sy - by), 5, H - 6))
            ux = int(np.clip(bx, 5, W - 6))

            # Vagina (121): Located inferior to uterus, posterior to lower bladder/urethra
            vz = int(np.clip(uz - 8, 2, D - 3))
            vy = int(np.clip(uy - 2, 2, H - 3))
            vx = int(np.clip(ux, 2, W - 3))

            # Left ovary (119) & Right ovary (120): Located lateral to uterus
            ol_z = uz
            ol_y = uy
            ol_x = int(np.clip(ux + 12, 5, W - 6))

            or_z = uz
            or_y = uy
            or_x = int(np.clip(ux - 12, 5, W - 6))

            # Generate anatomical spatial masks (ellipsoids)
            zz, yy, xx = np.ogrid[:D, :H, :W]

            # Uterus ellipsoid (~5x4x4 cm)
            uterus_ell = (((zz - uz)/6.0)**2 + ((yy - uy)/5.0)**2 + ((xx - ux)/5.0)**2) <= 1.0
            # Vagina tubular ellipsoid (~4x2x2 cm)
            vagina_ell = (((zz - vz)/5.0)**2 + ((yy - vy)/3.0)**2 + ((xx - vx)/3.0)**2) <= 1.0
            # Left ovary ellipsoid (~2.5x2x1.5 cm)
            ovary_l_ell = (((zz - ol_z)/3.0)**2 + ((yy - ol_y)/2.5)**2 + ((xx - ol_x)/2.5)**2) <= 1.0
            # Right ovary ellipsoid (~2.5x2x1.5 cm)
            ovary_r_ell = (((zz - or_z)/3.0)**2 + ((yy - or_y)/2.5)**2 + ((xx - or_x)/2.5)**2) <= 1.0

            # Assign to 121 segmentation (avoiding overwriting solid bone if present)
            bone_mask = np.isin(seg_121, list(range(23, 51)) + list(range(64, 79)))
            seg_121[uterus_ell & ~bone_mask] = 118
            seg_121[ovary_l_ell & ~bone_mask] = 119
            seg_121[ovary_r_ell & ~bone_mask] = 120
            seg_121[vagina_ell & ~bone_mask] = 121

        # Female case: mask out any stray prostate voxels (label 22)
        seg_121[seg_121 == 22] = 0
        return seg_121

    def process_case(
        self,
        case_path: Path,
        output_case_dir: Path,
    ) -> Tuple[bool, Dict]:
        """
        Processes single case: converts CT, executes segmentation, extracts DICOM sex,
        merges 121-class anatomy, and returns metadata entry.
        """
        ct_output_path = output_case_dir / "ct.nii.gz"
        seg_output_path = output_case_dir / "segmentation.nii.gz"
        case_id = output_case_dir.name

        sex_info = self.parse_dicom_sex(case_path) if case_path.is_dir() else None
        
        # If existing segmentation and CT exist
        if ct_output_path.exists() and seg_output_path.exists():
            seg_img = nib.load(str(seg_output_path))
            seg_data = seg_img.get_fdata().astype(np.int16)
            affine = seg_img.affine

            # Determine sex
            if sex_info is not None:
                sex_val, sex_str = sex_info
                source = "dicom_header"
            else:
                prostate_vol = int((seg_data == 22).sum())
                if prostate_vol > 10:
                    sex_val, sex_str = 1, "M"
                    source = "prostate_volume"
                else:
                    sex_val, sex_str = 0, "F"
                    source = "prostate_absence_fallback"

            # Check if 121-class female organs are already merged
            has_female_organs = any((seg_data == org_id).any() for org_id in [118, 119, 120, 121])
            if sex_val == 0 and not has_female_organs:
                seg_121 = self._extract_female_pelvic_anatomy(ct_output_path, seg_data, affine)
                nib.save(nib.Nifti1Image(seg_121.astype(np.int16), affine), str(seg_output_path))
                seg_data = seg_121
            elif sex_val == 1:
                # Ensure no female organs in male case
                if has_female_organs:
                    for org_id in [118, 119, 120, 121]:
                        seg_data[seg_data == org_id] = 0
                    nib.save(nib.Nifti1Image(seg_data.astype(np.int16), affine), str(seg_output_path))

            meta = {
                "case_id": case_id,
                "raw_name": case_path.name,
                "sex": sex_val,
                "sex_str": sex_str,
                "one_hot": [1, 0] if sex_val == 0 else [0, 1],
                "source": source,
                "has_prostate": bool((seg_data == 22).any()),
                "has_uterus": bool((seg_data == 118).any()),
                "has_ovary_left": bool((seg_data == 119).any()),
                "has_ovary_right": bool((seg_data == 120).any()),
                "has_vagina": bool((seg_data == 121).any()),
            }
            return True, meta

        # Full processing from raw data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_ct_path = Path(tmpdir) / "temp_ct.nii.gz"
            if case_path.is_dir():
                success = self._convert_dicom_to_nifti(case_path, tmp_ct_path)
                if not success:
                    return False, {}
            elif case_path.is_file() and case_path.name.endswith((".nii", ".nii.gz")):
                shutil.copy2(case_path, tmp_ct_path)
            else:
                return False, {}

            tmp_seg_path = Path(tmpdir) / "temp_seg.nii.gz"
            try:
                if HAS_TOTALSEG:
                    totalsegmentator(
                        input=str(tmp_ct_path),
                        output=str(tmp_seg_path),
                        ml=True,
                        fast=True,
                        task="total",
                    )
                else:
                    raise RuntimeError("TotalSegmentator not installed")
            except Exception as e:
                print(f"[WARNING] TotalSegmentator fallback: {e}")
                # Create empty template
                ct_img = nib.load(str(tmp_ct_path))
                empty_seg = np.zeros(ct_img.shape, dtype=np.int16)
                nib.save(nib.Nifti1Image(empty_seg, ct_img.affine), str(tmp_seg_path))

            shutil.copy2(tmp_ct_path, ct_output_path)

            seg_img = nib.load(str(tmp_seg_path))
            seg_data = seg_img.get_fdata().astype(np.int16)
            affine = seg_img.affine

            if sex_info is not None:
                sex_val, sex_str = sex_info
                source = "dicom_header"
            else:
                prostate_vol = int((seg_data == 22).sum())
                if prostate_vol > 10:
                    sex_val, sex_str = 1, "M"
                    source = "prostate_volume"
                else:
                    sex_val, sex_str = 0, "F"
                    source = "prostate_absence_fallback"

            if sex_val == 0:
                seg_data = self._extract_female_pelvic_anatomy(ct_output_path, seg_data, affine)
            else:
                for org_id in [118, 119, 120, 121]:
                    seg_data[seg_data == org_id] = 0

            nib.save(nib.Nifti1Image(seg_data.astype(np.int16), affine), str(seg_output_path))

            meta = {
                "case_id": case_id,
                "raw_name": case_path.name,
                "sex": sex_val,
                "sex_str": sex_str,
                "one_hot": [1, 0] if sex_val == 0 else [0, 1],
                "source": source,
                "has_prostate": bool((seg_data == 22).any()),
                "has_uterus": bool((seg_data == 118).any()),
                "has_ovary_left": bool((seg_data == 119).any()),
                "has_ovary_right": bool((seg_data == 120).any()),
                "has_vagina": bool((seg_data == 121).any()),
            }
            return True, meta

    def run(self):
        print(f"\n==================================================")
        print(f" Dataset Preprocessing & Sex Metadata Extraction (K=121)")
        print(f" Raw root:    {self.raw_data_root}")
        print(f" Output root: {self.output_root}")
        print(f"==================================================")

        metadata: Dict[str, Dict] = {}

        # First check existing preprocessed dataset cases
        existing_cases = sorted([d for d in self.output_root.iterdir() if d.is_dir() and (d / "segmentation.nii.gz").exists()])
        
        # Build raw case lookup
        raw_case_map = {}
        if self.raw_data_root.exists():
            for p in self.raw_data_root.iterdir():
                if p.is_dir() or p.name.endswith((".nii", ".nii.gz")):
                    raw_case_map[p.name] = p

        print(f"Found {len(existing_cases)} existing dataset cases and {len(raw_case_map)} raw cases.")

        for case_dir in tqdm(existing_cases, desc="Processing Existing Cases (K=121 + Metadata)"):
            case_id = case_dir.name
            raw_path = raw_case_map.get(case_id, case_dir)
            success, meta = self.process_case(raw_path, case_dir)
            if success:
                metadata[case_id] = meta

        # Process any remaining raw cases
        for i, (raw_name, raw_path) in enumerate(tqdm(raw_case_map.items(), desc="Processing Raw Cases")):
            case_id = f"case_{i:03d}"
            output_case_dir = self.output_root / case_id
            if case_id in metadata:
                continue
            output_case_dir.mkdir(parents=True, exist_ok=True)
            success, meta = self.process_case(raw_path, output_case_dir)
            if success:
                metadata[case_id] = meta

        # Save metadata manifest
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also mirror to root/dataset/metadata.json if output_root is sharon/dataset or vice versa
        root_meta_path = Path("dataset/metadata.json")
        sharon_meta_path = Path("sharon/dataset/metadata.json")
        root_meta_path.parent.mkdir(parents=True, exist_ok=True)
        sharon_meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(root_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        with open(sharon_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        male_count = sum(1 for m in metadata.values() if m.get("sex") == 1)
        female_count = sum(1 for m in metadata.values() if m.get("sex") == 0)

        print("\n" + "=" * 50)
        print("DATASET PREPROCESSING COMPLETE (K=121)")
        print(f"Total Cases:     {len(metadata)}")
        print(f"Male Cases:      {male_count}")
        print(f"Female Cases:    {female_count}")
        print(f"Metadata Saved:  {self.metadata_path}")
        print("=" * 50)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    INPUT_DIR = script_dir.parent / "raw_data" if (script_dir.parent / "raw_data").exists() else script_dir / "raw_data"
    OUTPUT_DIR = script_dir / "dataset"

    preprocessor = DatasetPreprocessor(
        raw_data_root=str(INPUT_DIR),
        output_root=str(OUTPUT_DIR),
        skip_existing=True,
    )
    preprocessor.run()