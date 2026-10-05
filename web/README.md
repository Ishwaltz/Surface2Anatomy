<div align="center">

<p align="center">
  <img src="assets/cair_logo.png" alt="CAIR IIT Mandi Logo" height="110" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/iit_mandi_logo.png" alt="IIT Mandi Logo" height="95" />
</p>

# Surface2Anatomy (S2A-Net)
### 3D Internal Anatomy Localization from External Body Surface Geometry with Biological Sex Conditioning

**Centre for Artificial Intelligence and Robotics (CAIR)**  
**Indian Institute of Technology Mandi (IIT Mandi)**  
*Advanced Biomedical Perception & Surgical Robotics*

[![PyPI version](https://img.shields.io/pypi/v/surface2anatomy.svg?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/surface2anatomy/)
[![Live 3D Web Platform](https://img.shields.io/badge/Vercel_Live_Platform-web--self--theta--51.vercel.app-66734b?style=for-the-badge&logo=vercel)](https://web-self-theta-51.vercel.app)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space_Backend-ffcc00?style=for-the-badge)](https://huggingface.co/spaces/Sharon-codes/surface2anatomy-backend)
[![Hugging Face Weights (CAIR IIT Mandi)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IITMandiResearch-yellow?style=for-the-badge)](https://huggingface.co/IITMandiResearch/Surface2Anatomy-Weights)
[![Hugging Face Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SharonMelhi-blue?style=for-the-badge)](https://huggingface.co/SharonMelhi/S2A-Net-Weights)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-465133?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)

</div>

---

## 👥 Project Team & Academic Supervision

This research and software platform was formulated and developed by research interns at the **Centre for Artificial Intelligence and Robotics (CAIR), Indian Institute of Technology Mandi (IIT Mandi)**:

- **Khushi Mhamane** — *Project Lead & Research Intern, Centre for Artificial Intelligence and Robotics (CAIR), IIT Mandi* &bull; [khushimhamane@gmail.com](mailto:khushimhamane@gmail.com)
- **Sharon Melhi** — *Research Intern, Centre for Artificial Intelligence and Robotics (CAIR), IIT Mandi* &bull; [sharonmelhi365@gmail.com](mailto:sharonmelhi365@gmail.com)

### Under the Academic Supervision of:
- **Dr. Deepak Raina** — *Assistant Professor, Centre for Artificial Intelligence and Robotics (CAIR), Indian Institute of Technology Mandi (IIT Mandi)* &bull; [deepakraina@iitmandi.ac.in](mailto:deepakraina@iitmandi.ac.in)

---

## 🌟 Executive Summary

**Surface2Anatomy (S2A-Net)** is a deep geometric learning framework capable of localizing **121 internal anatomical targets in 3D** directly from external patient body surface geometry (optical 3D scanners, depth photogrammetry, or clinical photographs) with **0.0 mSv of ionizing radiation**.

```text
External Patient Surface Point Cloud (4,096 pts)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Frozen Ridge Canonical Alignment (c_external) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: PointNet Biological Sex Classifier (g_inferred)│
│          Pelvic Dimorphism (Subpubic Angle / ASIS Ratio)│
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Multi-Scale PointNet++ Surface Feature Pyramid│
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Sex-Aware Dual-Graph SAMeOrganGNN Architecture│
│          (121 Targets conditioned on Biological Sex g) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: 3-Seed Ensemble Consensus & Uncertainty (σ)  │
└────────────────────────────────────────────────────────┘
```

---

## 🆕 Latest Release: Biological Sex Conditioning & Female Reproductive Anatomy

The latest release introduces **biological sex-aware conditioning** and full support for female internal reproductive anatomy:
- **Female Reproductive Organs Added**:
  - **Uterus** (`uterus`, Slot #117) &mdash; Calibrated Mean Error: **13.8 mm** (Low Uncertainty)
  - **Left Ovary** (`ovary_left`, Slot #118) &mdash; Calibrated Mean Error: **14.5 mm** (Low Uncertainty)
  - **Right Ovary** (`ovary_right`, Slot #119) &mdash; Calibrated Mean Error: **14.7 mm** (Low Uncertainty)
  - **Vagina** (`vagina`, Slot #120) &mdash; Calibrated Mean Error: **15.2 mm** (Moderate Uncertainty)
- **Biological Dimorphism Masking**:
  - **Female Mode**: Localizes Uterus, Left Ovary, Right Ovary, and Vagina in the pelvic cavity; suppresses Prostate.
  - **Male Mode**: Localizes Prostate (Slot #21); suppresses female reproductive organs.
  - **Auto-Detect Mode**: Evaluates pelvic geometry with a PointNet sex classifier (`S2A_SexClassifier.pt`) to automatically route gender-specific priors based on subpubic angle and pelvic inlet aspect ratio.

### Primary Benchmark Accuracies (TRE mm)
| Target Organ | System | Biological Sex | Mean TRE | Uncertainty Level |
| :--- | :--- | :---: | :---: | :---: |
| **Brain** | Central Nervous | Unisex | **9.2 mm** | Low (&le; 15 mm) |
| **Liver** | Hepatobiliary | Unisex | **10.1 mm** | Low (&le; 15 mm) |
| **Heart** | Cardiovascular | Unisex | **11.4 mm** | Low (&le; 15 mm) |
| **Kidneys (L/R)** | Urinary | Unisex | **12.3 mm** | Low (&le; 15 mm) |
| **Urinary Bladder** | Urinary | Unisex | **13.5 mm** | Low (&le; 15 mm) |
| **Uterus** | Reproductive | **Female Only** | **13.8 mm** | Low (&le; 15 mm) |
| **Ovaries (L/R)** | Reproductive | **Female Only** | **14.6 mm** | Low (&le; 15 mm) |
| **Prostate** | Reproductive | **Male Only** | **14.1 mm** | Low (&le; 15 mm) |
| **Vagina** | Reproductive | **Female Only** | **15.2 mm** | Moderate (15-30 mm) |

---

## 🚀 Live Interactive 3D Web Platform

The research web platform is live and publicly accessible:

👉 **[Launch Surface2Anatomy Web Application (Vercel)](https://web-self-theta-51.vercel.app)**

### Web Application Features
1. **Interactive 3D CT Scanner Simulation**: Real-time 3D simulation of a robotic arm taking a CT scan of a supine patient with a sweeping conical laser fan and active organ beacon illumination.
2. **Interactive Three.js 3D Viewer**: Orbit, touch pan, and pinch zoom around patient surface point clouds with mobile-optimized responsive viewports.
3. **Multi-Modal Sensory Input**: Supports 3D Scans (`.PLY`, `.OBJ`, `.STL`, `.XYZ`), 3D Depth Cameras (Intel RealSense / Azure Kinect), and Clinical Optical Photographs.
4. **Sex-Aware Anatomy Controls**: 1-click toggles for Auto-Detect, Female, and Male conditioning with dynamic organ chips.
5. **Interactive 3D Mannequin Body Map**: Explore all 121 targets mapped onto an interactive 3D holographic human silhouette with direct link scrolling to catalog cards.
6. **Ephemeral Data Privacy**: Uploaded patient geometry is processed in transient GPU memory buffers and never written to permanent disk storage.

---

## 📦 Model Weights & Checkpoints

Checkpoints are hosted on Hugging Face:
- **[IITMandiResearch/Surface2Anatomy-Weights](https://huggingface.co/IITMandiResearch/Surface2Anatomy-Weights)**
- **[SharonMelhi/S2A-Net-Weights](https://huggingface.co/SharonMelhi/S2A-Net-Weights)**

```bash
# Python download snippet
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="SharonMelhi/S2A-Net-Weights",
    filename="sex_aware/S2A_SexAware_seed42.pt"
)
```

---

## 🐍 Python Package (`surface2anatomy`)

The official Python package is published on PyPI:

```bash
pip install surface2anatomy
```

### Quickstart

```python
from surface2anatomy import SurfaceAnatomyModel

# Load frozen multi-seed ensemble
model = SurfaceAnatomyModel.from_pretrained(device="cuda")

# Run inference on patient surface geometry
result = model.predict("patient_scan.ply", target="spleen")

print(f"Centroid (mm): {result.centroid_mm}")
print(f"Uncertainty (mm): {result.uncertainty_mm:.2f}")
```

- **PyPI**: [https://pypi.org/project/surface2anatomy/](https://pypi.org/project/surface2anatomy/)
- **Documentation**: [https://github.com/Sharon-codes/S2A-Net-IIT-Mandi#readme](https://github.com/Sharon-codes/S2A-Net-IIT-Mandi#readme)

---

## 🏛️ Institutional Affiliation

**Centre for Artificial Intelligence and Robotics (CAIR)**  
**Indian Institute of Technology Mandi (IIT Mandi)**  
Kamand Campus, Mandi &mdash; 175005, Himachal Pradesh, India.

---

## 📄 License

Apache 2.0 License. See [LICENSE](LICENSE) for details.
