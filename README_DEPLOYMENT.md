# Surface2Anatomy — Deployment & Production Guide

This guide details how to build, test, and deploy the **Surface2Anatomy** web platform and GPU inference engine.

---

## Architecture Overview

- **Frontend**: Next.js 14 (React 18, TypeScript, Tailwind CSS, Three.js interactive 3D canvas with orbit controls, Web Speech API voice input, and multi-target selection).
- **Backend API**: FastAPI (Python 3.11, PyTorch CUDA singleton, frozen Ridge canonical alignment, 4,096-point sampling, multi-target batch inference, 3-seed RMS uncertainty).
- **Supported Surface Formats**: `.PLY` (ASCII/binary), `.PCD`, `.OBJ`, `.STL`, `.XYZ`, `.NPY`.
- **Target Vocabulary**: 104 benchmark targets (TotalSegmentator / CT-ORG) across 7 anatomical categories.
- **Model Checkpoints**:
  - `phase10r`: Official frozen 3-seed ensemble baseline (Seeds 42, 43, 44).
  - `phase16_brain`: Retrained whole-body brain-aware ensemble (resolves cranial target sign inversion, driving brain error from 177.42 mm down to 38.92 mm).

---

## 1. Local Development Quickstart

### Prerequisites
- Python 3.11 with PyTorch + CUDA
- Node.js >= 18 and npm >= 9

### Step 1: Start the FastAPI GPU Backend
From repository root:
```bash
# Activate virtual environment
source /home/sharon/env_py311/bin/activate

# Launch backend API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the backend is live:
```bash
curl http://localhost:8000/health
# Returns: {"status":"ok","models_loaded":1,"targets_available":104,"device":"cuda"}
```

### Step 2: Start the Next.js Frontend
In a separate terminal:
```bash
cd web
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 2. Running Automated Tests

### Backend Unit & Integration Tests (8 tests)
```bash
/home/sharon/env_py311/bin/python tests/test_api.py
```
Validates:
1. `/health` probe and CUDA status
2. `/targets` catalog resolution (104 targets)
3. Medical synonym normalization
4. Unit auto-detection (m, cm, mm)
5. 4,096-point deterministic downsampling
6. Single-target inference
7. Multi-target batched inference
8. NaN/Inf input rejection

### Scientific Regression Test
```bash
/home/sharon/env_py311/bin/python tests/test_regression.py
```
Asserts that API predictions match offline frozen PyTorch checkpoints within $< 1 \times 10^{-4}$ normalized discrepancy ($< 0.05\text{ mm}$) across all benchmarks.

---

## 3. Production Builds

### Next.js Production Build (Vercel Ready)
```bash
cd web
npm run build
```
Generates statically optimized routes and standalone bundles with zero build errors.

### Vercel Deployment
1. Import the `web/` folder to Vercel.
2. Set Environment Variable:
   ```env
   NEXT_PUBLIC_INFERENCE_API_URL=https://your-backend-domain.com
   ```
3. Deploy!

---

## 4. Docker Deployment (CUDA Backend)

### Build Docker Image
```bash
docker build -t surface2anatomy-api:latest -f deployment/Dockerfile .
```

### Run Container with NVIDIA GPU
```bash
docker run --gpus all -d -p 8000:8000 \
  --name surface2anatomy-api \
  --restart unless-stopped \
  surface2anatomy-api:latest
```

---

## 5. Serverless GPU Deployment with Modal

To deploy the backend serverlessly on an NVIDIA T4/A10G with automatic scaling and zero idle cost:

```bash
pip install modal
modal setup
modal deploy deployment/modal_app.py
```

Modal will output a permanent public HTTPS URL (e.g. `https://<workspace>--surface2anatomy-api-fastapi-app.modal.run`).  
Configure this URL in the frontend's `NEXT_PUBLIC_INFERENCE_API_URL`.

---

## 6. Privacy & Clinical Disclaimer

- **Ephemeral Memory Processing**: Uploaded geometry is parsed directly into transient GPU tensors and immediately freed. No point cloud or patient scan is stored on disk or persistent databases.
- **Universal Disclaimer**: *Research prototype — not for clinical diagnosis or procedural guidance.*
