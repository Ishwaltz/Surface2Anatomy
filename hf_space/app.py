try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is None:
                return lambda f: f
            return fn
    spaces = _MockSpaces()

import os
os.environ["GRADIO_SSR_MODE"] = "False"
import sys
import time
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add current directory to path
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from schemas.prediction import (
    PredictionRequest, PredictionResponse, MultiPredictionResponse,
    SingleTargetResult, HealthResponse
)
from inference.model_loader import get_models
from inference.preprocessing import preprocess_surface_bytes
from inference.predict import predict_single_or_multi
from utils.target_catalog import (
    CANONICAL_TARGET_NAMES, TARGET_CATEGORIES, SYNONYMS, resolve_target
)

@spaces.GPU
def run_model_inference(pts_norm, c_external, targets, model_variant="phase16_brain", sex="auto"):
    return predict_single_or_multi(
        pts_norm=pts_norm,
        c_external=c_external,
        targets=targets,
        model_variant=model_variant,
        sex=sex
    )

def create_gradio_demo():
    import gradio as gr

    @spaces.GPU
    def predict_ui(file_obj, selected_targets, model_variant, patient_sex):
        if file_obj is None:
            return "Please upload a 3D surface scan (.ply, .obj, .stl, .pcd, .xyz, .npy)."
        if not selected_targets:
            return "Please select at least one anatomical target landmark."

        try:
            with open(file_obj.name, "rb") as f:
                content = f.read()
            prep = preprocess_surface_bytes(Path(file_obj.name).name, content)
            pred_out = run_model_inference(
                pts_norm=prep["pts_norm"],
                c_external=prep["c_external"],
                targets=selected_targets,
                model_variant=model_variant,
                sex=patient_sex
            )
            
            output_lines = [
                "### Surface2Anatomy 3D Localization Results",
                f"- **Model**: {model_variant} (3-Seed Ensemble)",
                f"- **Geometry Scale**: {prep['detected_unit']}",
                f"- **Point Count**: {prep['raw_point_count']:,}",
                f"- **Latency**: {pred_out['model_latency_ms'] + prep['preprocessing_latency_ms']:.1f} ms",
                "",
                "| Target | Centroid World (mm) | Canonical (mm) | Ensemble Disagreement | Quality |",
                "|---|---|---|---|---|"
            ]
            for t_name, r in pred_out["results"].items():
                w = r["centroid_input_world_mm"]
                c = r["centroid_canonical_mm"]
                output_lines.append(
                    f"| **{t_name}** | [{w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}] | [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] | ±{r['uncertainty_mm']:.2f} mm | {r['uncertainty_level']} |"
                )
            return "\n".join(output_lines)
        except Exception as e:
            return f"**Inference error:** {str(e)}"

    blocks = gr.Blocks(title="Surface2Anatomy 3D Organ Localization API")
    with blocks:
        gr.Markdown("# 🫀 Surface2Anatomy Live Inference API")
        gr.Markdown(
            "Target-conditioned 3D Internal Anatomy Localization from External Body Surface Geometry.\n\n"
            "Official neural network backend for the **[Surface2Anatomy Web Application](https://web-self-theta-51.vercel.app/)** "
            "and the **[`surface2anatomy` Python package](https://pypi.org/project/surface2anatomy/)**."
        )
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload 3D Surface (.ply, .obj, .stl, .pcd, .xyz, .npy)")
                targets_input = gr.CheckboxGroup(
                    choices=CANONICAL_TARGET_NAMES[:20],
                    value=["liver", "spleen", "kidney_left", "kidney_right", "heart"],
                    label="Anatomical Targets"
                )
                with gr.Row():
                    variant_input = gr.Dropdown(
                        choices=["phase16_brain", "phase10r"],
                        value="phase16_brain",
                        label="Model Ensemble Variant"
                    )
                    sex_input = gr.Dropdown(
                        choices=["auto", "female", "male"],
                        value="auto",
                        label="Biological Sex Conditioning"
                    )
                btn = gr.Button("⚡ Predict 3D Organ Locations", variant="primary")
            with gr.Column():
                output_md = gr.Markdown("Prediction results will appear here after inference.")

        btn.click(
            fn=predict_ui,
            inputs=[file_input, targets_input, variant_input, sex_input],
            outputs=[output_md]
        )

        gr.Markdown("---")
        gr.Markdown(
            "### Developer & Clinical Integration\n"
            "- **REST API Docs**: [/docs](/docs) | [/health](/health) | [/targets](/targets)\n"
            "- **Python Package**: `pip install surface2anatomy`\n"
            "- **Web Platform**: [https://web-self-theta-51.vercel.app/](https://web-self-theta-51.vercel.app/)\n"
            "- **GitHub Repository**: [Sharon-codes/S2A-Net-IIT-Mandi](https://github.com/Sharon-codes/S2A-Net-IIT-Mandi)"
        )
    return blocks

demo = create_gradio_demo()
app = demo.app

# Enable CORS for Next.js frontend (local & Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting Surface2Anatomy HF Space API on ZeroGPU A10G...")

@app.get("/api")
@app.get("/api/info")
async def root():
    return {
        "service": "Surface2Anatomy 3D Organ Location Prediction API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "targets": "/targets"
    }

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device="cuda (ZeroGPU A10G)",
        models_loaded=3,
        targets_available=len(CANONICAL_TARGET_NAMES),
        version="Surface2Anatomy-1.0.0"
    )

@app.get("/targets")
@app.get("/api/targets")
async def get_targets():
    """
    Returns complete catalog of supported targets grouped by anatomical category.
    """
    return {
        "categories": TARGET_CATEGORIES,
        "all_targets": CANONICAL_TARGET_NAMES,
        "synonyms": SYNONYMS
    }

@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    surface_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    target: str = Form("spleen"),
    model_variant: str = Form("phase16_brain"),
    sex: str = Form("auto")
):
    """
    Locates a single requested internal anatomical target from uploaded 3D surface scan.
    """
    t_start = time.time()
    
    resolved = resolve_target(target)
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target '{target}'. Check /targets for supported anatomical structures."
        )
    canon_name, target_idx = resolved
    
    input_file = surface_file or file
    if input_file is None:
        raise HTTPException(status_code=400, detail="Missing required 3D surface file (.PLY, .OBJ, .STL, .PCD, .XYZ, .NPY).")
        
    try:
        content = await input_file.read()
        prep = preprocess_surface_bytes(input_file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    try:
        pred_out = run_model_inference(
            pts_norm=prep["pts_norm"],
            c_external=prep["c_external"],
            targets=[canon_name],
            model_variant=model_variant,
            sex=sex
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    res = pred_out["results"].get(canon_name)
    if not res:
        raise HTTPException(status_code=500, detail="Prediction failed to generate target coordinates.")
        
    total_ms = round((time.time() - t_start) * 1000.0, 2)
    
    return PredictionResponse(
        target=canon_name,
        target_index=target_idx,
        centroid_canonical_mm=res["centroid_canonical_mm"],
        centroid_input_world_mm=res["centroid_input_world_mm"],
        seed_predictions_mm=res["seed_predictions_mm"],
        ensemble_prediction_mm=res["ensemble_prediction_mm"],
        uncertainty_mm=res["uncertainty_mm"],
        uncertainty_level=res["uncertainty_level"],
        model_latency_ms=pred_out["model_latency_ms"],
        preprocessing_latency_ms=prep["preprocessing_latency_ms"],
        total_latency_ms=total_ms
    )

@app.post("/predict-multiple", response_model=MultiPredictionResponse)
@app.post("/api/predict-multiple", response_model=MultiPredictionResponse)
async def predict_multiple(
    surface_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    targets: str = Form("liver,spleen,kidney_left,kidney_right"),
    model_variant: str = Form("phase16_brain"),
    sex: str = Form("auto")
):
    """
    Locates multiple internal anatomical targets from uploaded 3D surface in a SINGLE model pass.
    """
    t_start = time.time()
    
    input_file = surface_file or file
    if input_file is None:
        raise HTTPException(status_code=400, detail="Missing required 3D surface file (.PLY, .OBJ, .STL, .PCD, .XYZ, .NPY).")
        
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    if not target_list:
        target_list = ["liver", "spleen", "kidney_left", "kidney_right"]
        
    try:
        content = await input_file.read()
        prep = preprocess_surface_bytes(input_file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
    try:
        pred_out = run_model_inference(
            pts_norm=prep["pts_norm"],
            c_external=prep["c_external"],
            targets=target_list,
            model_variant=model_variant,
            sex=sex
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    total_ms = round((time.time() - t_start) * 1000.0, 2)
    
    results_map = {}
    for t_name, r in pred_out["results"].items():
        results_map[t_name] = SingleTargetResult(
            target=r["target"],
            target_index=r["target_index"],
            centroid_canonical_mm=r["centroid_canonical_mm"],
            centroid_input_world_mm=r["centroid_input_world_mm"],
            seed_predictions_mm=r["seed_predictions_mm"],
            ensemble_prediction_mm=r["ensemble_prediction_mm"],
            uncertainty_mm=r["uncertainty_mm"],
            uncertainty_level=r["uncertainty_level"]
        )
        
    return MultiPredictionResponse(
        results=results_map,
        c_external_mm=[round(float(c), 2) for c in prep["c_external"]],
        point_count=prep["raw_point_count"],
        detected_unit=prep["detected_unit"],
        model_latency_ms=pred_out["model_latency_ms"],
        preprocessing_latency_ms=prep["preprocessing_latency_ms"],
        total_latency_ms=total_ms
    )
if __name__ == "__main__":
    demo.queue().launch(_app=app, ssr_mode=False, strict_cors=False)

