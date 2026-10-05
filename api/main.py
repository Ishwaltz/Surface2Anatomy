import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from api.schemas.prediction import (
    PredictionRequest, PredictionResponse, MultiPredictionResponse,
    SingleTargetResult, HealthResponse
)
from api.inference.model_loader import get_models
from api.inference.preprocessing import preprocess_surface_bytes
from api.inference.predict import predict_single_or_multi
from api.utils.target_catalog import (
    CANONICAL_TARGET_NAMES, TARGET_CATEGORIES, SYNONYMS, resolve_target
)

app = FastAPI(
    title="Surface2Anatomy API",
    description="Target-conditioned 3D Internal Anatomy Localization from External Body Surface Geometry",
    version="1.0.0"
)

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
    """
    Pre-loads models and canonical Ridge alignment into GPU memory on server start.
    """
    print("Starting Surface2Anatomy API...")
    try:
        # Load official frozen Phase 10R ensemble
        get_models("phase10r")
    except Exception as e:
        print(f"Warning during model warmup: {e}")

@app.get("/health", response_model=HealthResponse)
async def health():
    models = get_models("phase10r")
    device_name = str(next(models[0].parameters()).device)
    return HealthResponse(
        status="ok",
        device=device_name,
        models_loaded=len(models),
        targets_available=len(CANONICAL_TARGET_NAMES),
        version="Surface2Anatomy-1.0.0"
    )

@app.get("/targets")
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
async def predict(
    surface_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    target: str = Form("spleen"),
    model_variant: str = Form("phase10r"),
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
        pred_out = predict_single_or_multi(
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
async def predict_multiple(
    surface_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    targets: str = Form("liver,spleen,kidney_left,kidney_right"),
    model_variant: str = Form("phase10r"),
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
        pred_out = predict_single_or_multi(
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
