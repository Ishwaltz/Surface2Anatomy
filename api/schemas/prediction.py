from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    target: Optional[str] = Field(None, description="Requested internal target name, e.g. 'spleen' or 'liver'")
    targets: Optional[List[str]] = Field(None, description="Optional list of targets for multi-target query")
    points: Optional[List[List[float]]] = Field(None, description="Optional raw or preprocessed Nx3 coordinates")

class SeedPredictions(BaseModel):
    seed42: List[float]
    seed43: List[float]
    seed44: List[float]

class SingleTargetResult(BaseModel):
    target: str
    target_index: int
    centroid_canonical_mm: List[float]
    centroid_input_world_mm: List[float]
    seed_predictions_mm: Dict[str, List[float]]
    ensemble_prediction_mm: List[float]
    uncertainty_mm: float
    uncertainty_level: str = "Low"

class PredictionResponse(BaseModel):
    target: str
    target_index: int
    centroid_canonical_mm: List[float]
    centroid_input_world_mm: List[float]
    seed_predictions_mm: Dict[str, List[float]]
    ensemble_prediction_mm: List[float]
    uncertainty_mm: float
    uncertainty_level: str = "Low"
    model_latency_ms: float
    preprocessing_latency_ms: float
    total_latency_ms: float
    disclaimer: str = "Research prototype — not for clinical diagnosis or procedural guidance."

class MultiPredictionResponse(BaseModel):
    results: Dict[str, SingleTargetResult]
    c_external_mm: List[float]
    point_count: int
    detected_unit: str
    model_latency_ms: float
    preprocessing_latency_ms: float
    total_latency_ms: float
    disclaimer: str = "Research prototype — not for clinical diagnosis or procedural guidance."

class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: int
    targets_available: int
    version: str = "Surface2Anatomy-1.0.0"
