"""
Typed Results Objects and Serialization Utilities.
Exposes SinglePrediction and PredictionResult with dictionary, JSON, and DataFrame exports.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

@dataclass
class PreprocessingMetadata:
    """Diagnostic geometric metadata computed during surface validation and normalization."""
    original_point_count: int
    sampled_point_count: int
    units: str
    body_width_mm: float
    body_depth_mm: float
    body_height_mm: float
    canonical_center_mm: Tuple[float, float, float]
    preprocessing_latency_ms: float

@dataclass
class SinglePrediction:
    """Anatomical centroid prediction for a single internal target landmark."""
    target: str
    target_index: int
    centroid_mm: Tuple[float, float, float]
    seed_predictions_mm: Dict[str, Tuple[float, float, float]]
    ensemble_disagreement_mm: float
    centroid_input_world_mm: Optional[Tuple[float, float, float]] = None

    def __getitem__(self, idx: int) -> float:
        """Allows tuple-like indexing: pred[0] for X, pred[1] for Y, pred[2] for Z."""
        return self.centroid_mm[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "target_index": self.target_index,
            "centroid_mm": list(self.centroid_mm),
            "seed_predictions_mm": {k: list(v) for k, v in self.seed_predictions_mm.items()},
            "ensemble_disagreement_mm": self.ensemble_disagreement_mm,
            "centroid_input_world_mm": list(self.centroid_input_world_mm) if self.centroid_input_world_mm else None
        }

@dataclass
class PredictionResult:
    """Ensemble prediction result container for single or multi-target localization queries."""
    predictions: Dict[str, SinglePrediction]
    preprocessing: PreprocessingMetadata
    model_variant: str
    model_latency_ms: float
    total_latency_ms: float
    disclaimer: str = "Research prototype — not for clinical diagnosis or procedural guidance."

    def __getitem__(self, target_name: str) -> SinglePrediction:
        """Enables dictionary-style indexing: result['spleen']."""
        clean = target_name.strip().lower().replace("-", "_").replace(" ", "_")
        if clean in self.predictions:
            return self.predictions[clean]
        # Check canonical match
        for k, v in self.predictions.items():
            if k.lower() == clean:
                return v
        raise KeyError(f"Target '{target_name}' was not requested in this query.")

    def __contains__(self, target_name: str) -> bool:
        return target_name in self.predictions

    def __len__(self) -> int:
        return len(self.predictions)

    def keys(self):
        return self.predictions.keys()

    def values(self):
        return self.predictions.values()

    def items(self):
        return self.predictions.items()

    @property
    def centroid_mm(self) -> Tuple[float, float, float]:
        """Convenience property when querying a single target."""
        if len(self.predictions) == 1:
            return next(iter(self.predictions.values())).centroid_mm
        raise ValueError(
            f"Result contains {len(self.predictions)} targets. "
            "Access target-specific centroids via result['<target_name>'].centroid_mm"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts prediction results and metadata into a nested dictionary."""
        return {
            "predictions": {k: v.to_dict() for k, v in self.predictions.items()},
            "preprocessing": asdict(self.preprocessing),
            "model_variant": self.model_variant,
            "model_latency_ms": self.model_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "disclaimer": self.disclaimer
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes results into formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> pd.DataFrame:
        """Exports prediction results into a structured tabular pandas DataFrame."""
        rows = []
        for p in self.predictions.values():
            row = {
                "target": p.target,
                "target_index": p.target_index,
                "X_mm": p.centroid_mm[0],
                "Y_mm": p.centroid_mm[1],
                "Z_mm": p.centroid_mm[2],
                "disagreement_mm": p.ensemble_disagreement_mm
            }
            for seed, coords in p.seed_predictions_mm.items():
                row[f"seed_{seed}_X_mm"] = coords[0]
                row[f"seed_{seed}_Y_mm"] = coords[1]
                row[f"seed_{seed}_Z_mm"] = coords[2]
            if p.centroid_input_world_mm:
                row["world_X_mm"] = p.centroid_input_world_mm[0]
                row["world_Y_mm"] = p.centroid_input_world_mm[1]
                row["world_Z_mm"] = p.centroid_input_world_mm[2]
            rows.append(row)
        return pd.DataFrame(rows)
