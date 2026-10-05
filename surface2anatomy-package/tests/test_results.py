"""Tests SinglePrediction and PredictionResult data containers."""

import json
from surface2anatomy.results import SinglePrediction, PredictionResult, PreprocessingMetadata

def test_results_serialization():
    meta = PreprocessingMetadata(
        original_point_count=5000,
        sampled_point_count=4096,
        units="millimeters",
        body_width_mm=320.0,
        body_depth_mm=210.0,
        body_height_mm=750.0,
        canonical_center_mm=(0.0, 0.0, 0.0),
        preprocessing_latency_ms=12.5
    )
    pred_spleen = SinglePrediction(
        target="spleen",
        target_index=0,
        centroid_mm=(-61.2, 34.8, 105.4),
        seed_predictions_mm={"42": (-60.1, 35.0, 104.9), "43": (-62.4, 34.1, 105.8), "44": (-61.1, 35.3, 105.5)},
        ensemble_disagreement_mm=1.3,
        centroid_input_world_mm=(-61.2, 34.8, 105.4)
    )
    res = PredictionResult(
        predictions={"spleen": pred_spleen},
        preprocessing=meta,
        model_variant="phase16_brain",
        model_latency_ms=250.0,
        total_latency_ms=262.5
    )

    # Dictionary export
    d = res.to_dict()
    assert "predictions" in d
    assert "spleen" in d["predictions"]
    assert d["predictions"]["spleen"]["centroid_mm"] == [-61.2, 34.8, 105.4]

    # JSON export
    js = res.to_json()
    parsed = json.loads(js)
    assert parsed["predictions"]["spleen"]["ensemble_disagreement_mm"] == 1.3

    # DataFrame export
    df = res.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["target"] == "spleen"
    assert df.iloc[0]["X_mm"] == -61.2
