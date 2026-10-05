# Geometric Preprocessing & Canonical Alignment

1. **Physical Scale Normalization**: Coordinates converted to millimeters.
2. **Point Sampling**: Deterministically samples 4,096 points.
3. **Canonical Alignment**: Evaluates 105 external-only morphology features with the frozen Ridge regressor (`canonical_alignment_v3_ridge.joblib`) to estimate $c_{\text{external}}$.
4. **Centering & Scaling**: $(P - c_{\text{external}}) / 500.0$.
