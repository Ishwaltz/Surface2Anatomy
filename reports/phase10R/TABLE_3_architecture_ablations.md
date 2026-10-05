# Table 3: Architecture Ablations (Matched 65 Epochs, Seed 42)

| Ablation Code | Architectural Modification | Ablation Type | Macro MRE (mm) | Delta vs Full (mm) | Effect Size / P-value |
|---|---|---|---|---|---|
| **D1 (Full)** | **Proposed Multi-Scale Target-Query Model** | **Reference** | **24.39** | **0.00** | **Reference** |
| D2 | Single-Scale Surface Memory (SA3 coarse only) | Trained Ablation | 27.12 | +2.73 mm | p < 0.0002 |
| D3 | Without Atlas Coordinate Positional Prior | Trained Ablation | 32.84 | +8.45 mm | p < 0.0002 |
| D4 | Global Pooled Token Only (No localized tokens) | Trained Ablation | 29.50 | +5.11 mm | p < 0.0002 |
| D5 | Self-Attention Only (No surface cross-attention) | Trained Ablation | 31.40 | +7.01 mm | p < 0.0002 |
| D6 | Combined Target Self-Attention + Cross-Attention | Trained Variant | 24.55 | +0.16 mm | p = 0.42 (n.s.) |
| D7-2 | 2 Decoder Cross-Attention Layers | Trained Ablation | 25.80 | +1.41 mm | p < 0.001 |
| D7-6 | 6 Decoder Cross-Attention Layers | Trained Ablation | 24.31 | -0.08 mm | p = 0.68 (n.s.) |
| D9 | Target-Query Slot Permutation Diagnostic | Inference Diagnostic | 61.20 | +36.81 mm | p < 0.0002 |
| D10 | Patient Surface-Token Shuffle Diagnostic | Inference Diagnostic | 56.40 | +32.01 mm | p < 0.0002 |
