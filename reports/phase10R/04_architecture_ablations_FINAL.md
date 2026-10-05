# Phase 10R: Matched Architecture Ablations (Final Benchmark)

All trained ablations trained under matched 65-epoch protocol. Paired deltas computed against D1 Full Model.

## 1. Trained Architecture Ablations

| Model Variant | Description | Macro MRE (mm) | $\Delta$ vs Full (mm) | Median (mm) | P90 (mm) | SDR@20 (%) |
|---|---|---|---|---|---|---|
| **D1 Proposed Full Model** | Full multiscale (SA2+SA3) memory, 117 learned queries, atlas PE, 4 layers | **25.09** | +0.00 | 19.47 | 43.61 | 51.7% |
| **D2 Coarse SA3 Only** | SA3 tokens only (64 tokens, no 256 SA2 tokens) | **25.70** | +0.61 | 19.80 | 43.34 | 50.6% |
| **D3 No Atlas Prior** | Target queries initialized without atlas coordinate PE | **23.33** | -1.76 | 18.10 | 41.20 | 56.5% |
| **D4 Global Token Only** | Global pooled PointNet++ vector only (no spatial surface tokens) | **43.38** | +18.29 | 32.05 | 71.63 | 26.2% |
| **D5 Self-Attention Only** | Decoder self-attention only; surface cross-attention removed | **46.09** | +21.01 | 34.92 | 80.01 | 21.5% |
| **D6 Self + Cross Attention** | Full cross-attention plus query self-attention layers | **25.45** | +0.36 | 20.09 | 43.95 | 49.7% |
| **D7a 2 Decoder Layers** | Shallow decoder (2 transformer layers) | **24.77** | -0.32 | 19.19 | 42.89 | 52.7% |
| **D7b 6 Decoder Layers** | Deep decoder (6 transformer layers) | **31.94** | +6.85 | 25.19 | 56.40 | 35.9% |

## 2. Inference Diagnostics

| Diagnostic Experiment | Description | Macro MRE (mm) | $\Delta$ vs Normal (mm) | Interpretation |
|---|---|---|---|---|
| **D9 Target Query Permutation** | Permute target query indices at inference time | **46.63** | +21.54 | Queries are identity-specific; permutation destroys localization. |
| **D10 Patient Token Shuffle** | Shuffle surface memory tokens across different patients at test time | **96.59** | +71.51 | Surface tokens are patient-specific; cross-patient mismatch collapses performance to random chance. |
