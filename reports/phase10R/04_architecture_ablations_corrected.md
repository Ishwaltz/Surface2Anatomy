# Phase 10R: Corrected Architecture Ablation Study

All trained ablations matched to canonical 65-epoch protocol with AdamW, CosineAnnealingLR.

| Ablation Code | Architectural Modification | Type | Macro MRE (mm) | Delta vs Full (mm) |
|---|---|---|---|---|
| **D1 (Full)** | **Proposed MultiScale Target-Query Model** | **TRAINED** | **25.09** | **Reference** |
| D2_SingleScale_SA3 | D2_SingleScale_SA3 | TRAINED ABLATION | 25.70 | +0.61 mm |
| D3_NoAtlasPrior | D3_NoAtlasPrior | TRAINED ABLATION | 23.33 | -1.76 mm |
| D4_GlobalTokenOnly | D4_GlobalTokenOnly | TRAINED ABLATION | 43.38 | +18.29 mm |
| D5_SelfAttnOnly | D5_SelfAttnOnly | TRAINED ABLATION | 46.09 | +21.01 mm |
| D6_SelfAndCrossAttn | D6_SelfAndCrossAttn | TRAINED ABLATION | 25.45 | +0.36 mm |
| D7_Layer2 | D7_Layer2 | TRAINED ABLATION | 24.77 | -0.32 mm |
| D7_Layer6 | D7_Layer6 | TRAINED ABLATION | 31.94 | +6.85 mm |
| D9_QueryPermutation_Diagnostic | D9_QueryPermutation_Diagnostic | INFERENCE DIAGNOSTIC | 46.63 | +21.54 mm |
| D10_TokenShuffle_Diagnostic | D10_TokenShuffle_Diagnostic | INFERENCE DIAGNOSTIC | 96.59 | +71.51 mm |
