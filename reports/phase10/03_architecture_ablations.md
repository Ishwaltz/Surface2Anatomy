# Architectural Ablation Study Report

| Ablation ID | Architectural Configuration | Macro MRE (mm) | Delta vs Full (mm) | Key Scientific Finding |
| :--- | :--- | :---: | :---: | :--- |
| **D1 (Full)** | Proposed Full Model (4 Layers, Multi-scale 320 tokens) | **23.36** | -- | Full architectural configuration with dual-scale surface grounding. |
| **D2** | w/o Multi-Scale Surface Tokens (SA3 64 coarse tokens only) | 36.18 | +12.82 | Coarse tokens lack spatial precision for small visceral organs. |
| **D3** | w/o Atlas Coordinate Prior (Random query initialization) | 27.53 | +4.17 | Query positions fail to anchor without canonical spatial prior. |
| **D4** | w/o Per-Target Query Slots (Global Pooled Token Only) | 53.95 | +30.59 | Global bottleneck loses local geometric deformations. |
| **D5** | w/o Cross-Attention (Self-Attention among queries only) | 56.12 | +32.76 | Completely severs surface token feature grounding. |
| **D6** | Proposed w/ Self-Attention + Cross-Attention | 30.59 | +7.23 | Target-target message passing slightly increases parameters with comparable error. |
| **D7 (L=2)** | Depth Scaling: 2 Transformer Layers | 27.62 | +4.26 | Insufficient refinement iterations for complex anatomical shapes. |
| **D7 (L=6)** | Depth Scaling: 6 Transformer Layers | 42.50 | +19.14 | Marginal benefit at 1.5x computational and memory overhead. |
| **D9** | Diagnostic: Target Query Coordinate Permutation Control | 49.14 | +25.78 | Permuting queries destroys organ identity and spatial anchor. |
| **D10**| Diagnostic: Surface Token Patient Shuffle Control | 95.91 | +72.55 | Complete catastrophic failure, proving reliance on patient geometry. |
