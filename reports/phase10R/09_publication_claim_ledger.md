# Phase 10R: Publication Claim Ledger & Scientific Governance

This ledger establishes explicit scientific boundaries for all text, discussions, abstracts, and claims submitted for peer review.

### A. Patient-Specific Surface Geometry Matters
- **Supporting Result:** Patient token shuffle diagnostic (D10) increases MRE from 24.39 mm to 56.40 mm (+32.01 mm).
- **Statistical Evidence:** Paired Wilcoxon test p < 0.0002; bootstrap 95% CI on delta: [+28.4, +35.8] mm.
- **Methodological Limitation:** Surface geometry informs position, but depth coordinates exhibit higher variance than superficial landmarks.
- **Permitted Scientific Phrasing:** *"'The model learns patient-specific geometry rather than relying solely on mean anatomy.'"*
- **Prohibited / Unsafe Claim:** ~~"'The model perfectly memorizes all patient variations without error.'"~~

### B. Target Identity Representation Matters
- **Supporting Result:** Target-query slot permutation diagnostic (D9) increases MRE from 24.39 mm to 61.20 mm (+36.81 mm).
- **Statistical Evidence:** Paired Wilcoxon test p < 0.0002; bootstrap 95% CI on delta: [+32.1, +41.5] mm.
- **Methodological Limitation:** Target embeddings provide anatomical identity; cross-attention alone does not disambiguate targets.
- **Permitted Scientific Phrasing:** *"'Learned target embeddings and atlas priors successfully encode organ-specific spatial identities.'"*
- **Prohibited / Unsafe Claim:** ~~"'Attention heatmaps alone uniquely identify organ boundaries.'"~~

### C. Multi-Scale Spatial Memory Matters
- **Supporting Result:** Ablating multi-scale memory to coarse SA3 only (D2) degrades MRE from 24.39 mm to 27.12 mm (+2.73 mm).
- **Statistical Evidence:** Paired bootstrap delta 95% CI: [+1.85, +3.62] mm, p < 0.0002.
- **Methodological Limitation:** Coarse tokens capture general posture, but mid-level tokens (L2) are needed for local surface contours.
- **Permitted Scientific Phrasing:** *"'Multi-scale surface feature extraction provides significant localization improvements over single-scale coarse features.'"*
- **Prohibited / Unsafe Claim:** ~~"'Fine-scale millimeter-level skin ripples dictate internal organ placement.'"~~

### D. Cross-Attention to Patient Surface Tokens Matters
- **Supporting Result:** Removing surface cross-attention (D5 Self-Attn Only) degrades MRE to 31.40 mm (+7.01 mm).
- **Statistical Evidence:** Paired bootstrap delta 95% CI: [+5.42, +8.65] mm, p < 0.0002.
- **Methodological Limitation:** Cross-attention is diffuse rather than focal, aggregating distributed spatial context.
- **Permitted Scientific Phrasing:** *"'Cross-attention to patient surface tokens is essential for conditioning internal predictions on external geometry.'"*
- **Prohibited / Unsafe Claim:** ~~"'Attention is highly focal and pinpoints organ projections on the skin.'"~~

### E. Pooled Multi-Domain Training Benefits Both Sources
- **Supporting Result:** Matched 65-epoch domain transfer shows pooled model achieves 19.63 mm on V2 (vs 20.45 mm V2-only, delta=-0.82 mm) and 27.49 mm on TS (vs 28.10 mm TS-only, delta=-0.61 mm).
- **Statistical Evidence:** Bootstrap CIs confirm positive transfer: Delta_V2 [-1.35, -0.28] mm, Delta_TS [-1.12, -0.15] mm.
- **Methodological Limitation:** Performance remains higher on V2 (focused torso scans) than TotalSegmentator (heterogeneous whole-body scans).
- **Permitted Scientific Phrasing:** *"'Joint multi-source training provides positive cross-domain transfer without degradation.'"*
- **Prohibited / Unsafe Claim:** ~~"'Domain gap is completely eliminated and all distributions are identical.'"~~

### F. Ensemble Uncertainty Ranks Reliable Predictions
- **Supporting Result:** Ensemble predictive disagreement correlates with localization error (Spearman rho = 0.30, p < 1e-12). Retaining top 70% coverage reduces MRE by 2.15 mm.
- **Statistical Evidence:** Statistically significant rank correlation; risk-coverage curve monotonically decreases.
- **Methodological Limitation:** Correlation is modest (rho ~ 0.30); disagreement is not a fully calibrated predictive variance.
- **Permitted Scientific Phrasing:** *"'Model disagreement provides a useful ranking metric for selective prediction and quality control.'"*
- **Prohibited / Unsafe Claim:** ~~"'The model provides perfectly calibrated clinical error margins.'"~~

### G. Sensor Robustness under Depth Perturbations is Promising
- **Supporting Result:** Under depth noise up to sigma=5 mm, MRE remains resilient (25.80 mm vs 24.39 mm reference).
- **Statistical Evidence:** Monotonic degradation across controlled synthetic noise levels.
- **Methodological Limitation:** Evaluated on synthetic CT-derived meshes, not raw physical optical time-of-flight depth streams.
- **Permitted Scientific Phrasing:** *"'Simulated depth noise experiments demonstrate robust resilience to high-frequency sensor noise.'"*
- **Prohibited / Unsafe Claim:** ~~"'Real-world RGB-D clinical validation is complete and ready for operating room deployment.'"~~

### H. Partial Surface Visibility Remains an Important Limitation
- **Supporting Result:** Restricting surface visibility to a single frontal camera increases MRE to 33.50 mm (+9.11 mm degradation).
- **Statistical Evidence:** 2-camera oblique improves to 28.10 mm; 3-camera SGRT recovers to 25.40 mm.
- **Methodological Limitation:** Single-view anterior depth misses lateral and posterior skeletal landmarks.
- **Permitted Scientific Phrasing:** *"'Multi-camera arrangements (e.g. SGRT ceiling pods) are critical to mitigating partial-view occlusion.'"*
- **Prohibited / Unsafe Claim:** ~~"'A single smartphone or single-view camera is sufficient for comprehensive full-body internal mapping.'"~~

