# Domain Adaptation Decision & Protocol Report

> [!IMPORTANT]
> **DECISION TREE GOVERNANCE: ADAPTATION NOT NEEDED (TERMINATED AT STAGE 10)**
> In strict accordance with Section 24 ("DECISION TREE"):
> *"IF new external frame + V3-only FOV training reaches AMOS 25–30 mm: THEN STOP. Do not add AMOS-supervised training. This is the strongest scientific result."*

---

## 1. Decision Tree Evaluation

1. **Criterion 1 (Oracle Threshold):**
   - Translation Oracle achieved **17.24 mm** ($\le 30\text{ mm}$).
   - Mandate: Prioritize external origin/canonicalization; do NOT redesign anatomy network.
2. **Criterion 2 (Deployable External Frame):**
   - System 1 (Frozen Phase-10R + External Frame) achieved **27.25 mm** [95% CI: 26.07, 28.46 mm].
   - System 3 (Phase12-FOV + External Frame) achieved **23.98 mm** [95% CI: 22.96, 25.04 mm].
3. **Threshold Condition:**
   - Both System 1 ($27.25\text{ mm}$) and System 3 ($23.98\text{ mm}$) strictly achieved $\le 30.0\text{ mm}$ zero-shot external generalization without touching a single AMOS training label.
4. **Action:**
   - **STOP.**
   - Unsupervised domain adaptation is **NOT NEEDED**.
   - Supervised AMOS fine-tuning is **EXPLICITLY REJECTED** to preserve true zero-shot external generalizability.

---

## 2. Scientific Significance

Preserving the zero-shot status of the evaluation is the paramount scientific objective of Phase 12. Fine-tuning on AMOS organ masks would compromise external validity and convert the experiment into an ordinary in-domain transfer learning study.
Reaching **23.98 mm** solely through external surface geometry engineering on Dataset V3 establishes that:
*"A substantial component of the apparent external-domain degradation was attributable to field-of-view-dependent coordinate canonicalization rather than failure of patient-specific anatomy inference."*
