"use client";

import React, { useState } from "react";
import {
  Activity,
  Layers,
  Sparkles,
  User,
  ShieldCheck,
  Scan,
  Camera,
  Eye,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Cpu,
  Database,
  ArrowRight,
} from "lucide-react";
import { MathView } from "@/components/MathView";

interface StageData {
  id: number;
  title: string;
  shortTitle: string;
  tagline: string;
  category: string;
  formula: string;
  formulaLabel: string;
  variables: { symbol: string; meaning: string }[];
  content: React.ReactNode;
}

export default function MethodPage() {
  const [activeStage, setActiveStage] = useState(0);
  const [dimorphismSex, setDimorphismSex] = useState<"female" | "male">("female");
  const [sensorTab, setSensorTab] = useState<"mesh" | "depth" | "rgb">("depth");

  const STAGES: StageData[] = [
    {
      id: 1,
      title: "Problem Formulation & Continuous Spatial Manifold",
      shortTitle: "Spatial Operator",
      tagline: "Radiation-Free Continuous External-to-Internal Coordinate Mapping",
      category: "Theoretical Foundation",
      formula: String.raw`\mathcal{F}_\theta: (\mathcal{S}, \mathcal{T}, g) \longmapsto \left\{ (\hat{\mathbf{c}}_k, \hat{\mathbf{\Sigma}}_k) \in \mathbb{R}^3 \times \mathbb{S}_{++}^3 \;\Big|\; k = 1, \dots, K \right\}`,
      formulaLabel: "Joint Centroid & Uncertainty Mapping Operator",
      variables: [
        { symbol: String.raw`\mathcal{S}`, meaning: "External patient surface point cloud (N = 4,096 points on ∂Ω_body)" },
        { symbol: String.raw`\mathcal{T}`, meaning: "Set of K = 121 anatomical queries (visceral organs, vascular & skeletal)" },
        { symbol: "g", meaning: "Biological sex prior (Female: g = 0, Male: g = 1)" },
        { symbol: String.raw`\hat{\mathbf{c}}_k`, meaning: "Predicted 3D Euclidean centroid coordinate [X, Y, Z] in mm" },
        { symbol: String.raw`\hat{\mathbf{\Sigma}}_k`, meaning: "Calibrated 3x3 epistemic spatial covariance matrix" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Unlike conventional ionizing CT and MRI modalities that acquire direct volumetric slices, <strong>Surface2Anatomy</strong> solves the inverse boundary value problem: estimating the exact 3D internal metric centroids of internal anatomical organs strictly from the geometric topology of the external body surface.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-900 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                <span>Zero Radiation Dose</span>
              </span>
              <span className="text-xs text-slate-500">
                Safe for continuous intraoperative tracking, pediatrics, and repeated radiation oncology setups.
              </span>
            </div>
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-900 flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-sky-600" />
                <span>Real-Time Inference</span>
              </span>
              <span className="text-xs text-slate-500">
                Sub-50ms forward pass latency enables real-time guidance for robotic ultrasound and laparoscopic arms.
              </span>
            </div>
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl flex flex-col gap-1">
              <span className="font-bold text-slate-900 flex items-center gap-1.5">
                <Database className="w-4 h-4 text-amber-600" />
                <span>121 Anatomical Organs</span>
              </span>
              <span className="text-xs text-slate-500">
                Expanded sex-aware atlas covering cranial, thoracic, abdominal, vascular, and pelvic reproductive anatomy.
              </span>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 2,
      title: "Scanner Invariance & Frozen Ridge Canonical Pre-Alignment",
      shortTitle: "Canonical Alignment",
      tagline: "Eigenvalue Moment Invariants & Closed-Form Anchor Regression",
      category: "Geometric Normalization",
      formula: String.raw`\mathbf{m}(\mathcal{S}) = [ \lambda_1, \lambda_2, \lambda_3, \; \lambda_1 / \lambda_3, \; \mu_{200}, \mu_{020}, \mu_{002}, \dots, \mu_{111} ]^T \in \mathbb{R}^{16}`,
      formulaLabel: "16-Dimensional Rotation-Invariant Surface Moment Vector",
      variables: [
        { symbol: String.raw`\lambda_1, \lambda_2, \lambda_3`, meaning: "Eigenvalues of the 3D surface point covariance matrix" },
        { symbol: String.raw`\lambda_1 / \lambda_3`, meaning: "Global elongation aspect ratio of the patient torso" },
        { symbol: String.raw`\mu_{ijk}`, meaning: "Central spatial moments of cross-sectional body slices" },
        { symbol: String.raw`\mathbf{c}_{\text{external}}`, meaning: "Ridge-predicted metric anchor center [X, Y, Z] in mm" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Patient surface scans captured across varying hospital scanner couches or depth cameras arrive with arbitrary translations and rotations. To prevent neural overfitting to room coordinates, we extract a 16-D rotation-invariant geometric moment vector and pass it through a frozen L2-regularized Ridge estimator:
          </p>
          <div className="bg-[#F8F9F5] text-slate-800 p-4 rounded-xl border border-[#D8DCCF] font-mono text-xs sm:text-sm flex flex-col gap-2 shadow-2xs">
            <span className="text-slate-500 text-[11px] uppercase tracking-wider font-semibold">Canonical Centering &amp; Scaling:</span>
            <MathView math={String.raw`\mathcal{P}_{\text{canonical}} = \frac{\mathcal{S} - \mathbf{c}_{\text{external}}}{500.0} \quad \in [-1, 1]^{N \times 3}`} block={false} />
            <span className="text-[11px] text-primary-dark font-sans mt-1">
              &bull; Guarantees metric equivariance regardless of scanner couch elevation or patient translation.
            </span>
          </div>
        </div>
      ),
    },
    {
      id: 3,
      title: "Biological Sex Geometric Dimorphism Conditioning",
      shortTitle: "Sex Dimorphism",
      tagline: "Pelvic Sub-Cloud PointNet & Learned Reproductive Anatomical Priors",
      category: "Anatomical Conditioning",
      formula: String.raw`P(g = \text{Female} \mid \mathcal{S}_{\text{pelvis}}) = \sigma\left( \mathbf{W}_c \cdot \text{PointNet2}(\mathcal{S}_{\text{pelvis}}) + b_c \right)`,
      formulaLabel: "Pelvic Dimorphism Classification & Prior Routing",
      variables: [
        { symbol: String.raw`\mathcal{S}_{\text{pelvis}}`, meaning: "Sub-cloud bounded by bi-trochanteric pelvic landmarks" },
        { symbol: String.raw`\mathbf{W}_c, b_c`, meaning: "Learned linear projection weights on pelvic PointNet2 features" },
        { symbol: String.raw`\mathbf{e}_{\text{sex}}`, meaning: "Continuous 128-D sex embedding injected into GNN layers" },
        { symbol: String.raw`\mathbf{M}(g)`, meaning: "Biological exclusion mask suppressing discordant organs" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Pelvic skeletal anatomy displays profound sexual dimorphism (e.g. subpubic arch angle, bi-iliac width, pelvic brim aperture). Our pipeline dynamically classifies pelvic geometry or accepts clinician input to activate sex-specific anatomical routing:
          </p>

          {/* Interactive Prior Simulator */}
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/80 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800 text-xs uppercase tracking-wider flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-primary" />
                <span>Interactive Anatomical Mask Simulator:</span>
              </span>
              <div className="flex items-center gap-1 bg-white p-0.5 rounded-lg border border-slate-200 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => setDimorphismSex("female")}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    dimorphismSex === "female"
                      ? "bg-fuchsia-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  Female Prior (g = 0)
                </button>
                <button
                  type="button"
                  onClick={() => setDimorphismSex("male")}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    dimorphismSex === "male"
                      ? "bg-indigo-600 text-white shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  Male Prior (g = 1)
                </button>
              </div>
            </div>

            {dimorphismSex === "female" ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-fuchsia-50 border border-fuchsia-200 text-fuchsia-900 flex flex-col gap-1">
                  <span className="font-bold flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-fuchsia-700" />
                    <span>Active Female Structures (Weight &gt; 0):</span>
                  </span>
                  <span>&bull; Uterus (#117) &bull; Left Ovary (#118)</span>
                  <span>&bull; Right Ovary (#119) &bull; Vagina (#120)</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-100 border border-slate-200 text-slate-500 flex flex-col gap-1">
                  <span className="font-bold">Suppressed Discordant Structures:</span>
                  <span>&bull; Prostate Gland (#21) &rarr; Suppressed (Mask = 0)</span>
                  <span>&bull; Seminal Vesicles &rarr; Suppressed</span>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-indigo-50 border border-indigo-200 text-indigo-900 flex flex-col gap-1">
                  <span className="font-bold flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-indigo-700" />
                    <span>Active Male Structures (Weight &gt; 0):</span>
                  </span>
                  <span>&bull; Prostate Gland (#21) actively localized</span>
                  <span>&bull; Male pelvic visceral anchor nodes active</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-100 border border-slate-200 text-slate-500 flex flex-col gap-1">
                  <span className="font-bold">Suppressed Discordant Structures:</span>
                  <span>&bull; Uterus (#117) &rarr; Suppressed (Mask = 0)</span>
                  <span>&bull; Ovaries &amp; Vagina &rarr; Suppressed</span>
                </div>
              </div>
            )}
          </div>
        </div>
      ),
    },
    {
      id: 4,
      title: "Hierarchical Multi-Scale Surface Feature Pyramid",
      shortTitle: "Multi-Scale Encoder",
      tagline: "Farthest Point Sampling & Multi-Radius Ball Query Abstraction",
      category: "Encoder Architecture",
      formula: String.raw`\mathbf{f}_i^{(l+1)} = \max_{j \in \mathcal{N}_r(i)} \left( \text{MLP}\left( [\mathbf{f}_j^{(l)} \,\|\, \mathbf{p}_j - \mathbf{p}_i] \right) \right)`,
      formulaLabel: "PointNet++ Set Abstraction with Ball Query Neighborhoods",
      variables: [
        { symbol: String.raw`\mathbf{p}_i`, meaning: "Coordinate position of query centroid point i" },
        { symbol: String.raw`\mathcal{N}_r(i)`, meaning: "Spherical neighborhood of radius r around centroid i" },
        { symbol: String.raw`\mathbf{f}_j^{(l)}`, meaning: "Learned high-dimensional feature vector at layer l" },
        { symbol: String.raw`\|`, meaning: "Feature concatenation with relative coordinate offsets" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Patient surface morphology contains both fine curvature details (clavicles, sternum, ribs) and global posture curvatures (spinal curvature, pelvic tilt). We encode surfaces through a 3-tier hierarchical pyramid:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
            <div className="p-4 bg-white border border-slate-200 rounded-xl shadow-xs flex flex-col gap-1.5">
              <span className="font-bold text-slate-900 text-sm">Tier 1: Local Curvature</span>
              <span className="text-slate-500">FPS Points: 1,024</span>
              <span className="text-slate-500">Ball Radius: r = 0.20</span>
              <span className="font-bold text-primary text-[11px] mt-2">Tensor: [B, 1024, 128]</span>
            </div>
            <div className="p-4 bg-white border border-slate-200 rounded-xl shadow-xs flex flex-col gap-1.5">
              <span className="font-bold text-slate-900 text-sm">Tier 2: Regional Contours</span>
              <span className="text-slate-500">FPS Points: 256</span>
              <span className="text-slate-500">Ball Radius: r = 0.40</span>
              <span className="font-bold text-primary text-[11px] mt-2">Tensor: [B, 256, 256]</span>
            </div>
            <div className="p-4 bg-white border border-slate-200 rounded-xl shadow-xs flex flex-col gap-1.5">
              <span className="font-bold text-slate-900 text-sm">Tier 3: Global Posture</span>
              <span className="text-slate-500">FPS Points: 64</span>
              <span className="text-slate-500">Ball Radius: r = 0.80</span>
              <span className="font-bold text-primary text-[11px] mt-2">Tensor: [B, 64, 512]</span>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 5,
      title: "Dual-Graph SAMeOrganGNN Message Passing Architecture",
      shortTitle: "Dual-Graph GNN",
      tagline: "Rigid Skeletal Kinematic Anchors + Deformable Visceral Cross-Attention",
      category: "Deep GNN Decoder",
      formula: String.raw`\mathbf{h}_v^{(l+1)} = \text{Update}\left( \mathbf{h}_v^{(l)}, \; \sum_{u \in \mathcal{N}(v)} \text{Message}\left(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, \mathbf{e}_{uv}\right) + \mathbf{e}_{\text{sex}} \right)`,
      formulaLabel: "Sex-Conditioned Dual-Graph Message Passing Iteration",
      variables: [
        { symbol: String.raw`\mathbf{h}_v^{(l)}`, meaning: "Node state representation for organ v at message-passing layer l" },
        { symbol: String.raw`\mathcal{N}(v)`, meaning: "Anatomical graph adjacency set (skeletal joints + visceral neighbors)" },
        { symbol: String.raw`\mathbf{e}_{uv}`, meaning: "Learned anatomical edge relation prior (kinematic bone vs soft fascia)" },
        { symbol: String.raw`\mathbf{e}_{\text{sex}}`, meaning: "Conditioning vector modulating sex-specific reproductive subgraphs" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Monolithic point cloud networks fail on internal anatomy because bones and viscera obey fundamentally different biomechanics. <strong>SAMeOrganGNN</strong> decouples the anatomy into a dual-graph representation:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl flex flex-col gap-2">
              <span className="font-bold text-slate-900 text-sm">1. Rigid Skeletal Subgraph (63 Nodes)</span>
              <p className="text-xs text-slate-600 leading-relaxed">
                Spans vertebrae (C1&ndash;L5), cranial base, sternum, pubic symphysis, and iliac crests. Nodes obey rigid distance priors and kinematic articulation constraints.
              </p>
              <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 w-fit">
                Mean Rigidity Error: &le; 6.2 mm
              </span>
            </div>

            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl flex flex-col gap-2">
              <span className="font-bold text-slate-900 text-sm">2. Soft-Tissue Visceral Subgraph (58 Nodes)</span>
              <p className="text-xs text-slate-600 leading-relaxed">
                Spans deformable parenchymal viscera (Brain, Liver, Spleen, Kidneys, Bladder, Uterus). Predicted relative to neighboring skeletal anchors via cross-attention.
              </p>
              <span className="text-[11px] font-mono text-primary-dark bg-primary/10 px-2 py-0.5 rounded border border-primary/20 w-fit">
                Kinematic Anchor Relative Error: 10.8 mm
              </span>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 6,
      title: "Multi-Task Loss & Heteroscedastic Uncertainty Quantification",
      shortTitle: "Loss & Uncertainty",
      tagline: "Smooth L1 Coordinate Regression + 3-Seed Epistemic Disagreement",
      category: "Statistical Calibration",
      formula: String.raw`\mathcal{L}_{\text{joint}} = \sum_{k=1}^K m_k \left[ \text{Smooth}_{L1}(\hat{\mathbf{c}}_k - \mathbf{c}_k) + \frac{\|\hat{\mathbf{c}}_k - \mathbf{c}_k\|_2^2}{2\sigma_k^2} + \frac{1}{2} \log \sigma_k^2 \right]`,
      formulaLabel: "Masked Heteroscedastic Coordinate & Variance Loss",
      variables: [
        { symbol: "m_k", meaning: "Biological validity mask (1 if organ is active, 0 if discordant with sex)" },
        { symbol: String.raw`\mathbf{c}_k`, meaning: "Ground-truth organ centroid from gold-standard CT/MRI segmentation" },
        { symbol: String.raw`\sigma_k`, meaning: "Calibrated aleatoric spatial uncertainty bound (in mm)" },
        { symbol: String.raw`\text{Smooth}_{L1}`, meaning: "Huber loss thresholding coordinate outliers (delta = 1.0 mm)" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            During clinical execution, knowing <em>when</em> a prediction is uncertain is as critical as coordinate accuracy. Epistemic uncertainty is derived across an ensemble of 3 distinct random seed initializations (Seeds 42, 43, 44):
          </p>

          <div className="bg-[#F8F9F5] text-slate-800 p-4 rounded-xl border border-[#D8DCCF] font-mono text-xs sm:text-sm flex flex-col gap-2 shadow-2xs">
            <span className="text-slate-500 text-[11px] uppercase tracking-wider font-semibold">Multi-Seed Epistemic Disagreement Formula:</span>
            <MathView math={String.raw`\sigma_{\text{epistemic}} = \sqrt{ \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \| \hat{\mathbf{c}}_k^{(s)} - \bar{\mathbf{c}}_k \|^2 }`} block={false} />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs pt-1">
            <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 flex flex-col gap-0.5">
              <span className="font-bold">Low Unc (&le; 15.0 mm)</span>
              <span className="text-[11px]">Brain, Heart, Liver, Kidneys, Bladder, Uterus.</span>
            </div>
            <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 flex flex-col gap-0.5">
              <span className="font-bold">Moderate (15&ndash;30 mm)</span>
              <span className="text-[11px]">Vagina, Gallbladder, Pancreatic tail.</span>
            </div>
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex flex-col gap-0.5">
              <span className="font-bold">High Unc (&gt; 30.0 mm)</span>
              <span className="text-[11px]">Extreme anatomical deformation or truncated FOV.</span>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 7,
      title: "Multi-Modal Sensor Ingestion (Depth, RGB & 3D Scans)",
      shortTitle: "Sensor Ingestion",
      tagline: "Pinhole Camera Back-Projection & Photogrammetric Silhouette Lifting",
      category: "Sensor Modality Pipeline",
      formula: String.raw`\mathbf{P}(u, v, d) = \left[ \frac{(u - c_x) \cdot d}{f_x}, \; \frac{(v - c_y) \cdot d}{f_y}, \; d \right]^T \in \mathbb{R}^3`,
      formulaLabel: "Pinhole Camera Ray Unprojection from Depth Sensor Stream",
      variables: [
        { symbol: "u, v", meaning: "Pixel column and row coordinates in the 2D camera sensor plane" },
        { symbol: "d", meaning: "Calibrated 16-bit metric depth distance at sensor pixel (u, v)" },
        { symbol: "f_x, f_y", meaning: "Focal lengths from camera intrinsic calibration matrix K" },
        { symbol: "c_x, c_y", meaning: "Principal point optical center on the CMOS plane" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            To permit universal deployment without dedicated laser scanners, our pipeline supports three distinct clinical input modalities:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div
              onClick={() => setSensorTab("depth")}
              className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col gap-2 ${
                sensorTab === "depth"
                  ? "bg-primary/10 border-primary shadow-xs ring-1 ring-primary/40"
                  : "bg-white border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex items-center gap-2 font-bold text-slate-900 text-xs sm:text-sm">
                <Camera className="w-4 h-4 text-primary-dark" />
                <span>3D Depth Camera (RGB-D)</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Intel RealSense / Azure Kinect. Directly unprojects 16-bit depth pixels into 3D metric coordinates using pinhole ray equations.
              </p>
            </div>

            <div
              onClick={() => setSensorTab("rgb")}
              className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col gap-2 ${
                sensorTab === "rgb"
                  ? "bg-amber-50/70 border-amber-300 shadow-xs ring-1 ring-amber-300"
                  : "bg-white border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex items-center gap-2 font-bold text-slate-900 text-xs sm:text-sm">
                <Eye className="w-4 h-4 text-amber-600" />
                <span>Clinical RGB Photograph</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Standard anterior clinical photos. Segment patient body contour and lift 2D silhouette landmarks into a 3D metric boundary.
              </p>
            </div>

            <div
              onClick={() => setSensorTab("mesh")}
              className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col gap-2 ${
                sensorTab === "mesh"
                  ? "bg-emerald-50/70 border-emerald-300 shadow-xs ring-1 ring-emerald-300"
                  : "bg-white border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex items-center gap-2 font-bold text-slate-900 text-xs sm:text-sm">
                <Scan className="w-4 h-4 text-emerald-600" />
                <span>3D Mesh (.PLY / .OBJ)</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Optical or laser scanners. Vertices are uniformly sub-sampled via Farthest Point Sampling to exactly N = 4,096 points.
              </p>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 8,
      title: "Scientific Ablation & Architecture Benchmark Comparison",
      shortTitle: "Ablation Matrix",
      tagline: "Comprehensive Multi-Cohort Validation Across 450 Verified Patient Volumes",
      category: "Empirical Validation",
      formula: String.raw`\text{TRE} = \frac{1}{K} \sum_{k=1}^K \|\hat{\mathbf{c}}_k - \mathbf{c}_k^{\text{gold}}\|_2 \quad [\text{Target Registration Error in mm}]`,
      formulaLabel: "Mean Target Registration Error Across Anatomical Cohort",
      variables: [
        { symbol: String.raw`\mathbf{c}_k^{\text{gold}}`, meaning: "Voxel ground-truth centroid from contrast-enhanced CT/MRI" },
        { symbol: String.raw`\hat{\mathbf{c}}_k`, meaning: "Inferred 3D organ centroid from external surface geometry" },
        { symbol: "K", meaning: "Number of evaluated anatomical targets in cohort (104 - 121)" },
      ],
      content: (
        <div className="flex flex-col gap-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <p>
            Evaluated on the benchmark dataset comprising CT-ORG, TotalSegmentator, and HuMMan surface cohorts:
          </p>

          <div className="overflow-x-auto border border-slate-200 rounded-xl">
            <table className="w-full text-xs font-mono text-left bg-white">
              <thead className="bg-slate-100 text-slate-800 font-bold border-b border-slate-200">
                <tr>
                  <th className="p-3">Architecture</th>
                  <th className="p-3">Sex Conditioning</th>
                  <th className="p-3">Targets</th>
                  <th className="p-3">Brain TRE</th>
                  <th className="p-3">Torso TRE</th>
                  <th className="p-3">Pelvic/Repro</th>
                  <th className="p-3">Overall Mean TRE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                <tr>
                  <td className="p-3 font-medium text-slate-700 font-sans">Population Atlas (C0)</td>
                  <td className="p-3 text-slate-400">None</td>
                  <td className="p-3">104</td>
                  <td className="p-3 text-rose-600">42.8 mm</td>
                  <td className="p-3 text-rose-600">28.4 mm</td>
                  <td className="p-3 text-rose-600">32.1 mm</td>
                  <td className="p-3 font-bold text-rose-600">31.2 mm</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-slate-700 font-sans">PointNet++ Baseline (C2)</td>
                  <td className="p-3 text-slate-400">None</td>
                  <td className="p-3">104</td>
                  <td className="p-3 text-amber-600">24.5 mm</td>
                  <td className="p-3 text-amber-600">18.9 mm</td>
                  <td className="p-3 text-amber-600">22.4 mm</td>
                  <td className="p-3 font-bold text-amber-600">20.8 mm</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-slate-700 font-sans">DGCNN EdgeConv (C3)</td>
                  <td className="p-3 text-slate-400">None</td>
                  <td className="p-3">104</td>
                  <td className="p-3 text-amber-600">21.2 mm</td>
                  <td className="p-3 text-amber-600">17.1 mm</td>
                  <td className="p-3 text-amber-600">19.8 mm</td>
                  <td className="p-3 font-bold text-amber-600">18.4 mm</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-slate-700 font-sans">Phase 10R Baseline (C4)</td>
                  <td className="p-3 text-slate-400">None (Unisex)</td>
                  <td className="p-3">104</td>
                  <td className="p-3 text-amber-600">18.6 mm</td>
                  <td className="p-3 text-emerald-600">11.8 mm</td>
                  <td className="p-3 text-amber-600">16.5 mm</td>
                  <td className="p-3 font-bold text-emerald-600">14.2 mm</td>
                </tr>
                <tr>
                  <td className="p-3 font-medium text-slate-700 font-sans">Phase 16 Cranial Prior Guard</td>
                  <td className="p-3 text-slate-400">None</td>
                  <td className="p-3">117</td>
                  <td className="p-3 text-emerald-600">9.4 mm</td>
                  <td className="p-3 text-emerald-600">11.6 mm</td>
                  <td className="p-3 text-amber-600">15.9 mm</td>
                  <td className="p-3 font-bold text-emerald-600">12.8 mm</td>
                </tr>
                <tr className="bg-emerald-50/70 font-semibold border-t-2 border-emerald-300">
                  <td className="p-3 text-emerald-950 font-sans flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-emerald-600" />
                    <span>Proposed Sex-Aware SAMeOrganGNN</span>
                  </td>
                  <td className="p-3 text-emerald-800">Dynamic Dimorphism</td>
                  <td className="p-3 text-emerald-800">121</td>
                  <td className="p-3 text-emerald-700 font-bold">9.2 mm</td>
                  <td className="p-3 text-emerald-700 font-bold">10.8 mm</td>
                  <td className="p-3 text-emerald-700 font-bold">13.8 mm</td>
                  <td className="p-3 font-bold text-emerald-700 text-sm">11.6 mm</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ),
    },
  ];

  const currentStage = STAGES[activeStage];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-5 rounded-2xl border border-border shadow-xs">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-xs font-mono text-primary-dark w-fit mb-1.5">
            <Activity className="w-3.5 h-3.5 text-accent-green" />
            <span>CAIR IIT Mandi &bull; Theoretical &amp; Computational Foundation</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Methodology &amp; Mathematical Engine
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1 max-w-3xl">
            Interactive breakdown of the radiation-free spatial mapping operator from external patient surface manifolds to internal 3D organ centroids.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-600 bg-slate-100 px-3 py-1.5 rounded-xl border border-slate-200 shrink-0 self-start md:self-auto">
          <span className="font-semibold text-slate-900">8 Architectural Stages</span>
          <span>&bull;</span>
          <span>Sex-Aware 121-Organ Model</span>
        </div>
      </div>

      {/* Interactive Horizontal Pipeline Stepper (No Long Scroll!) */}
      <div className="bg-white p-2 rounded-2xl border border-slate-200 shadow-xs">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-1.5">
          {STAGES.map((s, idx) => {
            const isActive = activeStage === idx;
            return (
              <button
                key={s.id}
                onClick={() => setActiveStage(idx)}
                className={`flex flex-col text-left p-2.5 rounded-xl transition-all border ${
                  isActive
                    ? "bg-[#465133] text-white border-[#465133] shadow-md ring-2 ring-primary/50"
                    : "bg-slate-50/70 hover:bg-slate-100 text-slate-700 border-slate-200/80"
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span
                    className={`font-mono text-[10px] font-bold px-1.5 py-0.2 rounded ${
                      isActive ? "bg-white/20 text-white" : "bg-slate-200 text-slate-600"
                    }`}
                  >
                    0{s.id}
                  </span>
                  {isActive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
                </div>
                <span className="font-bold text-xs mt-1.5 leading-snug line-clamp-1">
                  {s.shortTitle}
                </span>
                <span
                  className={`text-[10px] truncate mt-0.5 ${
                    isActive ? "text-slate-300" : "text-slate-400"
                  }`}
                >
                  {s.category}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Stage Interactive Workbench */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
        {/* Stage Header Banner */}
        <div className="p-5 sm:p-6 border-b border-slate-200 bg-gradient-to-r from-slate-50 via-white to-slate-50 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#465133] text-white flex items-center justify-center font-mono font-bold text-base shadow-sm shrink-0">
              0{currentStage.id}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
                  {currentStage.category}
                </span>
                <span className="text-slate-300">&bull;</span>
                <span className="text-[11px] font-mono text-slate-500">
                  Stage {currentStage.id} of 8
                </span>
              </div>
              <h2 className="font-bold text-lg sm:text-xl text-slate-900 leading-snug">
                {currentStage.title}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">{currentStage.tagline}</p>
            </div>
          </div>

          {/* Quick Stage Switcher Arrows */}
          <div className="flex items-center gap-1.5 self-end sm:self-auto">
            <button
              onClick={() => setActiveStage(Math.max(0, activeStage - 1))}
              disabled={activeStage === 0}
              className={`p-2 rounded-xl border text-xs font-semibold flex items-center gap-1 transition-all ${
                activeStage === 0
                  ? "bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed"
                  : "bg-white hover:bg-slate-50 text-slate-700 border-slate-200 shadow-xs"
              }`}
            >
              <ChevronLeft className="w-4 h-4" />
              <span className="hidden sm:inline">Prev</span>
            </button>
            <button
              onClick={() => setActiveStage(Math.min(STAGES.length - 1, activeStage + 1))}
              disabled={activeStage === STAGES.length - 1}
              className={`p-2 rounded-xl border text-xs font-semibold flex items-center gap-1 transition-all ${
                activeStage === STAGES.length - 1
                  ? "bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed"
                  : "bg-[#465133] hover:bg-[#384228] text-white border-[#465133] shadow-xs"
              }`}
            >
              <span className="hidden sm:inline">Next</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="p-5 sm:p-7 flex flex-col gap-6">
          {/* Scientific Mathematical Card with High-Contrast KaTeX (Light Olive Theme) */}
          <div className="bg-[#F8F9F5] text-slate-800 rounded-2xl border border-[#D8DCCF] p-5 sm:p-6 shadow-xs flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-[#D8DCCF] pb-3">
              <span className="text-xs font-mono font-bold text-[#465133] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-primary" />
                <span>{currentStage.formulaLabel}</span>
              </span>
              <span className="text-[10px] font-mono text-slate-600 bg-white px-2 py-0.5 rounded border border-[#D8DCCF]">
                KaTeX Typeset &bull; Exact Formulation
              </span>
            </div>

            {/* Rendered Math Formula with KaTeX */}
            <div className="py-2 overflow-x-auto">
              <MathView math={currentStage.formula} block={true} className="bg-white text-slate-900 border-[#D8DCCF]" />
            </div>

            {/* Variable Decoder Grid */}
            <div className="mt-2 pt-3 border-t border-[#D8DCCF] flex flex-col gap-2">
              <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-600">
                Variable Definitions &amp; Dimensionality:
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {currentStage.variables.map((v, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-xs bg-white p-2.5 rounded-lg border border-[#D8DCCF] shadow-2xs"
                  >
                    <span className="font-mono font-bold text-primary-dark shrink-0">
                      <MathView math={v.symbol} block={false} />:
                    </span>
                    <span className="text-slate-600 font-sans text-[11px]">{v.meaning}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Interactive Feature & Details Section */}
          <div className="p-5 sm:p-6 rounded-2xl bg-slate-50/60 border border-slate-200">
            {currentStage.content}
          </div>
        </div>

        {/* Bottom Stepper Bar */}
        <div className="p-4 border-t border-slate-200 bg-slate-50/80 flex items-center justify-between">
          <button
            onClick={() => setActiveStage(Math.max(0, activeStage - 1))}
            disabled={activeStage === 0}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              activeStage === 0
                ? "text-slate-400 cursor-not-allowed"
                : "text-slate-700 hover:text-slate-900 hover:bg-white border border-transparent hover:border-slate-200"
            }`}
          >
            <ChevronLeft className="w-4 h-4" />
            <span>
              Previous: {activeStage > 0 ? STAGES[activeStage - 1].shortTitle : "Start"}
            </span>
          </button>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold text-slate-700">
              Stage 0{activeStage + 1} / 0{STAGES.length}
            </span>
          </div>

          <button
            onClick={() => setActiveStage(Math.min(STAGES.length - 1, activeStage + 1))}
            disabled={activeStage === STAGES.length - 1}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              activeStage === STAGES.length - 1
                ? "text-slate-400 cursor-not-allowed"
                : "bg-[#465133] hover:bg-[#384228] text-white shadow-xs"
            }`}
          >
            <span>
              Next:{" "}
              {activeStage < STAGES.length - 1 ? STAGES[activeStage + 1].shortTitle : "Complete"}
            </span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
