import React from "react";
import Image from "next/image";
import { User, Award, GraduationCap, Building2, Cpu, Sparkles, Package, Github, ExternalLink } from "lucide-react";

export default function TeamPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12 flex flex-col gap-10">
      {/* Logos & Header Section */}
      <div className="text-center max-w-3xl mx-auto flex flex-col items-center gap-4">
        {/* Dual Institution Logos: CAIR + IIT Mandi */}
        <div className="flex items-center justify-center gap-6 p-4 rounded-2xl bg-white border border-border shadow-xs">
          <img
            src="/cair_logo.png"
            alt="CAIR IIT Mandi Logo"
            className="h-16 w-auto object-contain"
          />
          <div className="h-12 w-px bg-border" />
          <img
            src="/iit_mandi_logo.png"
            alt="IIT Mandi Logo"
            className="h-14 w-auto object-contain"
          />
        </div>

        <div className="inline-flex items-center justify-center gap-2 px-3.5 py-1 rounded-full bg-white border border-border text-xs font-mono text-primary-dark shadow-xs">
          <GraduationCap className="w-3.5 h-3.5 text-accent-green" />
          <span>Centre for Artificial Intelligence and Robotics (CAIR) &bull; IIT Mandi</span>
        </div>

        <h1 className="text-3xl sm:text-4xl font-extrabold text-text-main tracking-tight">
          Research Team &amp; Academic Supervision
        </h1>
        <p className="text-sm sm:text-base text-text-muted leading-relaxed">
          Surface2Anatomy was researched and engineered by research interns at the <strong>Centre for Artificial Intelligence and Robotics (CAIR), Indian Institute of Technology Mandi (IIT Mandi)</strong> under the academic supervision of <strong>Dr. Deepak Raina</strong>.
        </p>
      </div>

      {/* 3-Column Team Grid: Khushi Mhamane -> Sharon Melhi -> Dr. Deepak Raina */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
        {/* Card 1: Khushi Mhamane - Project Lead */}
        <div className="bg-white rounded-2xl border-2 border-primary/40 p-6 shadow-xs flex flex-col justify-between gap-4 relative overflow-hidden">
          <div className="absolute top-0 right-0 bg-primary text-white text-[10px] font-mono px-3 py-0.5 rounded-bl-lg font-semibold tracking-wider uppercase">
            Project Lead
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
              <div className="w-24 h-24 rounded-2xl overflow-hidden border-2 border-primary/30 shadow-sm shrink-0 bg-slate-100">
                <img
                  src="/team/khushi_mhamane.jpg"
                  alt="Khushi Mhamane"
                  className="w-full h-full object-cover object-top"
                />
              </div>
              <span className="text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200 shrink-0 mr-16">
                Project Lead
              </span>
            </div>

            <div>
              <div className="text-[11px] font-mono text-primary font-bold uppercase tracking-wider mb-0.5">
                Project Lead &bull; Research Intern
              </div>
              <h2 className="text-xl font-bold text-text-main">Khushi Mhamane</h2>
              <p className="text-xs font-medium text-primary-dark mt-0.5">
                Research Intern &bull; CAIR, IIT Mandi
              </p>
              <a
                href="mailto:khushimhamane@gmail.com"
                className="text-[11px] text-accent-green hover:underline font-mono mt-1 inline-block"
              >
                khushimhamane@gmail.com
              </a>
            </div>

            <p className="text-xs text-text-muted leading-relaxed">
              Led the project formulation, 3D point cloud transformer architecture design, multi-organ loss balancing, and clinical validation across FLARE22, AMOS22, and CT-ORG benchmarks.
            </p>
          </div>

          <div className="pt-3 border-t border-border text-[11px] text-text-muted font-mono flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Building2 className="w-3.5 h-3.5 text-primary shrink-0" />
              <span className="truncate">CAIR, IIT Mandi</span>
            </div>
          </div>
        </div>

        {/* Card 2: Sharon Melhi - Research Intern */}
        <div className="bg-white rounded-2xl border-2 border-primary/30 p-6 shadow-xs flex flex-col justify-between gap-4 relative">
          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
              <div className="w-24 h-24 rounded-2xl overflow-hidden border-2 border-primary/30 shadow-sm shrink-0 bg-slate-100">
                <img
                  src="/team/sharon_melhi.png"
                  alt="Sharon Melhi"
                  className="w-full h-full object-cover object-top"
                />
              </div>
              <span className="text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full bg-primary/10 text-primary-dark border border-primary/20 shrink-0">
                Research Intern
              </span>
            </div>

            <div>
              <div className="text-[11px] font-mono text-primary font-bold uppercase tracking-wider mb-0.5">
                Research Intern
              </div>
              <h2 className="text-xl font-bold text-text-main">Sharon Melhi</h2>
              <p className="text-xs font-medium text-primary-dark mt-0.5">
                Research Intern &bull; CAIR, IIT Mandi
              </p>
              <a
                href="mailto:sharonmelhi365@gmail.com"
                className="text-[11px] text-accent-green hover:underline font-mono mt-1 inline-block"
              >
                sharonmelhi365@gmail.com
              </a>
            </div>

            <p className="text-xs text-text-muted leading-relaxed">
              Engineered the frozen Ridge canonical alignment pipeline, 3-seed ensemble epistemic uncertainty estimation, whole-body brain-aware augmentation, and GPU inference web deployment.
            </p>
          </div>

          <div className="pt-3 border-t border-border text-[11px] text-text-muted font-mono flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Building2 className="w-3.5 h-3.5 text-primary shrink-0" />
              <span className="truncate">CAIR, IIT Mandi</span>
            </div>
          </div>
        </div>

        {/* Card 3: Dr. Deepak Raina - Academic Project Supervisor */}
        <div className="bg-white rounded-2xl border-2 border-primary/30 p-6 shadow-xs flex flex-col justify-between gap-4 relative">
          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
              <div className="w-24 h-24 rounded-2xl overflow-hidden border-2 border-primary/30 shadow-sm shrink-0 bg-slate-100">
                <img
                  src="/team/deepak_raina.png"
                  alt="Dr. Deepak Raina"
                  className="w-full h-full object-cover object-top"
                />
              </div>
              <span className="text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full bg-primary/10 text-primary-dark border border-primary/20 shrink-0">
                Faculty Supervisor
              </span>
            </div>

            <div>
              <div className="text-[11px] font-mono text-accent-green font-semibold uppercase tracking-wider mb-0.5">
                Academic Project Supervisor
              </div>
              <h2 className="text-xl font-bold text-text-main">Dr. Deepak Raina</h2>
              <p className="text-xs font-medium text-primary-dark mt-0.5">
                Assistant Professor &bull; CAIR, IIT Mandi
              </p>
              <a
                href="mailto:deepakraina@iitmandi.ac.in"
                className="text-[11px] text-accent-green hover:underline font-mono mt-1 inline-block"
              >
                deepakraina@iitmandi.ac.in
              </a>
            </div>

            <p className="text-xs text-text-muted leading-relaxed">
              Supervised the architectural formulation, medical coordinate system invariance, cross-attention mechanism design, and scientific validation paradigms for radiation-free internal organ localization.
            </p>
          </div>

          <div className="pt-3 border-t border-border text-[11px] text-text-muted font-mono flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Building2 className="w-3.5 h-3.5 text-primary shrink-0" />
              <span className="truncate">CAIR, IIT Mandi</span>
            </div>
          </div>
        </div>
      </div>


      {/* Institutional Banner */}
      <div className="bg-[#FAF8F5] border border-border rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-6 shadow-xs">
        <div className="flex items-center gap-4">
          <img
            src="/cair_logo.png"
            alt="CAIR Logo"
            className="h-14 w-auto object-contain shrink-0"
          />
          <div className="flex flex-col text-xs text-text-muted">
            <span className="font-bold text-sm text-text-main">
              Centre for Artificial Intelligence and Robotics (CAIR)
            </span>
            <span>Indian Institute of Technology Mandi (IIT Mandi) &bull; Kamand Campus, Mandi, Himachal Pradesh 175005, India</span>
            <span className="text-[11px] text-primary-dark mt-0.5">Specialized centre pioneering intelligent robotics, biomedical perception, and surgical autonomy.</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <a
            href="https://pypi.org/project/surface2anatomy/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-border text-xs font-mono text-slate-800 hover:text-primary transition-colors shadow-xs"
          >
            <Package className="w-3.5 h-3.5 text-primary" />
            <span>PyPI</span>
          </a>
          <a
            href="https://huggingface.co/spaces/Sharon-codes/surface2anatomy-backend"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-border text-xs font-mono text-slate-800 hover:text-amber-700 transition-colors shadow-xs"
          >
            <span>🤗</span>
            <span>Hugging Face</span>
          </a>
          <a
            href="https://github.com/Sharon-codes/S2A-Net-IIT-Mandi"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-border text-xs font-mono text-slate-800 hover:text-slate-950 transition-colors shadow-xs"
          >
            <Github className="w-3.5 h-3.5 text-slate-800" />
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </div>
  );
}
