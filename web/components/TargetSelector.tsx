"use client";

import React, { useState, useEffect } from "react";
import { Search, Check, Sparkles, User, Lock, ShieldCheck } from "lucide-react";


interface TargetSelectorProps {
  selectedTargets: string[];
  onToggleTarget: (target: string) => void;
  onClearTargets: () => void;
  onSelectTargets: (targets: string[]) => void;
  onRunInference: () => void;
  isPredicting: boolean;
  hasGeometry: boolean;
  patientSex?: "auto" | "female" | "male";
  onPatientSexChange?: (sex: "auto" | "female" | "male") => void;
}

const COMMON_BENCHMARKS = [
  { id: "brain", label: "Brain" },
  { id: "heart", label: "Heart" },
  { id: "liver", label: "Liver" },
  { id: "kidney_left", label: "Left Kidney" },
  { id: "kidney_right", label: "Right Kidney" },
  { id: "urinary_bladder", label: "Bladder" },
  { id: "spleen", label: "Spleen" },
  { id: "aorta", label: "Aorta" },
  { id: "pancreas", label: "Pancreas" },
];

const FEMALE_REPRODUCTIVE = [
  { id: "uterus", label: "Uterus" },
  { id: "ovary_left", label: "Left Ovary" },
  { id: "ovary_right", label: "Right Ovary" },
  { id: "vagina", label: "Vagina" },
];

const MALE_REPRODUCTIVE = [
  { id: "prostate", label: "Prostate Gland" },
];

export function TargetSelector({
  selectedTargets,
  onToggleTarget,
  onClearTargets,
  onSelectTargets,
  onRunInference,
  isPredicting,
  hasGeometry,
  patientSex = "female",
  onPatientSexChange,
}: TargetSelectorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [allTargets, setAllTargets] = useState<string[]>([]);

  // Fetch targets from API catalog
  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL || "https://sharonmelhi-surface2anatomy-backend.hf.space";
    fetch(`${apiUrl}/targets`)
      .then((res) => res.json())
      .then((data) => {
        const targetList = data.all_targets || data.targets;
        if (targetList) {
          setAllTargets(targetList);
        }
      })
      .catch(() => {
        // Fallback default list
        setAllTargets([
          ...COMMON_BENCHMARKS.map((t) => t.id),
          ...FEMALE_REPRODUCTIVE.map((t) => t.id),
          ...MALE_REPRODUCTIVE.map((t) => t.id),
        ]);
      });
  }, []);

  const filteredTargets = allTargets.filter((t) =>

    t.toLowerCase().replace(/_/g, " ").includes(searchQuery.toLowerCase().trim())
  );

  return (
    <div className="flex flex-col gap-4 bg-white p-4 sm:p-5 rounded-2xl border border-border shadow-xs">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-3">
        <h3 className="text-xs font-bold text-text-main uppercase tracking-wider">
          2. Target Organs &amp; Biological Sex
        </h3>
        <span className="text-[11px] font-mono text-primary-dark font-semibold bg-primary/10 px-2.5 py-0.5 rounded-full border border-primary/20 w-fit">
          Sex-Aware 121-Organ Ensemble
        </span>
      </div>

      {/* Patient Biological Sex: Locked / Auto-detected from Scan Geometry */}
      <div className="flex flex-col gap-2 p-3 rounded-xl bg-background border border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
            <User className="w-3.5 h-3.5 text-primary" />
            <span>Biological Sex:</span>
          </div>

          {/* Auto-detected status pill with lock icon (Option disabled as requested) */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-border text-xs font-mono shadow-2xs">
            <Lock className="w-3 h-3 text-slate-400" />
            <span className="text-[11px] text-slate-500 font-sans">Auto-detected:</span>
            {patientSex === "female" ? (
              <span className="font-bold text-fuchsia-700 flex items-center gap-1">
                Female ♀
              </span>
            ) : (
              <span className="font-bold text-primary-dark flex items-center gap-1">
                Male ♂
              </span>
            )}
          </div>
        </div>

        {/* Dynamic Reproductive Organs based on Detected Sex */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1.5 border-t border-border/80">
          <span className="text-[11px] font-medium text-slate-500">Reproductive Priors:</span>
          {patientSex === "female" ? (
            FEMALE_REPRODUCTIVE.map((t) => {
              const isSelected = selectedTargets.includes(t.id);
              return (
                <button
                  key={t.id}
                  onClick={() => onToggleTarget(t.id)}
                  className={`px-2 py-0.5 rounded-md text-[11px] font-medium transition-all flex items-center gap-1 border ${
                    isSelected
                      ? "bg-fuchsia-600 text-white border-fuchsia-600 shadow-xs"
                      : "bg-white text-fuchsia-900 border-fuchsia-200 hover:bg-fuchsia-50"
                  }`}
                >
                  {t.label}
                  {isSelected && <Check className="w-2.5 h-2.5" />}
                </button>
              );
            })
          ) : (
            MALE_REPRODUCTIVE.map((t) => {
              const isSelected = selectedTargets.includes(t.id);
              return (
                <button
                  key={t.id}
                  onClick={() => onToggleTarget(t.id)}
                  className={`px-2 py-0.5 rounded-md text-[11px] font-medium transition-all flex items-center gap-1 border ${
                    isSelected
                      ? "bg-primary text-white border-primary shadow-xs"
                      : "bg-white text-slate-800 border-border hover:bg-primary/5"
                  }`}
                >
                  {t.label}
                  {isSelected && <Check className="w-2.5 h-2.5" />}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Target Search */}
      <div className="relative w-full">
        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
        <input
          type="text"
          placeholder="Search 121 organs (e.g. uterus, liver, aorta, brain)..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-9 pr-8 py-2 text-sm rounded-xl border border-border bg-slate-50/50 focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600"
          >
            ✕
          </button>
        )}
      </div>


      {/* Quick Select Common Benchmark Organ Chips (Styled with Olive Green - NO Dark Blue!) */}
      <div>
        <span className="text-xs font-medium text-text-muted block mb-2">Primary Clinical Benchmarks:</span>
        <div className="flex flex-wrap gap-1.5">
          {COMMON_BENCHMARKS.map((t) => {
            const isSelected = selectedTargets.includes(t.id);
            return (
              <button
                key={t.id}
                onClick={() => onToggleTarget(t.id)}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1 border ${
                  isSelected
                    ? "bg-primary text-white border-primary shadow-xs"
                    : "bg-white hover:bg-slate-100 text-slate-700 border-slate-200"
                }`}
              >
                {t.label}
                {isSelected && <Check className="w-3 h-3 ml-0.5" />}
              </button>
            );
          })}
        </div>
      </div>

      {/* Filtered Search Results Dropdown */}
      {searchQuery.trim().length > 0 && (
        <div className="max-h-40 overflow-y-auto border border-border rounded-xl p-2 bg-slate-50 flex flex-wrap gap-1">
          {filteredTargets.slice(0, 24).map((t) => (
            <button
              key={t}
              onClick={() => {
                onToggleTarget(t);
                setSearchQuery("");
              }}
              className={`px-2 py-0.5 rounded-lg text-xs border ${
                selectedTargets.includes(t)
                  ? "bg-primary text-white border-primary"
                  : "bg-white hover:bg-slate-100 border-slate-200 text-slate-800"
              }`}
            >
              {t.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      )}

      {/* Selected Targets Summary & Action Bar */}
      <div className="border-t border-border pt-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-text-muted">
            Selected: ({selectedTargets.length})
          </span>
          {selectedTargets.length > 0 && (
            <button
              onClick={onClearTargets}
              className="text-xs text-text-muted hover:text-accent-red underline"
            >
              Clear all
            </button>
          )}
        </div>

        <button
          onClick={onRunInference}
          disabled={!hasGeometry || selectedTargets.length === 0 || isPredicting}
          className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all flex items-center gap-2 shadow-sm ${
            !hasGeometry || selectedTargets.length === 0 || isPredicting
              ? "bg-gray-200 text-gray-400 cursor-not-allowed"
              : "bg-primary hover:bg-primary-dark text-white cursor-pointer"
          }`}
        >
          {isPredicting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Predicting Centroids...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Predict 3D Centroids</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
