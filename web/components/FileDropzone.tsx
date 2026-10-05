"use client";

import React, { useRef, useState } from "react";
import {
  Upload,
  Camera,
  Layers,
  CheckCircle,
  Image as ImageIcon,
  Sparkles,
  User,
  FileCheck,
  Cpu,
  RotateCcw,
} from "lucide-react";

interface FileDropzoneProps {
  onFileLoaded: (file: File | { name: string; content: ArrayBuffer }) => void;
  isLoading: boolean;
  activeFileName: string | null;
  activeModality?: "mesh" | "depth" | "rgb";
  detectedSex?: "female" | "male";
  pointCount?: number;
  onModalityChange?: (modality: "mesh" | "depth" | "rgb") => void;
  imageUrl?: string | null;
}

export function FileDropzone({
  onFileLoaded,
  isLoading,
  activeFileName,
  activeModality = "mesh",
  detectedSex = "female",
  pointCount = 4096,
  onModalityChange,
  imageUrl = null,
}: FileDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleProcessFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleProcessFile = (file: File) => {
    const lower = file.name.toLowerCase();
    if (lower.endsWith(".png") || lower.endsWith(".tiff") || lower.endsWith(".exr")) {
      if (lower.includes("depth") || lower.includes("realsense") || lower.includes("d435")) {
        onModalityChange?.("depth");
      } else {
        onModalityChange?.("rgb");
      }
    } else if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".webp")) {
      onModalityChange?.("rgb");
    } else {
      onModalityChange?.("mesh");
    }
    onFileLoaded(file);
  };

  const loadPreset = async (type: "depth_camera" | "clinical_rgb" | "canonical_scan") => {
    try {
      let filename = "";
      let outputName = "";
      const isMale = detectedSex === "male";

      if (type === "depth_camera") {
        filename = "sample_depth_camera.png";
        outputName = isMale ? "sample_patient_male_depth.png" : "sample_patient_female_depth.png";
        onModalityChange?.("depth");
      } else if (type === "clinical_rgb") {
        filename = isMale ? "sample_patient_male_rgb.jpg" : "sample_patient_female_rgb.jpg";
        outputName = isMale ? "sample_patient_male_rgb.jpg" : "sample_patient_female_rgb.jpg";
        onModalityChange?.("rgb");
      } else {
        filename = isMale ? "sample_male_body.ply" : "sample_female_body.ply";
        outputName = isMale ? "sample_male_scan_4096.ply" : "sample_female_scan_4096.ply";
        onModalityChange?.("mesh");
      }

      const res = await fetch(`/demo/${filename}`);
      if (!res.ok) throw new Error(`Failed to load ${filename}`);
      const buffer = await res.arrayBuffer();
      onFileLoaded({
        name: outputName,
        content: buffer,
      });
    } catch (err) {
      console.error("Failed to load preset scan:", err);
    }
  };

  return (
    <div className="flex flex-col gap-3.5 w-full bg-white p-4 sm:p-5 rounded-2xl border border-border shadow-xs">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-text-main uppercase tracking-wider">
          1. Patient Input &amp; Preprocessing
        </span>
        <span className="text-[11px] font-mono text-emerald-700 font-semibold flex items-center gap-1 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
          <CheckCircle className="w-3 h-3" />
          <span>Multi-Modal Ingestion</span>
        </span>
      </div>

      {/* Universal Drag & Drop Upload Zone (Accepts RGB, Depth Cam, or 3D Scans) */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-2 ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-slate-300 bg-background hover:border-primary/50"
        }`}
      >
        <div className="w-10 h-10 rounded-full bg-primary/10 text-primary-dark flex items-center justify-center">
          <Upload className="w-5 h-5 text-primary" />
        </div>

        <div>
          <p className="text-xs font-bold text-slate-900">
            Upload Patient Image, Depth Frame, or 3D Scan
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5 max-w-xs leading-relaxed">
            Drag &amp; drop any clinical RGB photo (.jpg/.png), 3D Depth map (.png/.exr), or 3D mesh (.ply/.obj).
          </p>
        </div>

        <div className="flex items-center gap-2 pt-1">
          <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-md bg-white border border-border text-slate-600">
            RGB Photos
          </span>
          <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-md bg-white border border-border text-slate-600">
            Depth Camera
          </span>
          <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-md bg-white border border-border text-slate-600">
            3D Mesh (.ply)
          </span>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.ply,.obj,.stl,.xyz,.exr,.tiff"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleProcessFile(e.target.files[0]);
          }
        }}
      />

      {/* Live Preprocessing Status Box */}
      <div className="p-3 rounded-xl bg-[#F8F9F5] border border-[#D8DCCF] flex flex-col gap-2 shadow-2xs">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-800 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <span>Active Patient Data:</span>
          </span>
          <span className="text-[11px] font-mono font-medium text-slate-600 truncate max-w-[180px]">
            {activeFileName || "sample_patient.ply"}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1 border-t border-border/80">
          <div className="bg-white p-2 rounded-lg border border-border flex flex-col">
            <span className="text-slate-400 font-sans text-[10px]">Detected Modality:</span>
            <span className="font-bold text-slate-800 capitalize">
              {activeModality === "mesh"
                ? "3D Surface Mesh"
                : activeModality === "depth"
                ? "3D Depth Camera"
                : "Clinical RGB Photo"}
            </span>
          </div>

          <div className="bg-white p-2 rounded-lg border border-border flex flex-col">
            <span className="text-slate-400 font-sans text-[10px]">Preprocessed Surface:</span>
            <span className="font-bold text-emerald-700">
              {pointCount.toLocaleString()} Metric Points
            </span>
          </div>
        </div>

        {/* Clinical Input Preview Thumbnail (For RGB Photos & 3D Depth Frames) */}
        {(activeModality === "rgb" || activeModality === "depth") && (
          <div className="flex items-center gap-3 p-2 rounded-lg bg-white border border-border mt-0.5">
            <div className="w-12 h-12 rounded-md overflow-hidden bg-slate-100 border border-slate-200 shrink-0 relative flex items-center justify-center">
              <img
                src={
                  imageUrl ||
                  (activeModality === "depth"
                    ? "/demo/sample_depth_camera.png"
                    : detectedSex === "male"
                    ? "/demo/sample_patient_male_rgb.jpg"
                    : "/demo/sample_patient_female_rgb.jpg")
                }
                alt="Patient Clinical Input"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="flex flex-col min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-slate-800 uppercase tracking-tight">
                  {activeModality === "depth" ? "16-Bit Depth Frame" : "Clinical Photograph"}
                </span>
                <span className="text-[9px] font-mono text-emerald-700 bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200 font-semibold">
                  Extracted
                </span>
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5 leading-tight">
                {activeModality === "depth"
                  ? "RealSense depth field registered to canonical CT frame."
                  : "Optical portrait registered to canonical CT frame."}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Quick Modality Sample Presets */}
      <div className="flex items-center justify-between pt-1 border-t border-border">
        <span className="text-[11px] font-mono text-slate-500">Quick Samples:</span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => loadPreset("depth_camera")}
            className="px-2 py-0.5 text-[11px] rounded-md font-medium bg-background hover:bg-slate-200 border border-border text-slate-700 transition-colors flex items-center gap-1"
            title="Load RealSense Depth Sensor Stream"
          >
            <Camera className="w-3 h-3 text-primary-dark" />
            <span>Depth Cam</span>
          </button>
          <button
            type="button"
            onClick={() => loadPreset("clinical_rgb")}
            className="px-2 py-0.5 text-[11px] rounded-md font-medium bg-background hover:bg-slate-200 border border-border text-slate-700 transition-colors flex items-center gap-1"
            title="Load Clinical Patient Photograph"
          >
            <ImageIcon className="w-3 h-3 text-amber-700" />
            <span>RGB Photo</span>
          </button>
          <button
            type="button"
            onClick={() => loadPreset("canonical_scan")}
            className="px-2 py-0.5 text-[11px] rounded-md font-medium bg-background hover:bg-slate-200 border border-border text-slate-700 transition-colors flex items-center gap-1"
            title="Load CT-ORG 3D Scan"
          >
            <Layers className="w-3 h-3 text-slate-600" />
            <span>3D Scan</span>
          </button>
        </div>
      </div>
    </div>
  );
}
