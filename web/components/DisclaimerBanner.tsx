import React from "react";
import { AlertCircle } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="w-full bg-[#EAECE4] border-b border-[#D8DCCF] px-4 py-2 text-xs text-text-muted flex items-center justify-center gap-2 font-medium tracking-wide">
      <AlertCircle className="w-4 h-4 text-primary-dark shrink-0" />
      <span>
        <strong>Research prototype</strong> — not for clinical diagnosis or procedural guidance. All coordinate outputs are experimental estimations.
      </span>
    </div>
  );
}
