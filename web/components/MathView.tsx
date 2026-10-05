import React from "react";
import katex from "katex";

interface MathViewProps {
  math: string;
  block?: boolean;
  className?: string;
}

export function MathView({ math, block = true, className = "" }: MathViewProps) {
  const html = katex.renderToString(math, {
    displayMode: block,
    throwOnError: false,
  });

  if (!block) {
    return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
  }

  return (
    <div
      className={`my-3 overflow-x-auto py-3 px-4 rounded-xl bg-[#F8F9F5] text-[#1F241B] shadow-2xs border border-[#D8DCCF] flex items-center justify-center font-sans ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
