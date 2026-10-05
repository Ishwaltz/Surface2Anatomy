"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Sparkles, Menu, X, Cpu, Github, Package, ExternalLink } from "lucide-react";

export function Header() {
  const pathname = usePathname();
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL || "https://sharonmelhi-surface2anatomy-backend.hf.space";
    fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(5000) })
      .then((res) => {
        if (res.ok) setApiOnline(true);
        else setApiOnline(false);
      })
      .catch(() => setApiOnline(false));
  }, []);

  // Close mobile drawer on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [pathname]);

  const navLinks = [
    { name: "3D Demo", href: "/demo" },
    { name: "Targets (121)", href: "/targets" },
    { name: "Methodology", href: "/method" },
    { name: "Validation", href: "/validation" },
    { name: "Team", href: "/team" },
  ];

  return (
    <header className="w-full bg-white/95 backdrop-blur-md border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
        {/* Left: Brand Identity & CAIR IIT Mandi Crest */}
        <div className="flex items-center gap-3 shrink-0">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="relative w-8 h-8 rounded-lg overflow-hidden border border-slate-200 bg-white flex items-center justify-center shadow-xs">
              <Image
                src="/cair_logo.png"
                alt="CAIR IIT Mandi"
                width={32}
                height={32}
                className="object-contain p-0.5"
                priority
              />
            </div>
            <span className="font-bold text-base md:text-lg text-slate-900 tracking-tight group-hover:text-primary transition-colors">
              Surface2Anatomy
            </span>
          </Link>
          <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            <span className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
            <span>CAIR &bull; IIT Mandi</span>
          </span>
        </div>

        {/* Center: Desktop Nav Links */}
        <nav className="hidden md:flex items-center gap-1 shrink-0">
          {navLinks.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.name}
                href={link.href}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  active
                    ? "bg-primary text-white font-semibold shadow-xs"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/70"
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </nav>

        {/* Right: External Resource Links, Status & Demo CTA */}
        <div className="flex items-center gap-2 shrink-0">
          {/* External Links: PyPI, Hugging Face, GitHub */}
          <div className="hidden lg:flex items-center gap-1.5 border-r border-slate-200 pr-2">
            <a
              href="https://pypi.org/project/surface2anatomy/"
              target="_blank"
              rel="noopener noreferrer"
              title="PyPI Package: pip install surface2anatomy"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-mono font-medium text-slate-700 hover:text-primary hover:bg-slate-100 border border-slate-200 transition-colors"
            >
              <Package className="w-3.5 h-3.5 text-primary" />
              <span>PyPI</span>
            </a>
            <a
              href="https://huggingface.co/spaces/Sharon-codes/surface2anatomy-backend"
              target="_blank"
              rel="noopener noreferrer"
              title="Hugging Face Space Backend"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-mono font-medium text-slate-700 hover:text-amber-700 hover:bg-amber-50 border border-slate-200 transition-colors"
            >
              <span>🤗</span>
              <span>HF Space</span>
            </a>
            <a
              href="https://github.com/Sharon-codes/S2A-Net-IIT-Mandi"
              target="_blank"
              rel="noopener noreferrer"
              title="GitHub Repository"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-mono font-medium text-slate-700 hover:text-slate-950 hover:bg-slate-100 border border-slate-200 transition-colors"
            >
              <Github className="w-3.5 h-3.5 text-slate-800" />
              <span>GitHub</span>
            </a>
          </div>

          <div className="hidden xl:flex items-center gap-1.5 text-[11px] font-mono px-2 py-1 rounded-full border border-slate-200 bg-slate-50 text-slate-600">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline === true
                  ? "bg-emerald-500 animate-pulse"
                  : apiOnline === false
                  ? "bg-amber-500"
                  : "bg-slate-400"
              }`}
            />
            <span>{apiOnline === true ? "GPU Online" : apiOnline === false ? "Local Standby" : "Checking..."}</span>
          </div>

          <Link
            href="/demo"
            className="hidden sm:inline-flex px-3 py-1.5 rounded-lg bg-primary hover:bg-primary-dark text-white text-xs font-semibold shadow-xs transition-all items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Launch Demo</span>
          </Link>

          {/* Mobile Hamburger Button */}
          <button
            type="button"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-xl text-slate-700 hover:bg-slate-100 border border-slate-200 transition-colors focus:outline-none"
            aria-label="Toggle Navigation Menu"
          >
            {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {isMobileMenuOpen && (
        <div className="md:hidden border-t border-slate-200 bg-white/98 backdrop-blur-xl px-4 py-4 flex flex-col gap-2 shadow-lg animate-in slide-in-from-top-2 duration-200">
          <div className="flex items-center justify-between pb-2 border-b border-slate-100 text-xs text-slate-500">
            <div className="flex items-center gap-1.5 font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiOnline === true ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              <span>{apiOnline === true ? "GPU Inference Online" : "Local Standby Mode"}</span>
            </div>
            <span className="font-mono text-[10px]">CAIR IIT Mandi</span>
          </div>

          <nav className="flex flex-col gap-1 my-1">
            {navLinks.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center justify-between ${
                    active
                      ? "bg-primary text-white font-semibold shadow-xs"
                      : "text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  <span>{link.name}</span>
                  {active && <span className="w-1.5 h-1.5 rounded-full bg-primary" />}
                </Link>
              );
            })}
          </nav>

          {/* Mobile Links for PyPI, HF, GitHub */}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-100">
            <a
              href="https://pypi.org/project/surface2anatomy/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center justify-center p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-700 transition-colors"
            >
              <Package className="w-4 h-4 text-primary mb-1" />
              <span>PyPI</span>
            </a>
            <a
              href="https://huggingface.co/spaces/Sharon-codes/surface2anatomy-backend"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center justify-center p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-700 transition-colors"
            >
              <span className="text-base leading-none mb-1">🤗</span>
              <span>HF Space</span>
            </a>
            <a
              href="https://github.com/Sharon-codes/S2A-Net-IIT-Mandi"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center justify-center p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-700 transition-colors"
            >
              <Github className="w-4 h-4 text-slate-800 mb-1" />
              <span>GitHub</span>
            </a>
          </div>

          <Link
            href="/demo"
            className="w-full py-2.5 rounded-xl bg-primary hover:bg-primary-dark text-white text-sm font-semibold shadow-xs transition-all flex items-center justify-center gap-2 mt-2"
          >
            <Sparkles className="w-4 h-4" />
            <span>Launch 3D Localization Demo</span>
          </Link>
        </div>
      )}
    </header>
  );
}
