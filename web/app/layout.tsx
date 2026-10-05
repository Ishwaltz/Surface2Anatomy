import type { Metadata } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { Header } from "@/components/Header";

export const metadata: Metadata = {
  title: "Surface2Anatomy | 3D Internal Anatomy Localization",
  description: "Real-time 3D localization of 104 internal anatomical targets directly from external patient body geometry using deep cross-attention transformer decoders.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="flex flex-col min-h-screen bg-background text-text-main antialiased selection:bg-primary/20 selection:text-primary-dark">
        <Header />
        <main className="flex-grow">{children}</main>
        <footer className="border-t border-border bg-white py-10 mt-16 text-sm text-text-muted">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col gap-8">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              {/* Col 1: Brand & Affiliation */}
              <div className="flex flex-col gap-2.5 md:col-span-1">
                <div className="flex items-center gap-2 font-bold text-base text-text-main">
                  <span>Surface2Anatomy</span>
                </div>
                <p className="text-xs text-text-muted leading-relaxed">
                  Target-conditioned 3D internal anatomy localization from external body surface geometry with zero ionizing radiation.
                </p>
                <div className="text-[11px] font-mono text-primary-dark pt-1">
                  Centre for Artificial Intelligence and Robotics (CAIR)<br />
                  Indian Institute of Technology Mandi (IIT Mandi)
                </div>
              </div>

              {/* Col 2: Open Source Ecosystem */}
              <div className="flex flex-col gap-2">
                <span className="font-semibold text-xs uppercase tracking-wider text-text-main font-mono">
                  Open Source Code &amp; Models
                </span>
                <ul className="flex flex-col gap-1.5 text-xs">
                  <li>
                    <a
                      href="https://pypi.org/project/surface2anatomy/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-primary transition-colors flex items-center gap-1.5"
                    >
                      <span className="font-mono text-primary font-semibold">PyPI:</span> surface2anatomy v0.1.1
                    </a>
                  </li>
                  <li>
                    <a
                      href="https://github.com/Sharon-codes/S2A-Net-IIT-Mandi"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-primary transition-colors flex items-center gap-1.5"
                    >
                      <span className="font-mono text-slate-800 font-semibold">GitHub:</span> S2A-Net-IIT-Mandi
                    </a>
                  </li>
                  <li>
                    <a
                      href="https://huggingface.co/spaces/Sharon-codes/surface2anatomy-backend"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-primary transition-colors flex items-center gap-1.5"
                    >
                      <span>🤗</span>
                      <span>Hugging Face Space Backend</span>
                    </a>
                  </li>
                  <li>
                    <a
                      href="https://huggingface.co/IITMandiResearch/Surface2Anatomy-Weights"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-primary transition-colors flex items-center gap-1.5"
                    >
                      <span>🤗</span>
                      <span>CAIR Hugging Face Weights</span>
                    </a>
                  </li>
                </ul>
              </div>

              {/* Col 3: Research Team */}
              <div className="flex flex-col gap-2">
                <span className="font-semibold text-xs uppercase tracking-wider text-text-main font-mono">
                  Research Team &amp; Contact
                </span>
                <ul className="flex flex-col gap-2 text-xs">
                  <li>
                    <div className="font-medium text-text-main">Khushi Mhamane <span className="text-[10px] font-mono text-primary uppercase font-bold">(Project Lead)</span></div>
                    <a href="mailto:khushimhamane@gmail.com" className="text-accent-green hover:underline font-mono text-[11px]">
                      khushimhamane@gmail.com
                    </a>
                  </li>
                  <li>
                    <div className="font-medium text-text-main">Sharon Melhi <span className="text-[10px] font-mono text-text-muted">(Research Intern)</span></div>
                    <a href="mailto:sharonmelhi365@gmail.com" className="text-accent-green hover:underline font-mono text-[11px]">
                      sharonmelhi365@gmail.com
                    </a>
                  </li>
                  <li>
                    <div className="font-medium text-text-main">Dr. Deepak Raina <span className="text-[10px] font-mono text-text-muted">(Supervisor)</span></div>
                    <a href="mailto:deepakraina@iitmandi.ac.in" className="text-accent-green hover:underline font-mono text-[11px]">
                      deepakraina@iitmandi.ac.in
                    </a>
                  </li>
                </ul>
              </div>

              {/* Col 4: Quick Pip Install */}
              <div className="flex flex-col gap-2">
                <span className="font-semibold text-xs uppercase tracking-wider text-text-main font-mono">
                  Install Python Package
                </span>
                <div className="p-3 bg-slate-900 text-slate-100 rounded-xl font-mono text-xs shadow-inner">
                  <div className="text-slate-400 text-[10px] select-none mb-1"># Official PyPI Release</div>
                  <div className="text-emerald-400 font-semibold select-all">$ pip install surface2anatomy</div>
                </div>
                <p className="text-[11px] text-text-muted leading-relaxed mt-1">
                  Install our frozen PointNet++ and cross-attention multi-seed inference pipeline locally.
                </p>
              </div>
            </div>

            {/* Bottom Row */}
            <div className="pt-6 border-t border-border flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-text-muted">
              <div>
                &copy; 2026 Centre for Artificial Intelligence and Robotics (CAIR), IIT Mandi. Released under Apache-2.0.
              </div>
              <div className="text-center md:text-right max-w-xl text-[11px]">
                Universal Medical Disclaimer: Surface2Anatomy is an academic scientific research prototype. It is not FDA/CE cleared for standalone clinical diagnosis, surgical navigation, or procedural execution.
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
