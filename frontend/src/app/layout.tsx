import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "TraceLens AI | Media DNA & Cross-Platform Intelligence Engine",
  description: "Enterprise cyber intelligence platform for OSINT analysts and cybersecurity researchers to extract, track, and analyze media transformations, lineages, and similarities.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full dark">
      <body className="min-h-full flex flex-col bg-[#0A0A0A] text-[#E5E5E5] antialiased">
        {/* Navigation Header */}
        <header className="sticky top-0 z-40 w-full border-b border-[rgba(255,255,255,0.06)] bg-[#0A0A0A]/80 backdrop-blur-md">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 items-center justify-between">
              {/* Logo Area */}
              <div className="flex items-center gap-3">
                <Link href="/" className="flex items-center gap-2">
                  <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#7C3AED] to-[#00E5FF] shadow-[0_0_10px_rgba(0,229,255,0.3)]">
                    <span className="font-mono text-sm font-bold text-white">TL</span>
                    <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-[#00FF9D] ring-2 ring-[#0A0A0A] animate-pulse"></span>
                  </div>
                  <div>
                    <span className="font-mono text-lg font-black tracking-wider text-white">
                      TRACELENS<span className="text-[#00E5FF]">.AI</span>
                    </span>
                    <span className="block text-[9px] uppercase tracking-widest text-[#555555]">
                      Media DNA Engine
                    </span>
                  </div>
                </Link>
              </div>

              {/* Navigation Links */}
              <nav className="flex items-center gap-6">
                <Link 
                  href="/" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  DASHBOARD
                </Link>
                <Link 
                  href="/upload" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  INGESTION
                </Link>
                <Link 
                  href="/compare" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  COMPARE DNA
                </Link>
                <Link 
                  href="/playground" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  PLAYGROUND
                </Link>
                <Link 
                  href="/evaluation" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  EVALUATION
                </Link>
                <Link 
                  href="/demo" 
                  className="font-mono text-xs font-semibold tracking-wider text-gray-400 transition-colors hover:text-[#00E5FF]"
                >
                  DEMO MODE
                </Link>
              </nav>

              {/* Connection Status badge */}
              <div className="hidden items-center gap-2 rounded-full border border-[rgba(0,255,157,0.15)] bg-[rgba(0,255,157,0.02)] px-3 py-1 sm:flex">
                <span className="h-2 w-2 rounded-full bg-[#00FF9D] shadow-[0_0_8px_rgba(0,255,157,0.6)]"></span>
                <span className="font-mono text-[9px] font-bold tracking-widest text-[#00FF9D]">SYSTEM CONNECTED</span>
              </div>
            </div>
          </div>
        </header>

        {/* Content Wrapper */}
        <main className="flex-1 flex flex-col max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-[rgba(255,255,255,0.04)] bg-[#070707] py-4">
          <div className="mx-auto max-w-7xl px-4 text-center font-mono text-[10px] text-gray-600 sm:px-6 lg:px-8">
            &copy; {new Date().getFullYear()} TRACELENS AI. UNCLASSIFIED OSINT REPORTING INSTRUMENT. ALL HASHE PIPELINES SECURE.
          </div>
        </footer>
      </body>
    </html>
  );
}
