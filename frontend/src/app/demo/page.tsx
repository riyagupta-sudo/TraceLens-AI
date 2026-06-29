"use client";

import React, { useState } from "react";
import { 
  FileText, Shield, Cpu, AlertTriangle, CheckCircle, Clock, 
  ArrowRight, Download, Eye, ExternalLink, HardDrive, Check, Play
} from "lucide-react";

interface CaseDetail {
  id: string;
  title: string;
  description: string;
  source: string;
  aiScore: number;
  rfScore: number;
  metadataScore: number;
  screenshotScore: number;
  stegoScore: number;
  casiaScore: number;
  consensusState: string;
  consensusExplanation: string;
  verdictLabel: string;
  verdictColor: string;
  hash: string;
  resolution: string;
  software: string;
  metadataDetails: string[];
}

const SAMPLE_CASES: CaseDetail[] = [
  {
    id: "case-1",
    title: "Case A: WhatsApp Stripped Smartphone Photo",
    description: "Authentic camera capture shared over social media, resulting in stripped EXIF metadata and elevated neural anomalies.",
    source: "WhatsApp Share / Media Transfer",
    aiScore: 85,
    rfScore: 15,
    metadataScore: 15, // Stripped metadata
    screenshotScore: 5,
    stegoScore: 10,
    casiaScore: 12,
    consensusState: "MIXED_SIGNALS",
    consensusExplanation: "The neural detector reports elevated AI-generation indicators, however supporting forensic evidence is insufficient to reach a high-confidence conclusion. This result does NOT indicate authenticity. Additional analyst review is recommended.",
    verdictLabel: "⚠ Neural Detector and Forensic Signals Disagree",
    verdictColor: "text-amber-500 bg-amber-500/10 border-amber-500/30 shadow-[0_0_12px_rgba(245,158,11,0.2)]",
    hash: "3b08e9a2c3a504ef51379c6d36e2f1e29ad3f27de58a8a4f61f7e02e88a914f6",
    resolution: "1280x960",
    software: "WhatsApp Transcode / None",
    metadataDetails: [
      "EXIF header block is completely missing.",
      "Camera Model: None / Unknown.",
      "GPS Position details are stripped."
    ]
  },
  {
    id: "case-2",
    title: "Case B: Midjourney AI Image (Standard)",
    description: "Text-to-image neural generation showcasing clear generative signature artifacts and no physical camera metadata.",
    source: "Midjourney v6 Discord Bot",
    aiScore: 99,
    rfScore: 85,
    metadataScore: 15,
    screenshotScore: 5,
    stegoScore: 12,
    casiaScore: 94,
    consensusState: "HIGH_CONFIDENCE_AI_GENERATED",
    consensusExplanation: "Multiple forensic systems strongly agree that the media is likely AI generated or manipulated.",
    verdictLabel: "HIGH CONFIDENCE AI GENERATED",
    verdictColor: "text-[#FF3366] bg-[#FF3366]/10 border-[#FF3366]/30 animate-pulse shadow-[0_0_12px_rgba(255,51,102,0.3)]",
    hash: "a4c28f1165a25e11df31d34c1b9795ef36e297df38a1a4f62e84d081f9a14d5e",
    resolution: "2048x2048",
    software: "None (Generated Output)",
    metadataDetails: [
      "No EXIF payload exists.",
      "Highly repetitive generative patterns detected in FFT spectrum.",
      "Resizing signature matches standard generative scale."
    ]
  },
  {
    id: "case-3",
    title: "Case C: Genuine iPhone 13 Portrait",
    description: "Unmodified smartphone camera capture containing intact camera metadata profile and clean forensic signals.",
    source: "Physical iPhone 13 Camera",
    aiScore: 12,
    rfScore: 8,
    metadataScore: 95,
    screenshotScore: 0,
    stegoScore: 5,
    casiaScore: 8,
    consensusState: "VERIFIED_AUTHENTIC",
    consensusExplanation: "All forensic engines indicate authentic characteristics and a trusted metadata chain.",
    verdictLabel: "VERIFIED AUTHENTIC",
    verdictColor: "text-[#00FF9D] bg-[#00FF9D]/10 border-[#00FF9D]/30 shadow-[0_0_12px_rgba(0,255,157,0.2)]",
    hash: "f42e58a729e11de9cf318a4f681a97df11b2de218a5a4f71a938c01f44a1420d",
    resolution: "3024x4032",
    software: "Apple iOS 15.4",
    metadataDetails: [
      "Camera Manufacturer: Apple.",
      "Camera Model: iPhone 13.",
      "Creation Timestamp: 2026:05:12 14:24:52.",
      "GPS Coordinates embedded correctly."
    ]
  },
  {
    id: "case-4",
    title: "Case D: GIMP Edited Photograph",
    description: "Authentic camera photograph that has been scaled, cropped, and saved in GIMP software, introducing editing anomalies.",
    source: "Canon EOS 5D / GIMP Editor",
    aiScore: 65,
    rfScore: 55,
    metadataScore: 40,
    screenshotScore: 5,
    stegoScore: 35,
    casiaScore: 78,
    consensusState: "INVESTIGATE_FURTHER",
    consensusExplanation: "Evidence is inconclusive and requires additional analyst review.",
    verdictLabel: "INVESTIGATE FURTHER",
    verdictColor: "text-orange-400 bg-orange-400/10 border-orange-400/30 shadow-[0_0_12px_rgba(249,115,22,0.2)]",
    hash: "82a17df39e23b10ea1b379c6d318eef26ad3c27da25a8a4f728c081e88f14a42",
    resolution: "1920x1080",
    software: "GIMP 2.10.30 (Linux)",
    metadataDetails: [
      "Software tag edited from original camera to GIMP.",
      "Compression table indicates Photoshop/GIMP default save quality (90).",
      "Slight ELA blockiness anomalies on localized pixel boundaries."
    ]
  }
];

export default function DemoMode() {
  const [selectedCase, setSelectedCase] = useState<CaseDetail>(SAMPLE_CASES[0]);
  const [activeStep, setActiveStep] = useState<number>(0);
  const [runningSim, setRunningSim] = useState<boolean>(false);

  const startSimulation = () => {
    setRunningSim(true);
    setActiveStep(0);
    const intervals = [1000, 2000, 3200, 4500, 5800];
    
    intervals.forEach((time, index) => {
      setTimeout(() => {
        setActiveStep(index + 1);
        if (index === intervals.length - 1) {
          setRunningSim(false);
        }
      }, time);
    });
  };

  const getConsensusColorClass = (state: string) => {
    switch(state) {
      case "HIGH_CONFIDENCE_AI_GENERATED": return "text-[#FF3366]";
      case "LIKELY_AI_GENERATED": return "text-red-400";
      case "MIXED_SIGNALS": return "text-amber-500";
      case "INVESTIGATE_FURTHER": return "text-orange-400";
      case "LIKELY_AUTHENTIC": return "text-teal-400";
      case "VERIFIED_AUTHENTIC": return "text-[#00FF9D]";
      default: return "text-gray-400";
    }
  };

  const handleDownloadReport = () => {
    const reportText = `===========================================================
               TRACELENS AI FORENSIC ANALYSIS REPORT
===========================================================
Asset ID:       ${selectedCase.id.toUpperCase()}
Filename:       ${selectedCase.title.split(": ")[1]}
SHA256 Hash:    ${selectedCase.hash}
Resolution:     ${selectedCase.resolution}
MIME Type:      image/jpeg
Editing Tools:  ${selectedCase.software}

-----------------------------------------------------------
1. FORENSIC CONSENSUS ASSESSMENT
-----------------------------------------------------------
Consensus State:  ${selectedCase.consensusState}
Confidence:       ${selectedCase.consensusState.includes("HIGH") ? "VERY HIGH" : "MEDIUM"}
Verdict Label:    ${selectedCase.verdictLabel}

Explanation:
${selectedCase.consensusExplanation}

-----------------------------------------------------------
2. FORENSIC SIGNALS BREAKDOWN
-----------------------------------------------------------
* Calibrated AI Score:    ${selectedCase.aiScore}%
* Random Forest Classifier: ${selectedCase.rfScore}%
* Metadata Trust Score:    ${selectedCase.metadataScore}%
* Steganography Suspicion: ${selectedCase.stegoScore}%
* Screenshot Probability:  ${selectedCase.screenshotScore}%
* CASIA Advisory Signal:   ${selectedCase.casiaScore}% (Advisory Only)

-----------------------------------------------------------
3. TECHNICAL METADATA INSIGHTS
-----------------------------------------------------------
${selectedCase.metadataDetails.map(d => `- ${d}`).join("\n")}

===========================================================
UNCLASSIFIED REPORT - FOR DIGITAL FORENSICS INVESTIGATION ONLY
===========================================================`;

    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `TraceLens_Report_${selectedCase.id}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-[rgba(255,255,255,0.06)] pb-4">
        <h1 className="font-mono text-2xl font-black tracking-wider text-white flex items-center gap-2">
          <Shield className="h-6 w-6 text-[#7C3AED] animate-pulse" />
          ENTERPRISE DEMO MODE
        </h1>
        <p className="font-mono text-[10px] text-gray-500 uppercase tracking-widest mt-1">
          Simulate full DNA pipelines, ingestion triggers, and forensics diagnostics under realistic scenarios.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Side: Case Selectors */}
        <div className="space-y-4">
          <h3 className="font-mono text-xs font-black tracking-widest text-gray-500 uppercase">
            Select Investigation Case
          </h3>

          <div className="space-y-3">
            {SAMPLE_CASES.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  if (!runningSim) {
                    setSelectedCase(item);
                    setActiveStep(5); // Show completed by default
                  }
                }}
                disabled={runningSim}
                className={`w-full text-left cyber-card p-4 transition-all duration-300 ${
                  selectedCase.id === item.id
                    ? "border-[#7C3AED] bg-[#7C3AED]/5 shadow-[0_0_12px_rgba(124,58,237,0.15)]"
                    : "opacity-75 hover:opacity-100"
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono text-[10px] text-gray-500 uppercase font-black tracking-wider">
                    {item.source}
                  </span>
                  <span className={`px-1.5 py-0.5 rounded text-[8px] font-mono font-bold uppercase ${
                    item.consensusState === "MIXED_SIGNALS" ? "bg-amber-500/10 text-amber-500" :
                    item.consensusState === "HIGH_CONFIDENCE_AI_GENERATED" ? "bg-[#FF3366]/10 text-[#FF3366]" :
                    item.consensusState === "VERIFIED_AUTHENTIC" ? "bg-[#00FF9D]/10 text-[#00FF9D]" :
                    "bg-orange-500/10 text-orange-400"
                  }`}>
                    {item.consensusState.replace(/_/g, " ")}
                  </span>
                </div>
                <h4 className="font-mono text-xs font-black text-white mb-1.5">{item.title}</h4>
                <p className="font-mono text-[9px] text-gray-400 leading-relaxed line-clamp-2">{item.description}</p>
              </button>
            ))}
          </div>

          {/* Trigger button */}
          <button
            onClick={startSimulation}
            disabled={runningSim}
            className="w-full font-mono text-xs font-bold tracking-widest bg-gradient-to-r from-[#7C3AED] to-[#00E5FF] hover:opacity-90 disabled:opacity-50 text-white py-3 rounded-md flex items-center justify-center gap-2 shadow-[0_0_10px_rgba(0,229,255,0.2)] transition-all"
          >
            <Play className={`h-4 w-4 ${runningSim ? "animate-spin" : ""}`} />
            {runningSim ? "RUNNING PIPELINE SIMULATION..." : "TRIGGER PIPELINE SIMULATION"}
          </button>
        </div>

        {/* Right Side: Stepper & Reports */}
        <div className="lg:col-span-2 space-y-6">
          {/* Verdict Status Card */}
          <div className="cyber-card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-[rgba(255,255,255,0.06)] pb-3">
              <span className="font-mono text-xs font-black tracking-widest text-white uppercase">
                Active Forensic Verdict
              </span>
              <button
                onClick={handleDownloadReport}
                className="font-mono text-[9px] font-black uppercase text-[#00E5FF] bg-[#00E5FF]/5 border border-[#00E5FF]/20 px-2.5 py-1 rounded hover:bg-[#00E5FF]/10 flex items-center gap-1 transition-colors"
              >
                <Download className="h-3 w-3" />
                Download Report
              </button>
            </div>

            <div className="space-y-4">
              <div className={`p-4 border rounded-md font-mono text-center text-xs font-black tracking-wider ${selectedCase.verdictColor}`}>
                {selectedCase.verdictLabel}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 pt-2">
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-wider mb-1">AI score</span>
                  <span className="block font-mono text-base font-black text-white">{selectedCase.aiScore}%</span>
                </div>
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-wider mb-1">RF prob</span>
                  <span className="block font-mono text-base font-black text-white">{selectedCase.rfScore}%</span>
                </div>
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-wider mb-1">meta trust</span>
                  <span className="block font-mono text-base font-black text-white">{selectedCase.metadataScore}%</span>
                </div>
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-wider mb-1">screenshot</span>
                  <span className="block font-mono text-base font-black text-white">{selectedCase.screenshotScore}%</span>
                </div>
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-wider mb-1">stego</span>
                  <span className="block font-mono text-base font-black text-white">{selectedCase.stegoScore}%</span>
                </div>
                <div className="bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded p-3 text-center relative group">
                  <span className="block font-mono text-[8px] text-gray-400 uppercase tracking-wider mb-1">casia*</span>
                  <span className="block font-mono text-base font-black text-gray-400">{selectedCase.casiaScore}%</span>
                  <div className="absolute hidden group-hover:block bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-gray-950 border border-gray-800 text-[8px] text-gray-400 font-mono rounded shadow-lg text-left z-10">
                    * CASIA is treated as experimental and advisory-only because it produces frequent false positives on authentic smartphone photos.
                  </div>
                </div>
              </div>

              <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 font-mono text-[11px] leading-relaxed text-gray-300">
                <span className="block text-[8px] text-gray-500 uppercase tracking-wider font-bold mb-1">Consensus Explanation</span>
                <p>{selectedCase.consensusExplanation}</p>
              </div>
            </div>
          </div>

          {/* Stepper Timeline */}
          <div className="cyber-card p-6 space-y-4">
            <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase border-b border-[rgba(255,255,255,0.06)] pb-3">
              DNA Pipeline Timeline Stepper
            </h3>

            <div className="relative border-l border-[rgba(255,255,255,0.06)] ml-3 pl-6 space-y-6 pt-2 font-mono text-xs">
              
              {/* Step 1 */}
              <div className={`relative ${activeStep >= 1 ? "opacity-100" : "opacity-40"}`}>
                <div className={`absolute -left-[31px] top-0 h-4.5 w-4.5 rounded-full border-2 flex items-center justify-center ${
                  activeStep >= 1 ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#0A0A0A] border-gray-700"
                }`}>
                  {activeStep >= 1 && <Check className="h-2.5 w-2.5 text-black stroke-[3px]" />}
                </div>
                <h4 className="font-bold text-white mb-1 uppercase tracking-wider flex items-center gap-1.5">
                  Step 1: Media Ingestion & Metadata Extract
                  {activeStep === 0 && runningSim && <Clock className="h-3.5 w-3.5 text-[#00E5FF] animate-spin" />}
                </h4>
                <div className="text-gray-400 text-[10px] space-y-1">
                  <p>Hash check computed: <span className="text-[#00E5FF]">{selectedCase.hash}</span></p>
                  <p>Software Profile: {selectedCase.software}</p>
                  <p>Resolution: {selectedCase.resolution} | Format: JPEG</p>
                </div>
              </div>

              {/* Step 2 */}
              <div className={`relative ${activeStep >= 2 ? "opacity-100" : "opacity-40"}`}>
                <div className={`absolute -left-[31px] top-0 h-4.5 w-4.5 rounded-full border-2 flex items-center justify-center ${
                  activeStep >= 2 ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#0A0A0A] border-gray-700"
                }`}>
                  {activeStep >= 2 && <Check className="h-2.5 w-2.5 text-black stroke-[3px]" />}
                </div>
                <h4 className="font-bold text-white mb-1 uppercase tracking-wider flex items-center gap-1.5">
                  Step 2: Heuristics & Diagnostics
                  {activeStep === 1 && runningSim && <Clock className="h-3.5 w-3.5 text-[#00E5FF] animate-spin" />}
                </h4>
                <div className="text-gray-400 text-[10px] space-y-1">
                  <p>Steganography suspicion evaluated: {selectedCase.stegoScore}%</p>
                  <p>Screenshot criteria check: {selectedCase.screenshotScore >= 25 ? "Match" : "No match"} ({selectedCase.screenshotScore}%)</p>
                  <p>JPEG quantization table quality estimated.</p>
                </div>
              </div>

              {/* Step 3 */}
              <div className={`relative ${activeStep >= 3 ? "opacity-100" : "opacity-40"}`}>
                <div className={`absolute -left-[31px] top-0 h-4.5 w-4.5 rounded-full border-2 flex items-center justify-center ${
                  activeStep >= 3 ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#0A0A0A] border-gray-700"
                }`}>
                  {activeStep >= 3 && <Check className="h-2.5 w-2.5 text-black stroke-[3px]" />}
                </div>
                <h4 className="font-bold text-white mb-1 uppercase tracking-wider flex items-center gap-1.5">
                  Step 3: Calibrated AI Detector Neural Inference
                  {activeStep === 2 && runningSim && <Clock className="h-3.5 w-3.5 text-[#00E5FF] animate-spin" />}
                </h4>
                <div className="text-gray-400 text-[10px] space-y-1">
                  <p>Neural classifier output: {selectedCase.aiScore}% (AI Generation Signature)</p>
                  <p>Calibrational calibration temperature scaling multiplier applied.</p>
                </div>
              </div>

              {/* Step 4 */}
              <div className={`relative ${activeStep >= 4 ? "opacity-100" : "opacity-40"}`}>
                <div className={`absolute -left-[31px] top-0 h-4.5 w-4.5 rounded-full border-2 flex items-center justify-center ${
                  activeStep >= 4 ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#0A0A0A] border-gray-700"
                }`}>
                  {activeStep >= 4 && <Check className="h-2.5 w-2.5 text-black stroke-[3px]" />}
                </div>
                <h4 className="font-bold text-white mb-1 uppercase tracking-wider flex items-center gap-1.5">
                  Step 4: Random Forest Classifier
                  {activeStep === 3 && runningSim && <Clock className="h-3.5 w-3.5 text-[#00E5FF] animate-spin" />}
                </h4>
                <div className="text-gray-400 text-[10px] space-y-1">
                  <p>Random forest ensemble validation probability: {selectedCase.rfScore}%</p>
                  <p>Ensemble weights applied to multi-model decision logic.</p>
                </div>
              </div>

              {/* Step 5 */}
              <div className={`relative ${activeStep >= 5 ? "opacity-100" : "opacity-40"}`}>
                <div className={`absolute -left-[31px] top-0 h-4.5 w-4.5 rounded-full border-2 flex items-center justify-center ${
                  activeStep >= 5 ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#0A0A0A] border-gray-700"
                }`}>
                  {activeStep >= 5 && <Check className="h-2.5 w-2.5 text-black stroke-[3px]" />}
                </div>
                <h4 className="font-bold text-white mb-1 uppercase tracking-wider flex items-center gap-1.5">
                  Step 5: Multi-Signal Consensus Resolver
                  {activeStep === 4 && runningSim && <Clock className="h-3.5 w-3.5 text-[#00E5FF] animate-spin" />}
                </h4>
                <div className="text-gray-400 text-[10px] space-y-1">
                  <p>Consensus State Resolved: <span className={getConsensusColorClass(selectedCase.consensusState)}>{selectedCase.consensusState}</span></p>
                  <p>Pipeline output committed to forensic logs.</p>
                </div>
              </div>

            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
