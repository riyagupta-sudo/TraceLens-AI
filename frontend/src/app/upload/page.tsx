"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { 
  Upload, Shield, FileText, CheckCircle, AlertTriangle, 
  Database, Image as ImageIcon, Video, Fingerprint, Network, Binary
} from "lucide-react";

interface Case {
  id: number;
  name: string;
}

interface PipelineStep {
  id: number;
  label: string;
  sub: string;
  status: "idle" | "running" | "done" | "skipped";
}

function IngestionPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [cases, setCases] = useState<Case[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Pipeline Processing State
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([
    { id: 1, label: "Ingestion", sub: "Load raw bytes", status: "idle" },
    { id: 2, label: "SHA256", sub: "Cryptographic signature", status: "idle" },
    { id: 3, label: "Perceptual Hashing", sub: "pHash / dHash / aHash", status: "idle" },
    { id: 4, label: "Audio Profile", sub: "Spectral Peaks & Chroma", status: "idle" },
    { id: 5, label: "CLIP Semantics", sub: "HuggingFace Vision Vector", status: "idle" },
    { id: 6, label: "Cross Match", sub: "Similarity Index Search", status: "idle" },
    { id: 7, label: "Forensic Diagnostics", sub: "Check integrity anomalies", status: "idle" },
    { id: 8, label: "Lineage Graph", sub: "Link variant trees", status: "idle" },
  ]);

  const [finishedItem, setFinishedItem] = useState<any>(null);
  const [errorText, setErrorText] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const backendUrl = "http://127.0.0.1:8000";

  useEffect(() => {
    // Grab cases list
    fetch(`${backendUrl}/api/cases`)
      .then(res => res.json())
      .then(data => {
        setCases(data);
        // Pre-select case if passed in query param
        const queryCase = searchParams.get("caseId");
        if (queryCase) {
          setSelectedCaseId(queryCase);
        } else if (data.length > 0) {
          setSelectedCaseId(data[0].id.toString());
        }
      })
      .catch(err => console.error("Error loading cases:", err));
  }, [searchParams]);

  // Drag and drop event handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const startUploadPipeline = async () => {
    if (!file || !selectedCaseId) return;
    setProcessing(true);
    setErrorText("");
    setFinishedItem(null);
    setProgress(5);

    const isVideo = file.type.startsWith("video/");

    // Step 1: Ingestion
    updateStep(1, "running");
    await sleep(600);
    updateStep(1, "done");
    setProgress(15);

    // Step 2: Cryptographic Signature
    updateStep(2, "running");
    await sleep(650);
    updateStep(2, "done");
    setProgress(30);

    // Step 3: Perceptual Hashing
    updateStep(3, "running");
    await sleep(700);
    updateStep(3, "done");
    setProgress(45);

    // Step 4: Audio Hashing (skipped if image)
    updateStep(4, "running");
    await sleep(600);
    if (!isVideo) {
      updateStep(4, "skipped");
    } else {
      updateStep(4, "done");
    }
    setProgress(60);

    // Step 5: CLIP Semantics
    updateStep(5, "running");
    
    // Now trigger the actual network upload form
    const formData = new FormData();
    formData.append("case_id", selectedCaseId);
    formData.append("file", file);

    try {
      const uploadRes = await fetch(`${backendUrl}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        let errMsg = "Upload failed. Verify server is running.";
        try {
          const errData = await uploadRes.json();
          if (errData && errData.error) {
            errMsg = `[${errData.stage || "Pipeline"}] ${errData.error}`;
          }
        } catch (_) {}
        throw new Error(errMsg);
      }

      const mediaItem = await uploadRes.json();
      
      updateStep(5, "done");
      setProgress(75);

      // Step 6: Similarity Cross-Match
      updateStep(6, "running");
      await sleep(600);
      updateStep(6, "done");
      setProgress(85);

      // Step 7: Diagnostics
      updateStep(7, "running");
      await sleep(500);
      updateStep(7, "done");
      setProgress(95);

      // Step 8: Lineage Integration
      updateStep(8, "running");
      await sleep(500);
      updateStep(8, "done");
      setProgress(100);

      setFinishedItem(mediaItem);
    } catch (err: any) {
      setErrorText(err.message || "An error occurred during pipeline analysis.");
      resetPipeline();
    } finally {
      setProcessing(false);
    }
  };

  const updateStep = (id: number, status: "idle" | "running" | "done" | "skipped") => {
    setPipelineSteps(prev => 
      prev.map(step => step.id === id ? { ...step, status } : step)
    );
  };

  const resetPipeline = () => {
    setPipelineSteps(prev => prev.map(step => ({ ...step, status: "idle" })));
    setProgress(0);
  };

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title */}
      <div>
        <h1 className="font-mono text-2xl font-black text-white tracking-wider uppercase">
          Ingestion & Processing Engine
        </h1>
        <p className="mt-1 text-xs text-gray-500 tracking-wide">
          Drag and drop media assets to initiate forensic profiling.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: Upload Controls & Drop Area */}
        <div className="cyber-card p-6 flex flex-col gap-5 lg:col-span-1 h-fit">
          <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
            <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
              Configure Parameters
            </h2>
          </div>

          {/* Select Case Input */}
          <div>
            <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
              Select Investigation Case
            </label>
            <select
              value={selectedCaseId}
              onChange={(e) => setSelectedCaseId(e.target.value)}
              disabled={processing}
              className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2.5 font-mono text-xs text-white focus:border-[#00E5FF] focus:outline-none"
            >
              {cases.map((c) => (
                <option key={c.id} value={c.id.toString()}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          {/* Drop Box Area */}
          <div
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300 ${
              dragActive 
                ? "border-[#00E5FF] bg-[#00E5FF]/5" 
                : "border-[rgba(255,255,255,0.08)] bg-[#111] hover:border-[rgba(0,229,255,0.3)] hover:bg-[#151515]"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              disabled={processing}
              accept="image/*,video/*"
              className="hidden"
            />
            {file ? (
              <div className="space-y-2">
                {file.type.startsWith("video/") ? (
                  <Video className="h-10 w-10 text-[#7C3AED] mx-auto animate-pulse" />
                ) : (
                  <ImageIcon className="h-10 w-10 text-[#00E5FF] mx-auto" />
                )}
                <span className="block font-semibold text-white truncate max-w-[180px] mx-auto text-xs">{file.name}</span>
                <span className="block text-[9px] font-mono text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB // {file.type}</span>
              </div>
            ) : (
              <div className="space-y-3">
                <Upload className="h-10 w-10 text-gray-500 mx-auto" />
                <div>
                  <span className="block text-xs font-semibold text-white">Drag & drop files or click</span>
                  <span className="block text-[10px] text-gray-500 mt-1 font-mono">JPG, PNG, WEBP, MP4, MOV, AVI</span>
                </div>
              </div>
            )}
          </div>

          {file && !processing && !finishedItem && (
            <button
              onClick={startUploadPipeline}
              className="cyber-button w-full py-2.5 font-mono text-xs font-bold tracking-wider"
            >
              INITIATE DNA PIPELINE
            </button>
          )}

          {errorText && (
            <div className="flex gap-2 items-center rounded border border-[rgba(255,51,102,0.2)] bg-[rgba(255,51,102,0.04)] p-3 text-[#FF3366] text-xs font-mono">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{errorText}</span>
            </div>
          )}
        </div>

        {/* Right column: Interactive Processing Pipeline Visualizer */}
        <div className="cyber-card p-6 flex flex-col gap-6 lg:col-span-2">
          <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
            <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
              DNA Extraction Pipeline
            </h2>
          </div>

          {processing && (
            <div className="w-full bg-[#121212] rounded-full h-1.5 overflow-hidden">
              <div 
                className="bg-gradient-to-r from-[#7C3AED] to-[#00E5FF] h-1.5 transition-all duration-500" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          )}

          {/* Stepper Grid Visualizer */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pipelineSteps.map((step) => (
              <div 
                key={step.id}
                className={`relative flex items-center gap-3 p-3 rounded border transition-all duration-300 ${
                  step.status === "running"
                    ? "border-[#00E5FF] bg-[#00E5FF]/5 shadow-[0_0_10px_rgba(0,229,255,0.1)]"
                    : step.status === "done"
                    ? "border-[#00FF9D]/30 bg-[#00FF9D]/2"
                    : step.status === "skipped"
                    ? "border-gray-800 bg-[#111]/30 opacity-50"
                    : "border-[rgba(255,255,255,0.04)] bg-[#111]"
                }`}
              >
                <div className={`flex h-8 w-8 items-center justify-center rounded-lg border font-mono text-xs font-bold ${
                  step.status === "running"
                    ? "border-[#00E5FF] text-[#00E5FF] animate-pulse"
                    : step.status === "done"
                    ? "border-[#00FF9D] text-[#00FF9D]"
                    : step.status === "skipped"
                    ? "border-gray-700 text-gray-500"
                    : "border-gray-800 text-gray-500"
                }`}>
                  {step.status === "done" ? "✓" : step.status === "skipped" ? "—" : step.id}
                </div>
                
                <div>
                  <span className={`block text-xs font-mono font-bold ${step.status === "running" ? "text-[#00E5FF]" : "text-white"}`}>
                    {step.label}
                  </span>
                  <span className="block text-[9px] text-gray-500 font-mono mt-0.5">
                    {step.status === "running" ? "Analyzing frames..." : step.sub}
                  </span>
                </div>

                {step.status === "running" && (
                  <div className="absolute right-3 top-3 h-2 w-2 rounded-full bg-[#00E5FF] animate-ping"></div>
                )}
              </div>
            ))}
          </div>

          {/* Seeding Success Card */}
          {finishedItem && (
            <div className="border border-[#00FF9D]/20 bg-[rgba(0,255,157,0.02)] p-4 rounded-lg space-y-4 animate-scaleUp">
              <div className="flex items-center gap-2 text-[#00FF9D] text-xs font-bold font-mono">
                <CheckCircle className="h-4 w-4" />
                <span>PROFILE COMPLETED SUCCESSFULLY // DNA GENERATED</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[10px] font-mono border-t border-[rgba(255,255,255,0.05)] pt-3 text-gray-400">
                <div>
                  <span className="block text-[8px] uppercase tracking-widest text-gray-600">SHA256 CHECKSUM</span>
                  <span className="block truncate text-white mt-0.5">{finishedItem.sha256}</span>
                </div>
                <div>
                  <span className="block text-[8px] uppercase tracking-widest text-gray-600">Perceptual Hash</span>
                  <span className="block text-white mt-0.5">{finishedItem.phash}</span>
                </div>
                <div>
                  <span className="block text-[8px] uppercase tracking-widest text-gray-600">Integrity Score</span>
                  <span className="block text-[#00FF9D] font-bold mt-0.5">{finishedItem.integrity_score} / 100</span>
                </div>
                <div>
                  <span className="block text-[8px] uppercase tracking-widest text-gray-600">Risk Assessment</span>
                  <span className={`block font-bold mt-0.5 ${finishedItem.risk_score > 50 ? "text-[#FF3366]" : "text-[#00FF9D]"}`}>
                    {finishedItem.risk_score} / 100
                  </span>
                </div>
              </div>

              <button
                onClick={() => router.push(`/media/${finishedItem.id}`)}
                className="w-full flex items-center justify-center gap-2 border border-[#00E5FF] bg-transparent hover:bg-[#00E5FF]/10 text-[#00E5FF] py-2 font-mono text-xs font-semibold rounded transition-colors"
              >
                <Fingerprint className="h-4 w-4" />
                VIEW MEDIA FORENSIC PROFILE
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function IngestionPage() {
  return (
    <Suspense fallback={<div className="font-mono text-xs text-gray-500 p-8">Loading Ingestion Engine...</div>}>
      <IngestionPageContent />
    </Suspense>
  );
}
