"use client";

import { useEffect, useState } from "react";
import { 
  GitCompare, ShieldAlert, CheckCircle, 
  AlertTriangle, Fingerprint, RefreshCw, BarChart2
} from "lucide-react";

interface MediaListItem {
  id: number;
  filename: string;
  mime_type: string;
  sha256: string;
  phash: string;
  resolution: string;
  file_size: number;
  risk_score: number;
  integrity_score: number;
}

interface CompareResult {
  source_file: string;
  target_file: string;
  sha256_match: boolean;
  phash_distance: number;
  dhash_distance: number;
  ahash_distance: number;
  visual_similarity: number;
  audio_similarity: number;
  semantic_similarity: number;
  confidence: number;
  relationship_type: string;
  explanation: string;
  source_sha256: string;
  target_sha256: string;
  source_phash: string;
  target_phash: string;
  source_dhash: string;
  target_dhash: string;
  source_ahash: string;
  target_ahash: string;
}

export default function CompareDNAPage() {
  const [mediaList, setMediaList] = useState<MediaListItem[]>([]);
  const [id1, setId1] = useState<string>("");
  const [id2, setId2] = useState<string>("");
  const [loading, setLoading] = useState(true);
  
  const [comparing, setComparing] = useState(false);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [errorText, setErrorText] = useState("");

  const backendUrl = "http://127.0.0.1:8000";

  useEffect(() => {
    fetch(`${backendUrl}/api/media`)
      .then(res => res.json())
      .then(data => {
        setMediaList(data);
        if (data.length > 1) {
          setId1(data[0].id.toString());
          setId2(data[1].id.toString());
        }
      })
      .catch(err => console.error("Error loading media:", err))
      .finally(() => setLoading(false));
  }, []);

  const runComparison = async () => {
    if (!id1 || !id2) return;
    if (id1 === id2) {
      setErrorText("DNA profiles must belong to two distinct assets.");
      return;
    }

    setComparing(true);
    setErrorText("");
    setResult(null);

    try {
      const res = await fetch(`${backendUrl}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_id: parseInt(id1), target_id: parseInt(id2) })
      });

      if (!res.ok) {
        throw new Error("DNA comparison analysis failed.");
      }

      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setErrorText(e.message || "Could not complete DNA comparison.");
    } finally {
      setComparing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent border-[#00E5FF]"></div>
        <span className="font-mono text-xs text-gray-500 tracking-widest font-bold">COMPILING COMPILATION MATRICES...</span>
      </div>
    );
  }

  // Render the 64 bits grid (8x8) of pHash comparison
  const renderBitComparisonGrid = (h1: string, h2: string) => {
    if (!h1 || !h2) return null;
    try {
      const b1 = binFromHex(h1);
      const b2 = binFromHex(h2);
      
      const grid = [];
      for (let i = 0; i < 64; i++) {
        const matches = b1[i] === b2[i];
        grid.push(
          <div 
            key={i}
            className={`h-5 w-5 rounded-sm border border-[#0A0A0A] flex items-center justify-center font-mono text-[8px] font-bold ${
              matches 
                ? "bg-[#00FF9D]/20 text-[#00FF9D] shadow-[0_0_4px_rgba(0,255,157,0.3)]" 
                : "bg-[#FF3366]/20 text-[#FF3366] shadow-[0_0_4px_rgba(255,51,102,0.3)]"
            }`}
            title={`Bit #${i+1}: File A = ${b1[i]}, File B = ${b2[i]} (${matches ? 'Match' : 'Mismatch'})`}
          >
            {b1[i]}
          </div>
        );
      }
      return (
        <div className="grid grid-cols-8 gap-1 p-2 bg-[#121212] border border-[rgba(255,255,255,0.05)] rounded w-fit mx-auto">
          {grid}
        </div>
      );
    } catch (e) {
      return <span className="text-xs text-red-500">Hash parse error</span>;
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title */}
      <div>
        <h1 className="font-mono text-2xl font-black text-white tracking-wider uppercase">
          DNA COMPARISON PANEL
        </h1>
        <p className="mt-1 text-xs text-gray-500 tracking-wide">
          Evaluate visual, audio, and semantic signatures side-by-side to track asset drift.
        </p>
      </div>

      {/* Target Selector Card */}
      <div className="cyber-card p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 items-end">
          {/* Target A Dropdown */}
          <div className="lg:col-span-2">
            <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
              Select Primary Asset (File A)
            </label>
            <select
              value={id1}
              onChange={(e) => setId1(e.target.value)}
              className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2.5 font-mono text-xs text-white focus:border-[#00E5FF] focus:outline-none"
            >
              {mediaList.map((m) => (
                <option key={m.id} value={m.id.toString()}>
                  [ID #{m.id}] {m.filename}
                </option>
              ))}
            </select>
          </div>

          <div className="flex justify-center pb-2 lg:col-span-1">
            <GitCompare className="h-5 w-5 text-gray-600 animate-pulse" />
          </div>

          {/* Target B Dropdown */}
          <div className="lg:col-span-2">
            <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
              Select Target Variant (File B)
            </label>
            <select
              value={id2}
              onChange={(e) => setId2(e.target.value)}
              className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2.5 font-mono text-xs text-white focus:border-[#00E5FF] focus:outline-none"
            >
              {mediaList.map((m) => (
                <option key={m.id} value={m.id.toString()}>
                  [ID #{m.id}] {m.filename}
                </option>
              ))}
            </select>
          </div>
        </div>

        {errorText && (
          <div className="mt-4 flex gap-2 items-center rounded border border-[rgba(255,51,102,0.2)] bg-[rgba(255,51,102,0.04)] p-3 text-[#FF3366] text-xs font-mono">
            <AlertTriangle className="h-4 w-4" />
            <span>{errorText}</span>
          </div>
        )}

        <button
          onClick={runComparison}
          disabled={comparing}
          className="cyber-button w-full mt-6 py-2.5 font-mono text-xs font-bold tracking-wider"
        >
          {comparing ? "CONDUCTING ANALYSIS..." : "EXECUTE COMPARE SIGNAL"}
        </button>
      </div>

      {/* Comparison Results Area */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-scaleUp">
          {/* Column 1 & 2: Main Similarity Summary */}
          <div className="lg:col-span-2 space-y-6">
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex justify-between items-center">
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Explainable Similarity Log
                </h2>
                <span className={`rounded border px-2.5 py-1 font-mono text-[10px] font-bold uppercase tracking-wider ${
                  result.relationship_type === "Original Asset" || result.relationship_type === "Exact Match" ? "bg-[#00FF9D]/10 border-[#00FF9D]/30 text-[#00FF9D] shadow-[0_0_10px_rgba(0,255,157,0.15)]" :
                  result.relationship_type === "Near Duplicate" ? "bg-[#00E5FF]/10 border-[#00E5FF]/30 text-[#00E5FF] shadow-[0_0_10px_rgba(0,229,255,0.15)]" :
                  result.relationship_type === "Cropped Variant" ? "bg-[#7C3AED]/10 border-[#7C3AED]/30 text-[#7C3AED] shadow-[0_0_10px_rgba(124,58,237,0.15)]" :
                  result.relationship_type === "Resized Variant" ? "bg-[#3B82F6]/10 border-[#3B82F6]/30 text-[#3B82F6] shadow-[0_0_10px_rgba(59,130,246,0.15)]" :
                  result.relationship_type === "Compressed Variant" ? "bg-[#00E5FF]/10 border-[#00E5FF]/30 text-[#00E5FF] shadow-[0_0_10px_rgba(0,229,255,0.15)]" :
                  result.relationship_type === "Watermarked Variant" ? "bg-[#F59E0B]/10 border-[#F59E0B]/30 text-[#F59E0B] shadow-[0_0_10px_rgba(245,158,11,0.15)]" :
                  result.relationship_type === "Manipulated Variant" ? "bg-[#EF4444]/10 border-[#EF4444]/30 text-[#EF4444] shadow-[0_0_10px_rgba(239,68,68,0.15)]" :
                  result.relationship_type === "Unknown Baseline Asset" ? "bg-[#9CA3AF]/10 border-[#9CA3AF]/30 text-[#9CA3AF] shadow-[0_0_10px_rgba(156,163,175,0.15)]" :
                  "bg-[#FF3366]/10 border-[#FF3366]/30 text-[#FF3366] shadow-[0_0_10px_rgba(255,51,98,0.15)]"
                }`}>
                  {result.relationship_type}
                </span>
              </div>

              {/* Source vs Target Filenames */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center bg-[#111] p-4 rounded border border-[rgba(255,255,255,0.04)] gap-3 font-mono text-xs">
                <div>
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Primary Source (File A)</span>
                  <span className="text-white font-bold truncate max-w-[250px] block" title={result.source_file}>{result.source_file}</span>
                </div>
                <div className="text-gray-600 hidden md:block font-black">// VS //</div>
                <div>
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Target Variant (File B)</span>
                  <span className="text-white font-bold truncate max-w-[250px] block" title={result.target_file}>{result.target_file}</span>
                </div>
              </div>

              {/* Similarity Score Gauges */}
              <div className="grid grid-cols-3 gap-4 text-center font-mono">
                <div className="bg-[#121212] p-3 rounded border border-[rgba(255,255,255,0.03)]">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Visual Similarity</span>
                  <span className="block mt-1 text-lg font-bold text-white">{Math.round(result.visual_similarity * 100)}%</span>
                </div>
                <div className="bg-[#121212] p-3 rounded border border-[rgba(255,255,255,0.03)]">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Audio Similarity</span>
                  <span className="block mt-1 text-lg font-bold text-white">{Math.round(result.audio_similarity * 100)}%</span>
                </div>
                <div className="bg-[#121212] p-3 rounded border border-[rgba(255,255,255,0.03)]">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Semantic Similarity</span>
                  <span className="block mt-1 text-lg font-bold text-white">{Math.round(result.semantic_similarity * 100)}%</span>
                </div>
              </div>

              {/* Combined Confidence Score Gauge */}
              <div className="bg-gradient-to-r from-[#7C3AED]/20 to-[#00E5FF]/20 p-4 rounded border border-[#00E5FF]/10 flex flex-col items-center justify-center text-center">
                <span className="font-mono text-[9px] text-gray-400 tracking-widest uppercase">Final Confidence Match</span>
                <span className="text-3xl font-black font-mono text-[#00FF9D] glow-text-green mt-1">
                  {Math.round(result.confidence * 100)}%
                </span>
              </div>

              {/* Explainable Text Paragraph */}
              <div className="rounded border border-[rgba(255,255,255,0.04)] bg-[#111] p-4 text-xs leading-relaxed text-gray-300 font-sans">
                <p>
                  <b>Forensic Explanation: </b> {result.explanation}
                </p>
              </div>
            </div>

            {/* Hash Breakdown Section */}
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  DNA Hash Breakdown
                </h2>
              </div>
              <div className="space-y-4 font-mono text-xs">
                {/* SHA256 */}
                <div className="border border-[rgba(255,255,255,0.04)] bg-[#111] rounded p-3">
                  <div className="flex justify-between border-b border-[rgba(255,255,255,0.05)] pb-1.5 mb-1.5">
                    <span className="text-gray-400 font-bold text-[10px]">SHA256 CHECKSUM</span>
                    <span className={result.sha256_match ? "text-[#00FF9D] text-[10px] font-bold" : "text-[#FF3366] text-[10px] font-bold"}>
                      {result.sha256_match ? "CRYPTOGRAPHIC MATCH" : "CHECKSUM MISMATCH"}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE A:</span>
                      <span className="block truncate text-white" title={result.source_sha256}>{result.source_sha256}</span>
                    </div>
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE B:</span>
                      <span className="block truncate text-white" title={result.target_sha256}>{result.target_sha256}</span>
                    </div>
                  </div>
                </div>

                {/* pHash */}
                <div className="border border-[rgba(255,255,255,0.04)] bg-[#111] rounded p-3">
                  <div className="flex justify-between border-b border-[rgba(255,255,255,0.05)] pb-1.5 mb-1.5">
                    <span className="text-gray-400 font-bold text-[10px]">PERCEPTUAL HASH (pHash)</span>
                    <span className="text-[#00E5FF] text-[10px] font-bold">Distance: {result.phash_distance} / 64 bits</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE A:</span>
                      <span className="block text-white font-bold">{result.source_phash || "N/A"}</span>
                    </div>
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE B:</span>
                      <span className="block text-white font-bold">{result.target_phash || "N/A"}</span>
                    </div>
                  </div>
                </div>

                {/* dHash */}
                <div className="border border-[rgba(255,255,255,0.04)] bg-[#111] rounded p-3">
                  <div className="flex justify-between border-b border-[rgba(255,255,255,0.05)] pb-1.5 mb-1.5">
                    <span className="text-gray-400 font-bold text-[10px]">DIFFERENCE HASH (dHash)</span>
                    <span className="text-[#00E5FF] text-[10px] font-bold">Distance: {result.dhash_distance} / 64 bits</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE A:</span>
                      <span className="block text-white font-bold">{result.source_dhash || "N/A"}</span>
                    </div>
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE B:</span>
                      <span className="block text-white font-bold">{result.target_dhash || "N/A"}</span>
                    </div>
                  </div>
                </div>

                {/* aHash */}
                <div className="border border-[rgba(255,255,255,0.04)] bg-[#111] rounded p-3">
                  <div className="flex justify-between border-b border-[rgba(255,255,255,0.05)] pb-1.5 mb-1.5">
                    <span className="text-gray-400 font-bold text-[10px]">AVERAGE HASH (aHash)</span>
                    <span className="text-[#00E5FF] text-[10px] font-bold">Distance: {result.ahash_distance} / 64 bits</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE A:</span>
                      <span className="block text-white font-bold">{result.source_ahash || "N/A"}</span>
                    </div>
                    <div>
                      <span className="block text-[8px] text-gray-500">FILE B:</span>
                      <span className="block text-white font-bold">{result.target_ahash || "N/A"}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Column 3: pHash Bit-by-bit comparison */}
          <div className="cyber-card p-6 flex flex-col gap-5 lg:col-span-1">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 text-center">
              <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                Perceptual Hash Bit-Grid Map
              </h2>
              <span className="block text-[8px] text-gray-500 tracking-wider font-mono mt-0.5">
                Red indicates altered pixels / frequency coefficients.
              </span>
            </div>

            {/* Render Grid */}
            {renderBitComparisonGrid(result.source_phash, result.target_phash)}

            {/* Details panel */}
            <div className="border-t border-[rgba(255,255,255,0.05)] pt-4 space-y-3 font-mono text-[10px] text-gray-400">
              <div className="flex justify-between">
                <span>File A pHash:</span>
                <span className="text-white font-bold">{result.source_phash || "N/A"}</span>
              </div>
              <div className="flex justify-between">
                <span>File B pHash:</span>
                <span className="text-white font-bold">{result.target_phash || "N/A"}</span>
              </div>
              <div className="flex justify-between items-center bg-[#111] px-2 py-1.5 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="text-xs text-gray-500">Hamming Distance:</span>
                <span className="text-xs text-[#00E5FF] font-black">
                  {hamming(result.source_phash, result.target_phash)} / 64 bits
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function binFromHex(hex: string): string {
  if (!hex) return "";
  let out = "";
  for (let i = 0; i < hex.length; i++) {
    const bin = parseInt(hex[i], 16).toString(2).padStart(4, "0");
    out += bin;
  }
  return out;
}

function hamming(h1: string, h2: string): number {
  if (!h1 || !h2) return 64;
  let b1 = binFromHex(h1);
  let b2 = binFromHex(h2);
  let count = 0;
  for (let i = 0; i < 64; i++) {
    if (b1[i] !== b2[i]) count++;
  }
  return count;
}
