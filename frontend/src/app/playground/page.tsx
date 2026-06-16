"use client";

import { useEffect, useState } from "react";
import { 
  Sliders, Settings, RefreshCcw, Eye, 
  HelpCircle, AlertCircle, Fingerprint, Award 
} from "lucide-react";

interface MediaListItem {
  id: number;
  filename: string;
  mime_type: string;
  phash: string;
}

interface PlaygroundResult {
  phash: string;
  dhash: string;
  ahash: string;
  integrity_score: number;
  visual_diff: number[];
  image_base64: string;
  explanation: string;
}

export default function PlaygroundPage() {
  const [mediaList, setMediaList] = useState<MediaListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [loading, setLoading] = useState(true);

  // Sliders State
  const [cropPct, setCropPct] = useState(0);
  const [watermarkOpacity, setWatermarkOpacity] = useState(0);
  const [compressQuality, setCompressQuality] = useState(100);
  const [resizeScale, setResizeScale] = useState(100);

  const [simulating, setSimulating] = useState(false);
  const [result, setResult] = useState<PlaygroundResult | null>(null);
  const [errorText, setErrorText] = useState("");

  const backendUrl = "http://127.0.0.1:8000";

  useEffect(() => {
    // Load image files only (videos do not support live Pillow manipulation in browser sandbox)
    fetch(`${backendUrl}/api/media`)
      .then(res => res.json())
      .then(data => {
        const imagesOnly = data.filter((m: any) => m.mime_type.startsWith("image/"));
        setMediaList(imagesOnly);
        if (imagesOnly.length > 0) {
          setSelectedId(imagesOnly[0].id.toString());
        }
      })
      .catch(err => console.error("Error loading images:", err))
      .finally(() => setLoading(false));
  }, []);

  const triggerSimulation = async () => {
    if (!selectedId) return;
    setSimulating(true);
    setErrorText("");

    try {
      const res = await fetch(`${backendUrl}/api/playground/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          media_id: parseInt(selectedId),
          crop_pct: cropPct,
          watermark_opacity: watermarkOpacity,
          compress_quality: compressQuality,
          resize_scale: resizeScale
        })
      });

      if (!res.ok) {
        throw new Error("Sandbox simulation failed.");
      }

      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setErrorText(e.message || "Failed to generate fingerprint sandbox simulation.");
    } finally {
      setSimulating(false);
    }
  };

  const getOriginalHash = () => {
    const item = mediaList.find(m => m.id.toString() === selectedId);
    return item ? item.phash : "";
  };

  // Render binary bit grid showing differences
  const renderInteractiveBits = (diffs: number[], newPh: string) => {
    const origPh = getOriginalHash();
    if (!origPh || !newPh) return null;
    
    const b1 = binFromHex(origPh);
    const b2 = binFromHex(newPh);

    const blocks = [];
    for (let i = 0; i < 64; i++) {
      const changed = diffs.includes(i);
      blocks.push(
        <div 
          key={i}
          className={`h-5 w-5 rounded-sm border border-[#0A0A0A] flex items-center justify-center font-mono text-[8px] font-bold ${
            changed 
              ? "bg-[#FF3366] text-white shadow-[0_0_8px_rgba(255,51,102,0.6)] animate-pulse" 
              : "bg-[#00FF9D]/20 text-[#00FF9D] shadow-[0_0_4px_rgba(0,255,157,0.3)]"
          }`}
          title={`Bit #${i+1}: Original = ${b1[i]}, Modified = ${b2[i]} (${changed ? 'Changed' : 'Unchanged'})`}
        >
          {b2[i]}
        </div>
      );
    }

    return (
      <div className="grid grid-cols-8 gap-1 p-2 bg-[#121212] border border-[rgba(255,255,255,0.05)] rounded w-fit mx-auto">
        {blocks}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent border-[#00E5FF]"></div>
        <span className="font-mono text-xs text-gray-500 tracking-widest font-bold">TUNING SIGNAL OSCILLATORS...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title */}
      <div>
        <h1 className="font-mono text-2xl font-black text-white tracking-wider uppercase">
          Fingerprint Sandbox Playground
        </h1>
        <p className="mt-1 text-xs text-gray-500 tracking-wide">
          Interactively simulate modifications and observe real-time bit divergence in perceptual hashes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Sliders & Controls */}
        <div className="cyber-card p-6 flex flex-col gap-6 lg:col-span-1 h-fit">
          <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
            <Sliders className="h-4 w-4 text-[#00E5FF]" />
            <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
              Control Parameters
            </h2>
          </div>

          {/* Asset Select */}
          <div>
            <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
              Select Seed Image
            </label>
            <select
              value={selectedId}
              onChange={(e) => {
                setSelectedId(e.target.value);
                setResult(null);
              }}
              className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2.5 font-mono text-xs text-white focus:border-[#00E5FF] focus:outline-none"
            >
              {mediaList.map((m) => (
                <option key={m.id} value={m.id.toString()}>
                  {m.filename}
                </option>
              ))}
            </select>
          </div>

          {/* Sliders Area */}
          <div className="space-y-4 font-mono text-xs">
            {/* 1. Crop Slider */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] uppercase text-gray-400">
                <span>Crop Width</span>
                <span className="text-[#00E5FF] font-bold">{cropPct}%</span>
              </div>
              <input 
                type="range" min="0" max="80" step="5"
                value={cropPct} onChange={(e) => setCropPct(parseInt(e.target.value))}
                className="w-full h-1 bg-[#1A1A1A] rounded-lg appearance-none cursor-pointer accent-[#00E5FF]"
              />
            </div>

            {/* 2. Watermark Opacity */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] uppercase text-gray-400">
                <span>Watermark Intensity</span>
                <span className="text-[#00E5FF] font-bold">{watermarkOpacity}%</span>
              </div>
              <input 
                type="range" min="0" max="100" step="10"
                value={watermarkOpacity} onChange={(e) => setWatermarkOpacity(parseInt(e.target.value))}
                className="w-full h-1 bg-[#1A1A1A] rounded-lg appearance-none cursor-pointer accent-[#00E5FF]"
              />
            </div>

            {/* 3. Compression Quality */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] uppercase text-gray-400">
                <span>JPEG Compress Quality</span>
                <span className="text-[#00E5FF] font-bold">{compressQuality}%</span>
              </div>
              <input 
                type="range" min="5" max="100" step="5"
                value={compressQuality} onChange={(e) => setCompressQuality(parseInt(e.target.value))}
                className="w-full h-1 bg-[#1A1A1A] rounded-lg appearance-none cursor-pointer accent-[#00E5FF]"
              />
            </div>

            {/* 4. Resize Scale */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] uppercase text-gray-400">
                <span>Resize Scaling</span>
                <span className="text-[#00E5FF] font-bold">{resizeScale}%</span>
              </div>
              <input 
                type="range" min="10" max="200" step="10"
                value={resizeScale} onChange={(e) => setResizeScale(parseInt(e.target.value))}
                className="w-full h-1 bg-[#1A1A1A] rounded-lg appearance-none cursor-pointer accent-[#00E5FF]"
              />
            </div>
          </div>

          {errorText && (
            <div className="flex gap-2 items-center rounded border border-[rgba(255,51,102,0.2)] bg-[rgba(255,51,102,0.04)] p-3 text-[#FF3366] text-xs font-mono">
              <AlertCircle className="h-4 w-4" />
              <span>{errorText}</span>
            </div>
          )}

          <button
            onClick={triggerSimulation}
            disabled={simulating || !selectedId}
            className="cyber-button w-full py-2.5 font-mono text-xs font-bold tracking-wider"
          >
            {simulating ? "SIMULATING DRIFT..." : "APPLY DIGITAL DISTORTION"}
          </button>
        </div>

        {/* Right Column: Visualization & Results */}
        <div className="cyber-card p-6 flex flex-col gap-6 lg:col-span-2">
          <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
            <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
              Fingerprint Drift Output
            </h2>
          </div>

          {!result ? (
            <div className="flex flex-col items-center justify-center py-20 text-center text-gray-500 font-mono text-xs">
              <Settings className="h-8 w-8 text-gray-700 animate-spin mb-4" />
              <span>ADJUST THE FILTER PARAMETERS AND RUN SIMULATION TO VISUALIZE RESULTS.</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-scaleUp">
              {/* Left Column: Simulated Image Output */}
              <div className="space-y-4">
                <span className="block font-mono text-[9px] uppercase tracking-widest text-gray-500">
                  Simulated Digital Variant
                </span>
                <div className="relative border border-[rgba(255,255,255,0.06)] bg-[#0C0C0C] rounded overflow-hidden flex items-center justify-center min-h-[200px]">
                  <img 
                    src={result.image_base64} 
                    alt="Simulated result" 
                    className="max-h-[250px] object-contain w-full"
                  />
                </div>

                <div className="flex justify-between items-center rounded border border-[rgba(0,255,157,0.15)] bg-[rgba(0,255,157,0.02)] p-3 font-mono">
                  <span className="text-[10px] text-gray-500">DNA Integrity:</span>
                  <span className="text-sm text-[#00FF9D] font-black glow-text-green">
                    {result.integrity_score} / 100
                  </span>
                </div>
              </div>

              {/* Right Column: Bit Divergence Map */}
              <div className="space-y-4">
                <span className="block font-mono text-[9px] uppercase tracking-widest text-gray-500 text-center">
                  Hash Bit Divergence Grid (pHash)
                </span>
                
                {renderInteractiveBits(result.visual_diff, result.phash)}

                <div className="rounded border border-[rgba(255,255,255,0.04)] bg-[#111] p-3 text-[10px] font-mono text-gray-400 space-y-2">
                  <div className="flex justify-between">
                    <span>Original pHash:</span>
                    <span className="text-white font-bold">{getOriginalHash()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Simulated pHash:</span>
                    <span className="text-white font-bold">{result.phash}</span>
                  </div>
                  <div className="flex justify-between text-[#FF3366] bg-[#FF3366]/5 px-2 py-1 border border-[#FF3366]/10 rounded mt-1">
                    <span>Diverging Bits:</span>
                    <span className="font-black">{result.visual_diff.length} / 64 bits</span>
                  </div>
                </div>
              </div>

              {/* Explanation Text */}
              <div className="md:col-span-2 border-t border-[rgba(255,255,255,0.05)] pt-4">
                <div className="rounded border border-[rgba(255,255,255,0.04)] bg-[#0E0E0E] p-3 text-xs leading-relaxed text-gray-300 font-sans">
                  <p>
                    <b>Simulation Analysis: </b> {result.explanation}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function binFromHex(hex: string): string {
  let out = "";
  for (let i = 0; i < hex.length; i++) {
    const bin = parseInt(hex[i], 16).toString(2).padStart(4, "0");
    out += bin;
  }
  return out;
}
