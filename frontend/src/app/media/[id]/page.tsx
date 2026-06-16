"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  FileText, Shield, AlertTriangle, Fingerprint, Calendar,
  Share2, Award, HardDrive, Download, Eye, Zap,
  Activity, Film, Volume2, Globe, Search,
  Cpu, Clock, CheckCircle2, AlertCircle, ExternalLink,
  ArrowRight, Info, EyeOff, LayoutGrid, CheckSquare, Sparkles, GitPullRequest,
  ShieldCheck
} from "lucide-react";

interface Keyframe {
  id: number;
  timestamp: number;
  filepath: string;
  phash: string;
}

interface MediaItem {
  id: number;
  case_id: number;
  filename: string;
  filepath: string;
  mime_type: string;
  sha256: string;
  phash: string;
  dhash: string;
  ahash: string;
  audio_fingerprint: any;
  metadata_sig: any;
  embedding?: number[];
  created_at: string;
  parent_id?: number;
  estimated_origin_id?: number;
  resolution: string;
  file_size: number;
  duration?: number;
  risk_score: number;
  integrity_score: number;
  modification_report: any;
  keyframes: Keyframe[];
}

interface PHashSteps {
  step1: string;
  step2: string;
  step3: string;
  step4: string;
  step5: string;
  step6: string;
  hash: string;
}

interface GraphData {
  nodes: Array<{ id: number; label: string; type: string; risk: number; integrity: number; mime_type: string }>;
  links: Array<{ source: number; target: number; score: number; type: string }>;
  graph_type?: string;
  family_size?: number;
  timeline_confidence?: number;
  timeline_inconclusive?: boolean;
}

function parseGPSInfo(gpsStr: string | undefined): {
  latitude?: string;
  longitude?: string;
  altitude?: string;
  timestamp?: string;
} {
  if (!gpsStr) return {};
  try {
    const latRefMatch = gpsStr.match(/1:\s*['"](N|S)['"]/);
    const latMatch = gpsStr.match(/2:\s*\(([^)]+)\)/);
    const lonRefMatch = gpsStr.match(/3:\s*['"](E|W)['"]/);
    const lonMatch = gpsStr.match(/4:\s*\(([^)]+)\)/);
    const altMatch = gpsStr.match(/6:\s*\(([^)]+)\)/) || gpsStr.match(/6:\s*([\d.]+)/);
    const timeMatch = gpsStr.match(/7:\s*\(([^)]+)\)/);
    const dateMatch = gpsStr.match(/29:\s*['"]([^'"]+)['"]/);

    const parseTupleToDecimal = (tupleStr: string | null): number | null => {
      if (!tupleStr) return null;
      const parts = tupleStr.split(',').map(s => s.trim());
      if (parts.length === 3) {
        const d = parseFloat(parts[0]);
        const m = parseFloat(parts[1]);
        const s = parseFloat(parts[2]);
        if (!isNaN(d) && !isNaN(m) && !isNaN(s)) {
          return d + m / 60 + s / 3600;
        }
      } else if (parts.length === 6) {
        const cleanParts = parts.map(p => p.replace(/[()]/g, ''));
        const d = parseFloat(cleanParts[0]) / (parseFloat(cleanParts[1]) || 1);
        const m = parseFloat(cleanParts[2]) / (parseFloat(cleanParts[3]) || 1);
        const s = parseFloat(cleanParts[4]) / (parseFloat(cleanParts[5]) || 1);
        if (!isNaN(d) && !isNaN(m) && !isNaN(s)) {
          return d + m / 60 + s / 3600;
        }
      }
      return null;
    };

    let latitude = "";
    if (latMatch) {
      const latVal = parseTupleToDecimal(latMatch[1]);
      if (latVal !== null) {
        const ref = latRefMatch ? latRefMatch[1] : "N";
        latitude = `${latVal.toFixed(6)}° ${ref}`;
      }
    }

    let longitude = "";
    if (lonMatch) {
      const lonVal = parseTupleToDecimal(lonMatch[1]);
      if (lonVal !== null) {
        const ref = lonRefMatch ? lonRefMatch[1] : "E";
        longitude = `${lonVal.toFixed(6)}° ${ref}`;
      }
    }

    let altitude = "";
    if (altMatch) {
      const cleanAlt = altMatch[1].replace(/[()]/g, '');
      const altParts = cleanAlt.split(',').map(s => s.trim());
      let altVal = 0;
      if (altParts.length === 2) {
        altVal = parseFloat(altParts[0]) / (parseFloat(altParts[1]) || 1);
      } else {
        altVal = parseFloat(cleanAlt);
      }
      if (!isNaN(altVal)) {
        altitude = `${altVal.toFixed(1)}m`;
      }
    }

    let timestamp = "";
    if (timeMatch) {
      const cleanParts = timeMatch[1].replace(/[()]/g, '').split(',').map(s => s.trim());
      let h = 0, m = 0, s = 0;
      if (cleanParts.length === 3) {
        h = parseFloat(cleanParts[0]);
        m = parseFloat(cleanParts[1]);
        s = parseFloat(cleanParts[2]);
      } else if (cleanParts.length === 6) {
        h = parseFloat(cleanParts[0]) / (parseFloat(cleanParts[1]) || 1);
        m = parseFloat(cleanParts[2]) / (parseFloat(cleanParts[3]) || 1);
        s = parseFloat(cleanParts[4]) / (parseFloat(cleanParts[5]) || 1);
      }
      const dateStr = dateMatch ? dateMatch[1].replace(/:/g, '-') : "";
      timestamp = `${dateStr} ${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor(m)).padStart(2, '0')}:${String(Math.floor(s)).padStart(2, '0')} UTC`;
    }

    return {
      latitude: latitude || undefined,
      longitude: longitude || undefined,
      altitude: altitude || undefined,
      timestamp: timestamp || undefined
    };
  } catch (err) {
    console.error("Error parsing GPSInfo:", err);
    return {};
  }
}

function formatFocalLength(focalStr: string | undefined): string {
  if (!focalStr) return "N/A";
  const clean = focalStr.replace(/[()]/g, '');
  const parts = clean.split(',').map(s => s.trim());
  if (parts.length === 2) {
    const val = parseFloat(parts[0]) / (parseFloat(parts[1]) || 1);
    return isNaN(val) ? "N/A" : `${val.toFixed(1)} mm`;
  }
  const val = parseFloat(clean);
  return isNaN(val) ? "N/A" : `${val.toFixed(1)} mm`;
}

function formatExposureTime(expStr: string | undefined): string {
  if (!expStr) return "N/A";
  const clean = expStr.replace(/[()]/g, '');
  const parts = clean.split(',').map(s => s.trim());
  if (parts.length === 2) {
    const num = parseFloat(parts[0]);
    const den = parseFloat(parts[1]);
    if (!isNaN(num) && !isNaN(den)) {
      if (num === 1) return `1/${den}s`;
      return `${(num / den).toFixed(4)}s`;
    }
  }
  const val = parseFloat(clean);
  if (!isNaN(val)) {
    if (val < 1 && val > 0) {
      const den = Math.round(1 / val);
      return `1/${den}s`;
    }
    return `${val}s`;
  }
  return expStr;
}

function getExifExplanation(exif: any, parsedGPS: any): string {
  const hasCamera = !!(exif.Make || exif.Model);
  const hasGPS = !!(parsedGPS.latitude || parsedGPS.longitude);
  const hasTime = !!exif.DateTimeOriginal;
  const hasSoftware = !!exif.Software;

  if (hasCamera && hasGPS && hasTime) {
    return "All original forensic metadata tags (Camera, GPS, and Time) are intact. Provenance chain is complete.";
  }
  if (!hasCamera && !hasGPS && !hasTime) {
    return "All EXIF metadata tags have been completely stripped from the file container. Original capture details are lost.";
  }

  const present = [];
  const missing = [];

  if (hasCamera) present.push("Camera Sensor (Make/Model)");
  else missing.push("Camera Sensor (Make/Model)");

  if (hasGPS) present.push("GPS Coordinates");
  else missing.push("GPS Coordinates");

  if (hasTime) present.push("Capture Timestamp");
  else missing.push("Capture Timestamp");

  if (hasSoftware) present.push("Editing Software Footprint");

  return `Partial metadata remains. Present: ${present.join(", ")}. Missing/Removed: ${missing.join(", ")}.`;
}

export default function MediaProfilePage() {
  const params = useParams();
  const router = useRouter();
  const mediaId = params.id as string;

  const [item, setItem] = useState<MediaItem | null>(null);
  const [phashSteps, setPhashSteps] = useState<PHashSteps | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [similarItems, setSimilarItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Partial Failures States
  const [graphFailed, setGraphFailed] = useState(false);
  const [similarFailed, setSimilarFailed] = useState(false);
  const [osintFailed, setOsintFailed] = useState(false);
  const [phashStepsFailed, setPhashStepsFailed] = useState(false);

  // Tab State
  const [activeTab, setActiveTab] = useState<"metadata" | "clues" | "aistego" | "osint" | "lineage">("metadata");

  // OSINT Scan States
  const [osintScanStatus, setOsintScanStatus] = useState<string>("Not Started");
  const [osintTags, setOsintTags] = useState<string[]>([]);
  const [osintResults, setOsintResults] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  // UI state
  const [activeVisualizerStep, setActiveVisualizerStep] = useState(1);

  const backendUrl = "http://127.0.0.1:8000";

  const getCleanedQuery = (filename: string) => {
    const namePart = filename.split('.')[0];
    const cleaned = namePart.replace(/[_|-]/g, ' ')
      .replace(/\b(original|copy|modified|variant)\b/gi, '');
    return cleaned.trim().replace(/\s+/g, ' ');
  };

  useEffect(() => {
    if (!mediaId) return;

    const fetchAllDetails = async () => {
      setLoading(true);

      // Reset failure states
      setGraphFailed(false);
      setSimilarFailed(false);
      setOsintFailed(false);
      setPhashStepsFailed(false);

      // 1. Fetch main media item (CRITICAL)
      try {
        const itemRes = await fetch(`${backendUrl}/api/media/${mediaId}`);
        if (itemRes.ok) {
          const itemData = await itemRes.json();
          setItem(itemData);
          setSearchQuery(getCleanedQuery(itemData.filename));
        } else {
          console.error("Main media item fetch returned non-OK status:", itemRes.status);
        }
      } catch (err) {
        console.error("Failed to load main media item details", err);
      }

      // 2. Fetch phash steps (OPTIONAL)
      try {
        const stepsRes = await fetch(`${backendUrl}/api/media/${mediaId}/phash-steps`);
        if (stepsRes.ok) {
          const stepsData = await stepsRes.json();
          setPhashSteps(stepsData);
        } else {
          setPhashStepsFailed(true);
          console.error("Phash steps fetch returned non-OK status:", stepsRes.status);
        }
      } catch (err) {
        setPhashStepsFailed(true);
        console.error("Failed to load phash steps", err);
      }

      // 3. Fetch relationship graph (OPTIONAL)
      try {
        const graphRes = await fetch(`${backendUrl}/api/media/${mediaId}/relationship-graph`);
        if (graphRes.ok) {
          const graphData = await graphRes.json();
          if (graphData.error === "graph_generation_failed") {
            setGraphFailed(true);
            console.error("Relationship graph endpoint returned backend error flag");
          } else {
            setGraphData(graphData);
          }
        } else {
          setGraphFailed(true);
          console.error("Relationship graph fetch returned non-OK status:", graphRes.status);
        }
      } catch (err) {
        setGraphFailed(true);
        console.error("Failed to load relationship graph", err);
      }

      // 4. Fetch similar items (OPTIONAL)
      try {
        const similarRes = await fetch(`${backendUrl}/api/media/${mediaId}/similar`);
        if (similarRes.ok) {
          const similarData = await similarRes.json();
          setSimilarItems(similarData);
        } else {
          setSimilarFailed(true);
          console.error("Similar items fetch returned non-OK status:", similarRes.status);
        }
      } catch (err) {
        setSimilarFailed(true);
        console.error("Failed to load similar items", err);
      }

      // 5. Fetch OSINT details (OPTIONAL)
      try {
        const statusRes = await fetch(`${backendUrl}/api/osint/status/${mediaId}`);
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setOsintTags(statusData.tags || []);

          const isFinal = ["Completed", "Verified Matches Found", "No Matches Found", "Provider Unavailable"].includes(statusData.status);
          if (isFinal) {
            const resultsRes = await fetch(`${backendUrl}/api/osint/results/${mediaId}`);
            if (resultsRes.ok) {
              const resultsData = await resultsRes.json();
              setOsintResults(resultsData);

              // Map scan status for backward compatibility
              let mappedStatus = statusData.status;
              if (statusData.status === "Completed") {
                const hasReal = resultsData.some((r: any) => r.source_type === "real_provider" || r.source_type === "apify");
                mappedStatus = hasReal ? "Verified Matches Found" : "Provider Unavailable";
              }
              setOsintScanStatus(mappedStatus);
            } else {
              setOsintFailed(true);
              console.error("OSINT results fetch returned non-OK status:", resultsRes.status);
            }
          } else if (statusData.status === "Running" || statusData.status === "Pending") {
            setIsScanning(true);
            setOsintScanStatus(statusData.status);
          } else {
            setOsintScanStatus("Not Searched");
          }
        } else {
          setOsintFailed(true);
          console.error("OSINT status fetch returned non-OK status:", statusRes.status);
        }
      } catch (err) {
        setOsintFailed(true);
        console.error("Failed to load OSINT details", err);
      }

      setLoading(false);
    };

    fetchAllDetails();
  }, [mediaId]);

  useEffect(() => {
    if (!mediaId) return;
    if (osintScanStatus === "Running" || osintScanStatus === "Pending") {
      let active = true;
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`${backendUrl}/api/osint/status/${mediaId}`);
          if (res.ok && active) {
            const data = await res.json();
            setOsintTags(data.tags || []);

            const isFinal = ["Completed", "Verified Matches Found", "No Matches Found", "Provider Unavailable"].includes(data.status);
            if (isFinal) {
              clearInterval(interval);
              setIsScanning(false);
              const resultsRes = await fetch(`${backendUrl}/api/osint/results/${mediaId}`);
              if (resultsRes.ok && active) {
                const resultsData = await resultsRes.json();
                setOsintResults(resultsData);

                let mappedStatus = data.status;
                if (data.status === "Completed") {
                  const hasReal = resultsData.some((r: any) => r.source_type === "real_provider" || r.source_type === "apify");
                  mappedStatus = hasReal ? "Verified Matches Found" : "Provider Unavailable";
                }
                setOsintScanStatus(mappedStatus);
              }
            } else if (data.status === "Failed") {
              clearInterval(interval);
              setIsScanning(false);
              setScanError(data.error_message || "OSINT scan failed.");
              setOsintScanStatus("Failed");
            } else {
              setOsintScanStatus(data.status);
            }
          }
        } catch (e) {
          console.error("Error polling OSINT status:", e);
          if (active) {
            clearInterval(interval);
            setIsScanning(false);
          }
        }
      }, 1500);

      return () => {
        active = false;
        clearInterval(interval);
      };
    }
  }, [osintScanStatus, mediaId]);

  const handleLaunchScan = async () => {
    setIsScanning(true);
    setScanError(null);
    setOsintScanStatus("Pending");

    try {
      const resp = await fetch(`${backendUrl}/api/osint/scan/${mediaId}`, {
        method: "POST"
      });
      if (!resp.ok) {
        throw new Error("Failed to trigger scan on backend.");
      }
      const data = await resp.json();
      setOsintScanStatus(data.status);
    } catch (err: any) {
      console.error(err);
      setScanError(err.message || "Could not launch OSINT scan.");
      setIsScanning(false);
      setOsintScanStatus("Failed");
    }
  };

  if (loading) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent border-[#00E5FF]"></div>
        <span className="font-mono text-xs text-gray-500 tracking-widest font-bold">DESERIALIZING CHROMATOGRAPHIC SIGNATURES...</span>
      </div>
    );
  }

  if (!item) {
    return (
      <div className="flex h-[50vh] flex-col items-center justify-center gap-4 text-center">
        <AlertTriangle className="h-10 w-10 text-[#FF3366]" />
        <span className="font-mono text-sm font-bold text-white uppercase">MEDIA PROFILE NOT FOUND</span>
        <button
          onClick={() => router.push("/")}
          className="rounded border border-[rgba(255,255,255,0.1)] px-4 py-2 font-mono text-xs text-gray-400 hover:text-white"
        >
          RETURN TO DASHBOARD
        </button>
      </div>
    );
  }

  const isVideo = item.mime_type.startsWith("video/");

  const stepLabels = [
    { id: 1, name: "Original Input", desc: "Source image aspect" },
    { id: 2, name: "Luma Grayscale", desc: "Luminosity filter" },
    { id: 3, name: "32x32 Resize", desc: "Low pass scale" },
    { id: 4, name: "2D DCT Heatmap", desc: "Frequency grids" },
    { id: 5, name: "8x8 Extract", desc: "Low frequencies" },
    { id: 6, name: "Binary Hash", desc: "Median compared" }
  ];

  // Render Directed hierarchical Tree (Probable Evolution Chain)
  const renderSVGGraph = () => {
    if (!graphData || graphData.nodes.length === 0) return null;

    const graphType = graphData.graph_type || (graphData.nodes.length < 20 ? "full" : graphData.nodes.length <= 100 ? "collapsible" : "summary");

    if (graphType === "summary") {
      const variantCounts: Record<string, number> = {};
      graphData.nodes.forEach((n: any) => {
        if (n.type !== "original") {
          const mime = n.mime_type?.split("/")[1] || "variant";
          variantCounts[mime] = (variantCounts[mime] || 0) + 1;
        }
      });
      return (
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.05)] rounded-lg p-6 font-mono text-xs text-gray-400 space-y-4">
          <div className="flex items-center justify-between border-b border-[rgba(255,255,255,0.06)] pb-2">
            <span className="text-white font-bold uppercase tracking-wider">Family Scale Warning: Node Overflow Protection</span>
            <span className="bg-[#FFCC00]/10 text-[#FFCC00] border border-[#FFCC00]/20 px-1.5 py-0.5 rounded text-[8px] font-black uppercase">Summary Mode</span>
          </div>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            This media cluster contains <span className="text-white font-black">{(graphData.family_size || graphData.nodes.length)} items</span>, which exceeds visual rendering thresholds. Direct SVG drawing is suppressed to maintain interface responsiveness.
          </p>
          <div className="grid grid-cols-2 gap-3 text-[10px] border-t border-[rgba(255,255,255,0.03)] pt-3">
            <div>
              <span className="block text-[8px] text-gray-500 uppercase">Total Variant Count</span>
              <span className="text-white font-bold text-sm">{(graphData.family_size || graphData.nodes.length) - 1} items</span>
            </div>
            <div>
              <span className="block text-[8px] text-gray-500">FORMAT DISTRIBUTION</span>
              <span className="text-[#00FF9D] font-semibold">
                {Object.entries(variantCounts).map(([fmt, cnt]) => `${cnt} ${fmt}`).join(", ")}
              </span>
            </div>
          </div>
        </div>
      );
    }

    if (graphType === "collapsible") {
      const grouped: Record<string, typeof graphData.nodes> = {};
      graphData.nodes.forEach((n: any) => {
        if (n.type === "original") return;
        const key = n.mime_type?.split("/")[1]?.toUpperCase() || "VARIANT";
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(n);
      });

      const rootNode = graphData.nodes.find((n: any) => n.type === "original") || graphData.nodes[0];

      return (
        <div className="bg-[#09090D]/40 border border-[rgba(255,255,255,0.03)] rounded-lg p-5 font-mono text-xs text-gray-400 space-y-4">
          <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.05)] pb-2 mb-2">
            <span className="text-white font-bold uppercase tracking-wider text-[10px]">Clustered Variants Hierarchy</span>
            <span className="bg-[#00E5FF]/10 text-[#00E5FF] border border-[#00E5FF]/20 px-1.5 py-0.5 rounded text-[8px] font-black uppercase">Clustered Mode</span>
          </div>

          <div className="space-y-3">
            <div className="p-2.5 rounded border border-[#7C3AED]/30 bg-[#7C3AED]/5 flex justify-between items-center">
              <div>
                <span className="block text-[8px] text-[#A78BFA] uppercase tracking-wider font-bold">Estimated Origin Root</span>
                <span className="text-white font-bold">{rootNode.label}</span>
              </div>
              <span className="text-[9px] text-[#A78BFA] font-black uppercase">ROOT</span>
            </div>

            <div className="space-y-2 pl-2 border-l border-[rgba(255,255,255,0.05)]">
              {Object.entries(grouped).map(([grpName, groupNodes]: any) => (
                <details key={grpName} className="group border border-[rgba(255,255,255,0.04)] bg-[#111116]/50 rounded p-2">
                  <summary className="cursor-pointer font-bold text-white flex justify-between items-center select-none text-[10px]">
                    <span className="flex items-center gap-1.5">
                      <span className="text-[#00E5FF]">[{grpName} CLUSTER]</span>
                      <span className="text-gray-500 font-normal">({groupNodes.length} items)</span>
                    </span>
                    <span className="text-[8px] text-gray-500 uppercase tracking-widest group-open:hidden">Expand +</span>
                    <span className="text-[8px] text-gray-500 uppercase tracking-widest hidden group-open:inline">Collapse -</span>
                  </summary>
                  <div className="mt-2 pl-3 space-y-1.5 border-t border-[rgba(255,255,255,0.03)] pt-2 max-h-40 overflow-y-auto">
                    {groupNodes.map((n: any) => {
                      const isCurrent = n.id === item.id;
                      return (
                        <div
                          key={n.id}
                          onClick={() => {
                            if (n.id !== item.id) {
                              router.push(`/media/${n.id}`);
                            }
                          }}
                          className={`p-1.5 rounded text-[9px] flex justify-between items-center cursor-pointer transition-colors ${isCurrent
                              ? "bg-[rgba(0,229,255,0.05)] text-[#00E5FF] font-bold"
                              : "text-gray-400 hover:text-white"
                            }`}
                        >
                          <span className="truncate max-w-[200px]">{n.label}</span>
                          <span className="text-gray-500 text-[8px]">INTEGRITY: {n.integrity}</span>
                        </div>
                      );
                    })}
                  </div>
                </details>
              ))}
            </div>
          </div>
        </div>
      );
    }

    const width = 450;
    const height = 300;
    const center_x = width / 2;

    const children: Record<number, number[]> = {};
    graphData.nodes.forEach(n => children[n.id] = []);
    graphData.links.forEach(l => {
      if (children[l.source]) {
        children[l.source].push(l.target);
      }
    });

    const rootNode = graphData.nodes.find(n => n.type === "original") || graphData.nodes[0];

    const levels: Record<number, number> = {};
    const levelNodes: Record<number, number[]> = {};
    const queue: Array<[number, number]> = [[rootNode.id, 0]];

    while (queue.length > 0) {
      const [id, lvl] = queue.shift()!;
      levels[id] = lvl;
      if (!levelNodes[lvl]) levelNodes[lvl] = [];
      if (!levelNodes[lvl].includes(id)) {
        levelNodes[lvl].push(id);
      }
      (children[id] || []).forEach(childId => {
        queue.push([childId, lvl + 1]);
      });
    }

    const maxLevel = Math.max(...Object.values(levels), 0);
    const coords: Record<number, { x: number; y: number }> = {};
    const yStep = maxLevel > 0 ? 200 / maxLevel : 0;

    graphData.nodes.forEach(n => {
      const id = n.id;
      const lvl = levels[id] !== undefined ? levels[id] : 0;
      const nodesAtLvl = levelNodes[lvl] || [id];
      const idx = nodesAtLvl.indexOf(id);
      const numNodes = nodesAtLvl.length;

      const x = numNodes === 1
        ? center_x
        : center_x - 130 + (idx * 260) / (numNodes - 1);
      const y = 40 + lvl * yStep;

      coords[id] = { x, y };
    });

    return (
      <svg width="100%" height="300" className="bg-[#09090D]/40 border border-[rgba(255,255,255,0.03)] rounded-lg">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="17" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#00E5FF" />
          </marker>
        </defs>

        {graphData.links.map((link, idx) => {
          const src = coords[link.source];
          const tgt = coords[link.target];
          if (!src || !tgt) return null;
          return (
            <g key={idx}>
              <line
                x1={src.x} y1={src.y}
                x2={tgt.x} y2={tgt.y}
                stroke="#00E5FF"
                strokeWidth="1.5"
                strokeOpacity="0.6"
                markerEnd="url(#arrow)"
              />
              <text
                x={(src.x + tgt.x) / 2} y={(src.y + tgt.y) / 2 - 5}
                className="font-mono fill-[#00FF9D] text-[7px] font-bold"
                textAnchor="middle"
              >
                {link.type.replace("Variant", "")} ({Math.round(link.score * 100)}%)
              </text>
            </g>
          );
        })}

        {graphData.nodes.map((node) => {
          const coord = coords[node.id];
          if (!coord) return null;
          const isCurrent = node.id === item.id;
          return (
            <g
              key={node.id}
              className="cursor-pointer"
              onClick={() => {
                if (node.id !== item.id) {
                  router.push(`/media/${node.id}`);
                }
              }}
            >
              <circle
                cx={coord.x} cy={coord.y} r={node.type === "original" ? "14" : "10"}
                fill={node.type === "original" ? "#7C3AED" : "#0D0C15"}
                stroke={isCurrent ? "#00E5FF" : node.type === "original" ? "#7C3AED" : "rgba(255,255,255,0.3)"}
                strokeWidth={isCurrent ? "2.5" : "1.5"}
                className={isCurrent ? "animate-pulse" : ""}
              />
              <text
                x={coord.x} y={coord.y + (node.type === "original" ? "26" : "22")}
                className={`font-mono text-[8px] font-semibold ${isCurrent ? "fill-[#00E5FF]" : "fill-gray-400"}`}
                textAnchor="middle"
              >
                {node.label.length > 15 ? `${node.label.slice(0, 12)}...` : node.label}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  const getConfidenceBadgeColor = (level: string) => {
    if (level === "High") return "text-[#00FF9D] bg-[rgba(0,255,157,0.08)] border-[rgba(0,255,157,0.2)]";
    if (level === "Medium") return "text-[#F59E0B] bg-[rgba(245,158,11,0.08)] border-[rgba(245,158,11,0.2)]";
    return "text-[#FF3366] bg-[rgba(255,51,102,0.08)] border-[rgba(255,51,102,0.2)]";
  };

  const getVerdictBadge = () => {
    const risk = item.risk_score;
    const isScreenshot = item.modification_report?.screenshot_indicators?.status === "Likely Screenshot" || item.modification_report?.screenshot_indicators?.level === "High";
    const stegoSuspicion = item.modification_report?.forensic_investigation?.suspicion_score ?? 0;
    const aiProb = item.modification_report?.ai_detection?.probability ?? 0;

    if (risk > 65 || stegoSuspicion >= 50 || aiProb >= 50) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-[#FF3366]/30 bg-[#FF3366]/10 text-[9px] font-black text-[#FF3366] uppercase tracking-wider glow-text-red">
          <AlertTriangle className="h-3 w-3" />
          HIGH RISK / MANIPULATED
        </span>
      );
    }
    if (risk > 30) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-[#F59E0B]/30 bg-[#F59E0B]/10 text-[9px] font-black text-[#F59E0B] uppercase tracking-wider">
          <AlertCircle className="h-3 w-3" />
          MODERATE RISK / MODIFIED
        </span>
      );
    }
    if (isScreenshot) {
      return (
        <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-[#00E5FF]/30 bg-[#00E5FF]/10 text-[9px] font-black text-[#00E5FF] uppercase tracking-wider glow-text-cyan">
          <Info className="h-3 w-3" />
          SYSTEM SCREENSHOT
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-[#00FF9D]/30 bg-[#00FF9D]/10 text-[9px] font-black text-[#00FF9D] uppercase tracking-wider glow-text-green">
        <CheckCircle2 className="h-3 w-3" />
        LOW RISK / CLEAN BASELINE
      </span>
    );
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title Cover Header */}
      <div className="relative overflow-hidden rounded-xl border border-[rgba(255,255,255,0.05)] bg-[#0C0C12] p-6 shadow-2xl">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-[9px] uppercase tracking-widest text-[#7C3AED]">Asset Profile Registry</span>
              {getVerdictBadge()}
            </div>
            <h1 className="font-mono text-xl font-black text-white tracking-wider truncate max-w-[400px]">
              {item.filename}
            </h1>
            {(graphFailed || similarFailed || osintFailed || phashStepsFailed) && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded border border-[#FFCC00]/20 bg-[#FFCC00]/10 text-[#FFCC00] font-mono text-[9px] w-fit">
                <AlertTriangle className="h-3.5 w-3.5 animate-pulse" />
                <span>PARTIAL TELEMETRY OFFLINE: {
                  [
                    graphFailed ? "RELATIONSHIP GRAPH" : null,
                    similarFailed ? "SIMILARITY ENGINE" : null,
                    phashStepsFailed ? "PHASH DIAGNOSTICS" : null,
                    osintFailed ? "OSINT SERVICES" : null
                  ].filter(Boolean).join(" | ")
                }</span>
              </div>
            )}
            <div className="flex flex-wrap gap-4 text-[10px] font-mono text-gray-500">
              <span className="flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5" />
                {new Date(item.created_at).toLocaleDateString()}
              </span>
              <span className="flex items-center gap-1">
                <HardDrive className="h-3.5 w-3.5" />
                {(item.file_size / (1024 * 1024)).toFixed(2)} MB
              </span>
              <span className="flex items-center gap-1">
                <Activity className="h-3.5 w-3.5" />
                MIME: {item.mime_type}
              </span>
            </div>
          </div>

          {/* Export PDF Button */}
          <a
            href={`${backendUrl}/api/media/${item.id}/pdf-report`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded border border-[#00E5FF] bg-transparent hover:bg-[#00E5FF]/10 px-4 py-2 font-mono text-xs text-[#00E5FF] font-semibold transition-colors"
          >
            <Download className="h-4 w-4 animate-bounce" />
            EXPORT PDF REPORT
          </a>
        </div>
      </div>

      {/* Investigation Summary Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-9 gap-4">
        {/* Asset Type */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Asset Classification</span>
          <span className="block mt-1.5 font-mono text-[11px] font-bold text-white uppercase tracking-wide truncate">
            {item.modification_report?.forensic_investigation?.asset_classification ?? item.modification_report?.asset_classification ?? "Not Evaluated"}
          </span>
        </div>

        {/* Manipulation Risk */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <div className="flex justify-between items-center">
            <span className="font-mono text-[8px] text-gray-500 uppercase tracking-widest">Manipulation Risk</span>
            <span className={`font-mono text-[10px] font-bold ${item.risk_score > 60 ? "text-[#FF3366]" : item.risk_score > 30 ? "text-[#F59E0B]" : "text-[#00FF9D]"}`}>
              {item.risk_score}%
            </span>
          </div>
          <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden mt-2">
            <div className={`h-full rounded-full ${item.risk_score > 60 ? "bg-[#FF3366]" : item.risk_score > 30 ? "bg-[#F59E0B]" : "bg-[#00FF9D]"}`} style={{ width: `${item.risk_score}%` }}></div>
          </div>
        </div>

        {/* ML Classification */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">ML Classification</span>
          <span className={`block mt-1.5 font-mono text-[11px] font-bold uppercase tracking-wide truncate ${(item.modification_report?.ml_classification ?? "NOT EVALUATED") === "TAMPERED" ? "text-[#FF3366]" :
              (item.modification_report?.ml_classification ?? "NOT EVALUATED") === "CLEAN" ? "text-[#00FF9D]" :
                "text-gray-400"
            }`}>
            {item.modification_report?.ml_classification ?? "NOT EVALUATED"}
          </span>
        </div>

        {/* ML Tampering Probability */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <div className="flex justify-between items-center">
            <span className="font-mono text-[8px] text-gray-500 uppercase tracking-widest">ML Tampering Prob.</span>
            <span className={`font-mono text-[10px] font-bold ${(item.modification_report?.ml_classification ?? "NOT EVALUATED") === "TAMPERED" ? "text-[#FF3366]" : (item.modification_report?.ml_classification ?? "NOT EVALUATED") === "CLEAN" ? "text-[#00FF9D]" : "text-gray-400"}`}>
              {Math.round((item.modification_report?.ml_tampering_probability ?? 0) * 100)}%
            </span>
          </div>
          <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden mt-2">
            <div className={`h-full rounded-full ${(item.modification_report?.ml_classification ?? "NOT EVALUATED") === "TAMPERED" ? "bg-[#FF3366]" : (item.modification_report?.ml_classification ?? "NOT EVALUATED") === "CLEAN" ? "bg-[#00FF9D]" : "bg-gray-500"}`} style={{ width: `${Math.round((item.modification_report?.ml_tampering_probability ?? 0) * 100)}%` }}></div>
          </div>
        </div>

        {/* Screenshot Probability */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <div className="flex justify-between items-center">
            <span className="font-mono text-[8px] text-gray-500 uppercase tracking-widest">Screenshot Prob.</span>
            <span className="font-mono text-[10px] font-bold text-[#00E5FF]">
              {item.modification_report?.screenshot_indicators?.confidence ?? 0}%
            </span>
          </div>
          <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden mt-2">
            <div className="h-full rounded-full bg-[#00E5FF]" style={{ width: `${item.modification_report?.screenshot_indicators?.confidence ?? 0}%` }}></div>
          </div>
        </div>

        {/* AI Gen Probability */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <div className="flex justify-between items-center">
            <span className="font-mono text-[8px] text-gray-500 uppercase tracking-widest">AI Artifact Score</span>
            <span className="font-mono text-[10px] font-bold text-[#FFCC00]">
              {item.modification_report?.ai_detection?.probability ?? 0}%
            </span>
          </div>
          <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden mt-2">
            <div className="h-full rounded-full bg-[#FFCC00]" style={{ width: `${item.modification_report?.ai_detection?.probability ?? 0}%` }}></div>
          </div>
        </div>

        {/* Steganography Suspicion */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <div className="flex justify-between items-center">
            <span className="font-mono text-[8px] text-gray-500 uppercase tracking-widest">Stego Suspicion</span>
            <span className="font-mono text-[10px] font-bold text-[#FF3366]">
              {item.modification_report?.forensic_investigation?.suspicion_score ?? 0}%
            </span>
          </div>
          <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden mt-2">
            <div className="h-full rounded-full bg-[#FF3366]" style={{ width: `${item.modification_report?.forensic_investigation?.suspicion_score ?? 0}%` }}></div>
          </div>
        </div>

        {/* Reverse Search Status */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Reverse Search</span>
          <span className={`block mt-1.5 font-mono text-[9px] font-extrabold uppercase tracking-wider ${osintScanStatus === "Verified Matches Found" ? "text-[#00FF9D] glow-text-green" :
              osintScanStatus === "No Matches Found" ? "text-gray-400" :
                osintScanStatus === "Provider Unavailable" ? "text-[#FFCC00] glow-text-yellow" :
                  osintScanStatus === "Running" || osintScanStatus === "Pending" ? "text-[#00E5FF] animate-pulse" :
                    "text-gray-500"
            }`}>
            {osintScanStatus === "Not Started" || osintScanStatus === "Pending" ? "Not Searched" : osintScanStatus}
          </span>
        </div>

        {/* Confidence Score */}
        <div className="bg-[#09090D]/60 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 flex flex-col justify-between min-h-[90px] hover:border-[rgba(255,255,255,0.1)] transition-colors">
          <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Forensic Conf.</span>
          <span className="block mt-1.5 font-mono text-sm font-extrabold text-white">
            {item.modification_report?.overall_investigation_confidence?.score ?? item.modification_report?.executive_summary?.confidence_score ?? 90}%
          </span>
        </div>
      </div>

      {/* Tab Selector */}
      <div className="flex flex-wrap border-b border-[rgba(255,255,255,0.06)] gap-6">
        <button
          onClick={() => setActiveTab("metadata")}
          className={`pb-4 font-mono text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${activeTab === "metadata"
              ? "border-[#00E5FF] text-[#00E5FF] shadow-[0_4px_12px_-4px_rgba(0,229,255,0.4)]"
              : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
        >
          Metadata & Manipulation
        </button>
        <button
          onClick={() => setActiveTab("clues")}
          className={`pb-4 font-mono text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${activeTab === "clues"
              ? "border-[#00FF9D] text-[#00FF9D] shadow-[0_4px_12px_-4px_rgba(0,255,157,0.4)]"
              : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
        >
          Blind Clues
        </button>
        <button
          onClick={() => setActiveTab("aistego")}
          className={`pb-4 font-mono text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${activeTab === "aistego"
              ? "border-[#FF3366] text-[#FF3366] shadow-[0_4px_12px_-4px_rgba(255,51,102,0.4)]"
              : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
        >
          AI & Stego Detection
        </button>
        <button
          onClick={() => setActiveTab("osint")}
          className={`pb-4 font-mono text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${activeTab === "osint"
              ? "border-[#7C3AED] text-[#7C3AED] shadow-[0_4px_12px_-4px_rgba(124,58,237,0.4)]"
              : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
        >
          Web OSINT Hunt
        </button>
        <button
          onClick={() => setActiveTab("lineage")}
          className={`pb-4 font-mono text-xs font-bold tracking-widest uppercase transition-all border-b-2 ${activeTab === "lineage"
              ? "border-[#F59E0B] text-[#F59E0B] shadow-[0_4px_12px_-4px_rgba(245,158,11,0.4)]"
              : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
        >
          Case Clustering & Lineage
        </button>
      </div>

      {/* Metadata & Manipulation Tab */}
      {activeTab === "metadata" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fadeIn">
          {/* Metadata Profile Column */}
          <div className="cyber-card p-6 space-y-6 lg:col-span-1">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
              <Fingerprint className="h-4 w-4 text-[#00E5FF]" />
              <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                Media DNA Profile
              </h2>
            </div>

            <div className="space-y-4 font-mono text-xs text-gray-400">
              <div>
                <span className="block text-[8px] uppercase tracking-widest text-gray-500">SHA256 CHECKSUM</span>
                <span className="block break-all text-white bg-[#14141A] p-2 rounded border border-[rgba(255,255,255,0.03)] mt-1 text-[10px] select-all font-mono">
                  {item.sha256}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)] text-center">
                  <span className="block text-[7px] text-gray-500 uppercase tracking-widest">pHash</span>
                  <span className="block mt-0.5 text-[9px] font-bold text-white truncate">{item.phash || "N/A"}</span>
                </div>
                <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)] text-center">
                  <span className="block text-[7px] text-gray-500 uppercase tracking-widest">dHash</span>
                  <span className="block mt-0.5 text-[9px] font-bold text-white truncate">{item.dhash || "N/A"}</span>
                </div>
                <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)] text-center">
                  <span className="block text-[7px] text-gray-500 uppercase tracking-widest">aHash</span>
                  <span className="block mt-0.5 text-[9px] font-bold text-white truncate">{item.ahash || "N/A"}</span>
                </div>
              </div>

              {/* Technical Parameters */}
              <div className="border-t border-[rgba(255,255,255,0.05)] pt-4 space-y-2 text-[10px]">
                <div className="flex justify-between">
                  <span>Dimensions:</span>
                  <span className="text-white font-bold">{item.resolution || "Unknown"}</span>
                </div>
                {isVideo && (
                  <div className="flex justify-between">
                    <span>Duration:</span>
                    <span className="text-white font-bold">{item.duration?.toFixed(2)} seconds</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span>Camera EXIF Info:</span>
                  <span className="text-white font-bold">
                    {item.metadata_sig?.exif?.Model || "None / Stripped"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* EXIF Metadata Inspector Table */}
          <div className="cyber-card p-6 space-y-4 lg:col-span-1">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
              <FileText className="h-4 w-4 text-[#A78BFA]" />
              <h2 className="font-mono text-xs font-bold text-[#A78BFA] tracking-widest uppercase">
                EXIF Metadata Tags
              </h2>
            </div>
            {item.metadata_sig?.exif && Object.keys(item.metadata_sig.exif).length > 0 ? (() => {
              const exif = item.metadata_sig.exif;
              const parsedGPS = parseGPSInfo(exif.GPSInfo);
              const hasCamera = !!(exif.Make || exif.Model);
              const hasGPS = !!(parsedGPS.latitude || parsedGPS.longitude);
              const hasTime = !!exif.DateTimeOriginal;
              const hasSoftware = !!exif.Software;

              const forensicRows = [
                { label: "Camera Make", val: exif.Make },
                { label: "Camera Model", val: exif.Model },
                { label: "Capture Time", val: exif.DateTimeOriginal },
                { label: "Exposure Time", val: formatExposureTime(exif.ExposureTime) },
                { label: "ISO Speed", val: exif.ISOSpeedRatings },
                { label: "Focal Length", val: formatFocalLength(exif.FocalLength) },
                { label: "GPS Latitude", val: parsedGPS.latitude },
                { label: "GPS Longitude", val: parsedGPS.longitude },
                { label: "GPS Altitude", val: parsedGPS.altitude },
                { label: "GPS Timestamp", val: parsedGPS.timestamp },
                { label: "Editing Software", val: exif.Software }
              ].filter(row => row.val !== undefined && row.val !== "N/A");

              return (
                <div className="space-y-3 mt-2">
                  {/* Summary Section */}
                  <div className="p-3 bg-[#111116] rounded border border-[rgba(255,255,255,0.03)] space-y-1.5 font-mono text-[9px]">
                    <span className="block text-[8px] text-gray-500 font-bold uppercase tracking-wider mb-1">EXIF Assessment</span>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Camera Metadata:</span>
                      <span className={hasCamera ? "text-[#00FF9D] font-bold" : "text-[#FF3366]"}>{hasCamera ? "Present" : "Absent"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">GPS Metadata:</span>
                      <span className={hasGPS ? "text-[#00FF9D] font-bold" : "text-[#FF3366]"}>{hasGPS ? "Present" : "Absent"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Capture Timestamp:</span>
                      <span className={hasTime ? "text-[#00FF9D] font-bold" : "text-[#FF3366]"}>{hasTime ? "Present" : "Absent"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Editing Software:</span>
                      <span className={hasSoftware ? "text-[#FFCC00] font-bold" : "text-[#00FF9D]"}>{hasSoftware ? "Present" : "Absent"}</span>
                    </div>
                  </div>

                  {/* Forensic Table */}
                  <div className="overflow-y-auto max-h-[160px] border border-[rgba(255,255,255,0.05)] rounded p-2 bg-[#08080C]">
                    <table className="w-full text-left font-mono text-[9px] text-gray-400">
                      <tbody>
                        {forensicRows.map((row, idx) => (
                          <tr key={idx} className="border-b border-[rgba(255,255,255,0.03)]">
                            <td className="py-1 text-gray-500 font-bold pr-2">{row.label}</td>
                            <td className="py-1 text-white break-all">{row.val}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Explanation Section */}
                  <div className="p-2.5 bg-[#FF3366]/5 border border-[#FF3366]/10 rounded font-mono text-[9px] text-gray-400 leading-tight">
                    {getExifExplanation(exif, parsedGPS)}
                  </div>
                </div>
              );
            })() : (
              <div className="p-4 bg-[#FF3366]/5 border border-[#FF3366]/10 rounded font-mono text-xs mt-2 flex flex-col gap-2 h-[220px] overflow-y-auto text-left">
                <div className="flex flex-col">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Metadata Status</span>
                  <span className="text-[#FF3366] font-bold text-[11px]">Missing</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Camera Information</span>
                  <span className="text-gray-300 text-[11px]">Unavailable</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">GPS Information</span>
                  <span className="text-gray-300 text-[11px]">Unavailable</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Capture Timestamp</span>
                  <span className="text-gray-300 text-[11px]">Unavailable</span>
                </div>
                <div className="flex flex-col pt-1 border-t border-[rgba(255,51,102,0.1)]">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Provenance Impact</span>
                  <span className="text-[#FF3366] font-bold text-[11px]">High</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Forensic Consequence</span>
                  <span className="text-gray-300 text-[10px] leading-tight">Original capture source cannot be verified.</span>
                </div>
              </div>
            )}
          </div>

          {/* Metadata Trust Score & Diagnostics */}
          <div className="cyber-card p-6 space-y-4 lg:col-span-1">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
              <Shield className="h-4 w-4 text-[#00FF9D]" />
              <h2 className="font-mono text-xs font-bold text-[#00FF9D] tracking-widest uppercase">
                Metadata Trust & ELA
              </h2>
            </div>

            <div className="space-y-3 font-mono text-xs text-gray-400">
              <div className="p-3 bg-[#111116] rounded border border-[rgba(255,255,255,0.03)] flex justify-between items-center">
                <span className="text-gray-500 font-bold uppercase tracking-wider text-[10px]">Metadata Trust Score</span>
                <span className={`text-sm font-black ${(item.modification_report?.metadata_intelligence?.metadata_trust_score ?? 100) >= 70 ? "text-[#00FF9D]" : "text-[#FFCC00]"
                  }`}>
                  {item.modification_report?.metadata_intelligence?.metadata_trust_score ?? 100}%
                </span>
              </div>

              {item.modification_report?.metadata_intelligence?.metadata_evidence_summary?.trust_score_breakdown && (
                <div className="p-3 bg-[#111116] rounded border border-[rgba(255,255,255,0.03)] space-y-2">
                  <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.05)] pb-1">
                    <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Trust Score Breakdown</span>
                    <span className="text-[10px] text-gray-400 font-bold">
                      Base: {item.modification_report.metadata_intelligence.metadata_evidence_summary.trust_score_breakdown.base_score}%
                    </span>
                  </div>
                  <div className="space-y-1">
                    {item.modification_report.metadata_intelligence.metadata_evidence_summary.trust_score_breakdown.penalties?.map((p: any, idx: number) => (
                      <div key={idx} className="flex justify-between text-[10px] leading-tight">
                        <span className="text-gray-400 font-bold">{p.name}:</span>
                        <span className="text-[#FF3366] font-bold">-{p.deduction}%</span>
                      </div>
                    ))}
                    {(!item.modification_report.metadata_intelligence.metadata_evidence_summary.trust_score_breakdown.penalties ||
                      item.modification_report.metadata_intelligence.metadata_evidence_summary.trust_score_breakdown.penalties.length === 0) && (
                        <span className="text-[#00FF9D] text-[10px]">No penalties applied.</span>
                      )}
                  </div>
                  <div className="text-[9px] text-gray-500 italic pt-1 border-t border-[rgba(255,255,255,0.03)] leading-tight">
                    {item.modification_report.metadata_intelligence.metadata_evidence_summary.trust_score_breakdown.explanation}
                  </div>
                </div>
              )}

              {item.modification_report?.metadata_intelligence?.metadata_evidence_summary && (
                <div className="p-3 bg-[#111116] rounded border border-[rgba(255,255,255,0.03)] space-y-2">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Provenance Confidence</span>
                    <span className={`font-bold ${item.modification_report.metadata_intelligence.metadata_evidence_summary.provenance_confidence === "HIGH"
                        ? "text-[#00FF9D]"
                        : item.modification_report.metadata_intelligence.metadata_evidence_summary.provenance_confidence === "MEDIUM"
                          ? "text-[#FFCC00]"
                          : "text-[#FF3366]"
                      }`}>
                      {item.modification_report.metadata_intelligence.metadata_evidence_summary.provenance_confidence}
                    </span>
                  </div>
                  <div className="flex flex-col gap-1 border-t border-[rgba(255,255,255,0.05)] pt-1 text-[10px]">
                    <span className="text-gray-500 font-bold uppercase tracking-wider text-[9px]">Likely Distribution Channels</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {item.modification_report.metadata_intelligence.metadata_evidence_summary.likely_distribution_channel?.map((ch: string, idx: number) => (
                        <span key={idx} className="px-1.5 py-0.5 rounded text-[8px] bg-[rgba(255,255,255,0.05)] text-gray-300 border border-[rgba(255,255,255,0.1)]">
                          {ch}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-1.5 text-[10px] pt-1">
                <div className="flex justify-between">
                  <span>EXIF Structure:</span>
                  <span className="text-white">{item.modification_report?.metadata_intelligence?.exif_status ?? "Analysis Unavailable"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Recompression:</span>
                  <span className="text-white">{item.modification_report?.manipulation_analysis?.recompression_indicators ?? "Analysis Unavailable"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Copy-Move Indicators:</span>
                  <span className="text-white">{item.modification_report?.manipulation_analysis?.copy_move_indicators ?? "Analysis Unavailable"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Compression Level:</span>
                  <span className="text-white">{item.modification_report?.manipulation_analysis?.compression_artifacts ?? "Analysis Unavailable"}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Anomaly Status Flags */}
          <div className="cyber-card p-6 lg:col-span-3 space-y-4">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[#FFCC00]" />
              <h2 className="font-mono text-xs font-bold text-white tracking-widest uppercase">
                Forensic Modification Indicators
              </h2>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
              {[
                { name: "Metadata Stripped", active: item.modification_report?.metadata_stripped, color: "text-[#FF3366] border-[#FF3366]/30 bg-[#FF3366]/5" },
                {
                  name: item.modification_report?.compression_status === "LOW" ? "Compression Detected" : "Heavy Compression",
                  active: item.modification_report?.heavy_compression || item.modification_report?.compression_status === "LOW",
                  statusText: item.modification_report?.compression_status === "LOW" ? "LOW" : item.modification_report?.heavy_compression ? "Detected" : "Clean",
                  color: item.modification_report?.compression_status === "LOW" ? "text-[#FFCC00] border-[#FFCC00]/25 bg-[#FFCC00]/5" : "text-[#FFCC00] border-[#FFCC00]/30 bg-[#FFCC00]/5"
                },
                { name: "Low Resolution", active: item.modification_report?.low_resolution, color: "text-[#F59E0B] border-[#F59E0B]/30 bg-[#F59E0B]/5" },
                { name: "Manipulated Canvas", active: item.modification_report?.manipulation_indicator, color: "text-[#FF3366] border-[#FF3366]/30 bg-[#FF3366]/5" },
                { name: "Re-Encoded Software", active: item.modification_report?.re_encoded, color: "text-[#A78BFA] border-[#A78BFA]/30 bg-[#A78BFA]/5" },
                { name: "Crop Detected", active: item.modification_report?.cropping_detected, color: "text-[#00E5FF] border-[#00E5FF]/30 bg-[#00E5FF]/5" },
                { name: "Resize Detected", active: item.modification_report?.resizing_detected, color: "text-[#00E5FF] border-[#00E5FF]/30 bg-[#00E5FF]/5" },
                { name: "Watermark Overlay", active: item.modification_report?.watermark_detected, color: "text-[#FF3366] border-[#FF3366]/30 bg-[#FF3366]/5" }
              ].map((flag, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded border font-mono text-center flex flex-col justify-center gap-1 min-h-[65px] transition-colors ${flag.active
                      ? flag.color
                      : "border-[rgba(255,255,255,0.03)] bg-[#09090D]/20 text-gray-600"
                    }`}
                >
                  <span className="text-[9px] font-black uppercase leading-tight">{flag.name}</span>
                  <span className="text-[7px] uppercase font-bold tracking-wider">
                    {flag.statusText ? flag.statusText : flag.active ? "Detected" : "Clean"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Keyframes (For Videos) */}
          {isVideo && item.keyframes && item.keyframes.length > 0 && (
            <div className="cyber-card p-6 lg:col-span-3 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Film className="h-4 w-4 text-[#00E5FF]" />
                <h2 className="font-mono text-xs font-bold text-white tracking-widest uppercase">
                  Video Keyframe Index ({item.keyframes.length})
                </h2>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {item.keyframes.map((kf) => (
                  <div key={kf.id} className="bg-[#12121A] border border-[rgba(255,255,255,0.04)] rounded p-2 space-y-2">
                    <img
                      src={`${backendUrl}/media/uploads/${kf.filepath.split('/').pop()}`}
                      alt={`Keyframe ${kf.timestamp}`}
                      className="w-full h-24 object-cover rounded border border-[rgba(255,255,255,0.04)]"
                    />
                    <div className="font-mono text-[8px] text-gray-500 flex justify-between">
                      <span>Time: {kf.timestamp.toFixed(1)}s</span>
                      <span className="text-white truncate max-w-[80px]">{kf.phash}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Blind Investigation Clues Tab */}
      {activeTab === "clues" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fadeIn font-mono text-xs">
          {/* Text & Languages */}
          <div className="cyber-card p-5 space-y-4">
            <h3 className="font-mono text-xs font-bold text-[#00FF9D] uppercase tracking-widest border-b border-[rgba(255,255,255,0.05)] pb-2">
              Extracted Text & Language
            </h3>
            <div className="space-y-3 text-[11px] text-gray-400">
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">OCR Extraction Status</span>
                <span className="text-white font-semibold">{item.modification_report?.investigation_intelligence?.ocr_results ?? "Analysis Unavailable"}</span>
              </div>
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Extracted Text</span>
                <p className="bg-[#08080C] border border-[rgba(255,255,255,0.04)] p-3 rounded text-white italic max-h-[120px] overflow-y-auto mt-1 leading-relaxed">
                  {item.modification_report?.investigation_intelligence?.extracted_text || "No machine-readable text found."}
                </p>
              </div>
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Languages Identified</span>
                <div className="flex flex-wrap gap-2 mt-1.5">
                  {(item.modification_report?.investigation_intelligence?.languages_detected || ["None Detected"]).map((lang: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#00FF9D]/10 border border-[#00FF9D]/20 text-[9px] text-[#00FF9D] font-bold">
                      {lang}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Objects & Location */}
          <div className="cyber-card p-5 space-y-4">
            <h3 className="font-mono text-xs font-bold text-[#00FF9D] uppercase tracking-widest border-b border-[rgba(255,255,255,0.05)] pb-2">
              Object & Location Clues
            </h3>
            <div className="space-y-3 text-[11px] text-gray-400">
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Detected Objects (CLIP / Semantics)</span>
                <div className="flex flex-wrap gap-1.5 mt-1.5">
                  {(item.modification_report?.investigation_intelligence?.objects_detected || ["General visual subject"]).map((obj: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#1A1A24] border border-[rgba(255,255,255,0.08)] text-[9px] text-white">
                      {obj}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Estimated Scene Context</span>
                <p className="text-white font-medium mt-1">
                  {item.modification_report?.investigation_intelligence?.scene_description ?? "Analysis Unavailable"}
                </p>
              </div>
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Geographical Clues</span>
                <div className="flex flex-wrap gap-2 mt-1.5">
                  {(item.modification_report?.investigation_intelligence?.location_clues || ["Unspecified coordinates"]).map((loc: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#00E5FF]/10 border border-[#00E5FF]/20 text-[9px] text-[#00E5FF] font-bold">
                      {loc}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Narrative Summary */}
          <div className="cyber-card p-5 md:col-span-2 space-y-4 border-l-4 border-l-[#00FF9D]">
            <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest flex items-center justify-between">
              <span>INVESTIGATOR FORENSIC NARRATIVE</span>
              <span className="text-[10px] text-gray-500 font-normal font-mono">
                Evidence Confidence: {item.modification_report?.investigation_intelligence?.evidence_confidence !== undefined ? `${item.modification_report.investigation_intelligence.evidence_confidence}%` : "Not Evaluated"}
              </span>
            </h3>
            <p className="font-mono text-xs text-gray-300 leading-relaxed bg-[#08080C]/80 border border-[rgba(255,255,255,0.03)] p-4 rounded-lg">
              {item.modification_report?.investigation_narrative || "No narrative summary calculated."}
            </p>
          </div>
        </div>
      )}

      {/* AI & Stego Detection Tab */}
      {activeTab === "aistego" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 animate-fadeIn font-mono text-xs">
          {/* Steganography & Byte Diagnostics */}
          <div className="cyber-card p-5 space-y-4 border-t-2 border-t-[#FF3366]">
            <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.05)] pb-2">
              <h3 className="font-mono text-xs font-bold text-[#FF3366] uppercase tracking-widest flex items-center gap-1.5">
                <HardDrive className="h-4 w-4" />
                Steganography & Trailing Payloads
              </h3>
              <span className={`px-2 py-0.5 rounded text-[8px] font-black border uppercase ${item.modification_report?.forensic_investigation?.stego_detected
                  ? "bg-[#FF3366]/10 text-[#FF3366] border-[#FF3366]/20"
                  : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                }`}>
                {item.modification_report?.forensic_investigation?.stego_detected ? "SUSPICION CONFIRMED" : "CLEAN STRUCTURE"}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-[9px]">
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Avg Byte Entropy</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.forensic_investigation?.entropy ?? 0.0}
                </span>
              </div>
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Overlay Bytes</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.forensic_investigation?.overlay_bytes ?? 0} B
                </span>
              </div>
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Embedded Resources</span>
                <span className="block mt-1 text-[9px] font-black text-[#FF3366] truncate">
                  {(item.modification_report?.forensic_investigation?.embedded_resources || []).join(", ") || "None"}
                </span>
              </div>
            </div>

            <div className="space-y-2 text-[10px] text-gray-400">
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Supporting Stego Evidence</span>
                <ul className="list-disc list-inside mt-1 space-y-1 text-gray-300">
                  {(item.modification_report?.forensic_investigation?.supporting_evidence || []).map((ev: string, idx: number) => (
                    <li key={idx} className="leading-relaxed">{ev}</li>
                  )) || <li>No steganography supporting evidence.</li>}
                </ul>
              </div>
              {item.modification_report?.forensic_investigation?.contradicting_evidence?.length > 0 && (
                <div>
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Contradicting Evidence</span>
                  <ul className="list-disc list-inside mt-1 space-y-1 text-gray-400">
                    {item.modification_report.forensic_investigation.contradicting_evidence.map((ev: string, idx: number) => (
                      <li key={idx} className="leading-relaxed">{ev}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* AI Generation Forensics */}
          <div className="cyber-card p-5 space-y-4 border-t-2 border-t-[#FFCC00]">
            <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.05)] pb-2">
              <h3 className="font-mono text-xs font-bold text-[#FFCC00] uppercase tracking-widest flex items-center gap-1.5">
                <Cpu className="h-4.5 w-4.5" />
                AI Generation Detection
              </h3>
              <span className={`px-2 py-0.5 rounded text-[8px] font-black border uppercase ${(item.modification_report?.ai_detection?.probability ?? 0) >= 50
                  ? "bg-[#FFCC00]/10 text-[#FFCC00] border-[#FFCC00]/20"
                  : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                }`}>
                {(item.modification_report?.ai_detection?.probability ?? 0) >= 50 ? "AI SUSPECT" : "AUTHENTIC"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[9px]">
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">AI Probability</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.ai_detection?.probability ?? 0}%
                </span>
              </div>
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Analysis Confidence</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.ai_detection?.confidence ?? 50}%
                </span>
              </div>
            </div>

            <div className="space-y-2 text-[10px] text-gray-400">
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Supporting Generative Indicators</span>
                <ul className="list-disc list-inside mt-1 space-y-1 text-gray-300">
                  {(item.modification_report?.ai_detection?.supporting_evidence || []).map((ev: string, idx: number) => (
                    <li key={idx} className="leading-relaxed">{ev}</li>
                  )) || <li>No AI indicators detected.</li>}
                </ul>
              </div>
              {item.modification_report?.ai_detection?.contradicting_evidence?.length > 0 && (
                <div>
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Contradicting Evidence</span>
                  <ul className="list-disc list-inside mt-1 space-y-1 text-gray-400">
                    {item.modification_report.ai_detection.contradicting_evidence.map((ev: string, idx: number) => (
                      <li key={idx} className="leading-relaxed">{ev}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* CASIA Tampering Forensics */}
          <div className="cyber-card p-5 space-y-4 border-t-2 border-t-[#00E5FF]">
            <div className="flex justify-between items-center border-b border-[rgba(255,255,255,0.05)] pb-2">
              <h3 className="font-mono text-xs font-bold text-[#00E5FF] uppercase tracking-widest flex items-center gap-1.5">
                <Shield className="h-4.5 w-4.5" />
                CASIA Tampering Detection
              </h3>
              <span className={`px-2 py-0.5 rounded text-[8px] font-black border uppercase ${(item.modification_report?.casia_detection?.probability ?? 0) >= 50
                  ? "bg-[#FF3366]/10 text-[#FF3366] border-[#FF3366]/20"
                  : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                }`}>
                {item.modification_report?.casia_detection?.class ?? "AUTHENTIC"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[9px]">
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Tampering Probability</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.casia_detection?.probability ?? 0}%
                </span>
              </div>
              <div className="bg-[#12121A] p-2.5 rounded border border-[rgba(255,255,255,0.03)] text-center">
                <span className="block text-gray-500 uppercase">Analysis Confidence</span>
                <span className="block mt-1 text-sm font-black text-white">
                  {item.modification_report?.forensic_findings?.find((f: any) => f.finding === "CASIA Tampering")?.confidence ?? 80}%
                </span>
              </div>
            </div>

            <div className="space-y-2 text-[10px] text-gray-400">
              <div>
                <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Forensic Evidence & Verdict</span>
                <ul className="list-disc list-inside mt-1 space-y-1 text-gray-300">
                  {(item.modification_report?.forensic_findings?.find((f: any) => f.finding === "CASIA Tampering")?.evidence || [
                    (item.modification_report?.casia_detection?.probability ?? 0) >= 50
                      ? `EfficientNet-B0 CASIA classifier identified tampering patterns with ${item.modification_report?.casia_detection?.probability ?? 0}% probability.`
                      : `EfficientNet-B0 CASIA classifier verified authentic structure with ${100 - (item.modification_report?.casia_detection?.probability ?? 0)}% confidence.`
                  ]).map((ev: string, idx: number) => (
                    <li key={idx} className="leading-relaxed">{ev}</li>
                  ))}
                </ul>
              </div>
              {item.modification_report?.forensic_findings?.find((f: any) => f.finding === "CASIA Tampering")?.alternative_explanation && (
                <div>
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Alternative Explanation</span>
                  <p className="mt-1 text-gray-400 leading-relaxed font-mono">
                    {item.modification_report?.forensic_findings?.find((f: any) => f.finding === "CASIA Tampering")?.alternative_explanation}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Case Clustering & Lineage Tab Layout */}
      {activeTab === "lineage" && (
        <>
          {/* Prominent Primary Investigation Widgets */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Card 1: Probable Origin Assessment */}
            <div className="cyber-card p-6 flex flex-col justify-between md:col-span-1 border-l-4 border-l-[#7C3AED] bg-gradient-to-br from-[#0C0C14] to-[#0A0A0A] shadow-2xl">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Fingerprint className="h-5 w-5 text-[#A78BFA]" />
                  <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">Most Probable Origin</span>
                </div>
                <div>
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Origin File Asset</span>
                  <span className="text-sm font-black text-white break-all block mt-1">
                    {item.modification_report?.relationship_analysis?.probable_origin_asset ?? "Uncalculated Baseline"}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-4 pt-2 border-t border-[rgba(255,255,255,0.03)]">
                  <div>
                    <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Selection Confidence</span>
                    <span className="text-xl font-bold font-mono text-[#00E5FF] block mt-0.5">
                      {item.modification_report?.relationship_analysis?.origin_confidence ?? "N/A"}%
                    </span>
                  </div>
                  <div>
                    <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest">Probability Score</span>
                    <span className="text-xl font-bold font-mono text-[#00FF9D] block mt-0.5">
                      {item.modification_report?.relationship_analysis?.origin_probability ?? "N/A"}%
                    </span>
                  </div>
                </div>
                <div className="pt-2 border-t border-[rgba(255,255,255,0.03)]">
                  <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest mb-1">Origin Status</span>
                  <span className={`inline-block px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-widest border ${item.modification_report?.relationship_analysis?.origin_undetermined
                      ? "bg-[#FFCC00]/10 text-[#FFCC00] border-[#FFCC00]/20"
                      : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                    }`}>
                    {item.modification_report?.relationship_analysis?.origin_undetermined ? "Undetermined / Probabilistic Fallback" : "Determined Baseline"}
                  </span>
                </div>
              </div>
            </div>

            {/* Card 2: Origin Selection Audit Trail */}
            <div className="cyber-card p-6 flex flex-col justify-between md:col-span-1 border-l-4 border-l-[#00E5FF] bg-gradient-to-br from-[#0C0C14] to-[#0A0A0A] shadow-2xl">
              <div className="space-y-3">
                <div className="flex items-center gap-2 border-b border-[rgba(255,255,255,0.05)] pb-2">
                  <Activity className="h-5 w-5 text-[#00E5FF]" />
                  <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">Origin Selection Audit Trail</span>
                </div>
                <div className="space-y-2 font-mono text-[9px]">
                  {[
                    { label: "Resolution (40%)", key: "resolution_contribution", color: "bg-[#00E5FF]" },
                    { label: "Metadata Richness (20%)", key: "metadata_contribution", color: "bg-[#7C3AED]" },
                    { label: "Compression Quality (20%)", key: "compression_contribution", color: "bg-[#00FF9D]" },
                    { label: "File Fidelity (15%)", key: "fidelity_contribution", color: "bg-[#FFCC00]" },
                    { label: "Chronology (5%)", key: "chronology_contribution", color: "bg-[#FF3366]" }
                  ].map((audit) => {
                    const val = item.modification_report?.relationship_analysis?.origin_audit_trail?.[audit.key] ?? 0;
                    return (
                      <div key={audit.key} className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-gray-400">{audit.label}:</span>
                          <span className="text-white font-bold">{val}%</span>
                        </div>
                        <div className="w-full bg-[#1A1A24] h-1.5 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${audit.color}`} style={{ width: `${(val / 100) * 200}%` }}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Card 3: Investigation Confidence & Narrative */}
            <div className="cyber-card p-6 flex flex-col justify-between md:col-span-1 border-l-4 border-l-[#00FF9D] bg-gradient-to-br from-[#0C0C14] to-[#0A0A0A] shadow-2xl">
              <div className="space-y-3 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-start flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      <Award className="h-5 w-5 text-[#00FF9D]" />
                      <span className="font-mono text-xs font-bold text-white uppercase tracking-wider">Investigation Confidence</span>
                    </div>
                    <span className={`inline-block px-1.5 py-0.5 rounded text-[8px] font-black border uppercase ${item.modification_report?.overall_investigation_confidence?.sufficiency === "Strong"
                        ? "bg-[#00FF9D]/10 text-[#00FF9D] border-[rgba(0,255,157,0.3)]"
                        : item.modification_report?.overall_investigation_confidence?.sufficiency === "Moderate"
                          ? "bg-[#F59E0B]/10 text-[#F59E0B] border-[rgba(245,158,11,0.3)]"
                          : "bg-[#FF3366]/10 text-[#FF3366] border-[rgba(255,51,102,0.3)]"
                      }`}>
                      Sufficiency: {item.modification_report?.overall_investigation_confidence?.sufficiency ?? "Not Evaluated"}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2 mt-2">
                    <span className="text-3xl font-extrabold text-white tracking-tight">
                      {item.modification_report?.overall_investigation_confidence?.score !== undefined
                        ? `${item.modification_report.overall_investigation_confidence.score}%`
                        : item.modification_report?.executive_summary?.confidence_score !== undefined
                          ? `${item.modification_report.executive_summary.confidence_score}%`
                          : "Not Evaluated"}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[8px] font-black uppercase border ${getConfidenceBadgeColor(
                      item.modification_report?.overall_investigation_confidence?.level ?? "Not Evaluated"
                    )}`}>
                      {item.modification_report?.overall_investigation_confidence?.level ?? "Not Evaluated"}
                    </span>
                  </div>
                </div>
                <div className="border-t border-[rgba(255,255,255,0.05)] pt-2 mt-2 font-mono text-[9px] text-gray-400">
                  <span className="block text-[8px] text-gray-500 uppercase font-semibold">Narrative Summary</span>
                  <p className="line-clamp-3 leading-relaxed mt-1">
                    {item.modification_report?.investigation_narrative ?? "No narrative summary calculated."}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Grid: Details, Forensics & Graph */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* DNA Profile Column */}
            <div className="cyber-card p-6 space-y-6 lg:col-span-1">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Fingerprint className="h-4 w-4 text-[#00E5FF]" />
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Media DNA Profile
                </h2>
              </div>

              <div className="space-y-4 font-mono text-xs text-gray-400">
                <div>
                  <span className="block text-[8px] uppercase tracking-widest text-gray-500">SHA256 CHECKSUM</span>
                  <span className="block break-all text-white bg-[#14141A] p-2 rounded border border-[rgba(255,255,255,0.03)] mt-1 text-[10px] select-all font-mono">
                    {item.sha256}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)]">
                    <span className="block text-[7px] text-gray-500 uppercase tracking-widest">pHash</span>
                    <span className="block mt-0.5 text-[10px] font-bold text-white truncate">{item.phash || "N/A"}</span>
                  </div>
                  <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)]">
                    <span className="block text-[7px] text-gray-500 uppercase tracking-widest">dHash</span>
                    <span className="block mt-0.5 text-[10px] font-bold text-white truncate">{item.dhash || "N/A"}</span>
                  </div>
                  <div className="bg-[#121212] p-2 rounded border border-[rgba(255,255,255,0.03)]">
                    <span className="block text-[7px] text-gray-500 uppercase tracking-widest">aHash</span>
                    <span className="block mt-0.5 text-[10px] font-bold text-white truncate">{item.ahash || "N/A"}</span>
                  </div>
                </div>

                {/* Technical Parameters */}
                <div className="border-t border-[rgba(255,255,255,0.05)] pt-4 space-y-2 text-[10px]">
                  <div className="flex justify-between">
                    <span>Dimensions:</span>
                    <span className="text-white font-bold">{item.resolution || "Unknown"}</span>
                  </div>
                  {isVideo && (
                    <div className="flex justify-between">
                      <span>Duration:</span>
                      <span className="text-white font-bold">{item.duration?.toFixed(2)} seconds</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span>Camera EXIF Info:</span>
                    <span className="text-white font-bold">
                      {item.metadata_sig?.exif?.Model || "None / Striped"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Refined Explicit Anomaly Flags */}
            <div className="cyber-card p-6 space-y-6 lg:col-span-1">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Shield className="h-4 w-4 text-[#7C3AED]" />
                <h2 className="font-mono text-xs font-bold text-[#7C3AED] tracking-widest uppercase">
                  Modification Status Flags
                </h2>
              </div>

              <div className="grid grid-cols-2 gap-4 text-center font-mono">
                <div className="bg-[#121212] p-3 rounded border border-[rgba(255,255,255,0.03)]">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Integrity Score</span>
                  <span className="block mt-1 text-2xl font-black text-[#00FF9D] glow-text-green">
                    {item.integrity_score}
                  </span>
                </div>
                <div className="bg-[#121212] p-3 rounded border border-[rgba(255,255,255,0.03)]">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Risk Assessment</span>
                  <span className={`block mt-1 text-2xl font-black ${item.risk_score > 50 ? "text-[#FF3366]" : "text-[#00FF9D]"}`}>
                    {item.risk_score}
                  </span>
                </div>
              </div>

              {/* Explicit Checklist Flags */}
              <div className="border-t border-[rgba(255,255,255,0.05)] pt-4 space-y-2 font-mono text-[10px]">
                <span className="block text-[8px] uppercase tracking-widest text-gray-500 mb-1">Checks & Flags</span>

                {[
                  { label: "Metadata Removed", key: "metadata_stripped" },
                  { label: "Crop Detected", key: "cropping_detected" },
                  { label: "Resize Detected", key: "resizing_detected" },
                  { label: "Re-encoding", key: "re_encoded" },
                  { label: "Compression Changes", key: "heavy_compression" },
                  { label: "Watermark Indicators", key: "watermark_detected" }
                ].map((flag) => {
                  const val = item.modification_report?.[flag.key] ?? false;
                  return (
                    <div
                      key={flag.key}
                      className={`flex justify-between items-center px-2 py-1 rounded border transition-colors ${val
                          ? "bg-[rgba(255,51,102,0.05)] text-[#FF3366] border-[rgba(255,51,102,0.15)]"
                          : "bg-[rgba(0,255,157,0.02)] text-gray-400 border-[rgba(255,255,255,0.04)]"
                        }`}
                    >
                      <span className="uppercase">{flag.label}</span>
                      <span className="font-bold">{val ? "DETECTED" : "CLEAN / UNCHANGED"}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Probable Evolution Chain (Directed hierarchical lineage graph) */}
            <div className="cyber-card p-6 space-y-4 lg:col-span-1">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-2 flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <Share2 className="h-4 w-4 text-[#00FF9D]" />
                  <h2 className="font-mono text-xs font-bold text-[#00FF9D] tracking-widest uppercase">
                    Probable Evolution Chain
                  </h2>
                </div>
              </div>
              {renderSVGGraph()}
            </div>
          </div>

          {/* Chronological Estimated Timeline block */}
          {graphData && graphData.nodes && graphData.nodes.length > 0 && (
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex justify-between items-center gap-2 flex-wrap font-mono">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-[#00E5FF]" />
                  <h2 className="text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                    Estimated Timeline
                  </h2>
                </div>
                {graphData.timeline_confidence !== undefined && (
                  <span className={`px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-wider border ${graphData.timeline_inconclusive
                      ? "bg-[#FF3366]/10 text-[#FF3366] border-[#FF3366]/20"
                      : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                    }`}>
                    Timeline Confidence: {graphData.timeline_confidence}% ({graphData.timeline_inconclusive ? "INCONCLUSIVE" : "VALIDATED"})
                  </span>
                )}
              </div>

              {graphData.timeline_inconclusive ? (
                <div className="rounded border border-[rgba(255,51,102,0.15)] bg-[rgba(255,51,102,0.02)] p-6 font-mono text-center space-y-2">
                  <AlertCircle className="h-8 w-8 text-[#FF3366] mx-auto animate-pulse" />
                  <h3 className="text-white font-bold text-xs uppercase tracking-wide">Evolution Sequence Inconclusive</h3>
                  <p className="text-[10px] text-gray-500 max-w-xl mx-auto leading-relaxed">
                    Timeline reconstruction confidence is insufficient (<span className="text-[#FF3366] font-bold">{graphData.timeline_confidence}%</span>, which is below the 40% validation threshold). Missing camera metadata tags, uniform dimensions, or weak compression delta signals prevent a defensible evolutionary sequence mapping.
                  </p>
                </div>
              ) : (
                <div className="relative flex flex-col md:flex-row items-center justify-between p-4 rounded-lg bg-[#08080C] border border-[rgba(255,255,255,0.04)] gap-6 overflow-x-auto">
                  {graphData.nodes.map((node, index) => {
                    const isOrigin = node.type === "original";
                    const isCurrent = node.id === item.id;

                    return (
                      <div key={node.id} className="flex flex-col md:flex-row items-center gap-4 flex-shrink-0">
                        <div
                          onClick={() => {
                            if (node.id !== item.id) {
                              router.push(`/media/${node.id}`);
                            }
                          }}
                          className={`p-3 rounded border text-center font-mono cursor-pointer transition-all w-40 min-h-[100px] flex flex-col justify-between ${isCurrent
                              ? "border-[#00E5FF] bg-[rgba(0,229,255,0.05)] shadow-[0_0_8px_rgba(0,229,255,0.2)]"
                              : isOrigin
                                ? "border-[#7C3AED] bg-[rgba(124,58,237,0.05)]"
                                : "border-[rgba(255,255,255,0.06)] bg-[#111115] hover:bg-[#181822]"
                            }`}
                        >
                          <div className="space-y-1">
                            <span className={`block text-[8px] uppercase tracking-wider font-bold ${isOrigin ? "text-[#A78BFA]" : "text-gray-500"
                              }`}>
                              {isOrigin ? "Most Probable Origin" : `Step ${index}`}
                            </span>
                            <span className="block text-[10px] text-white truncate font-semibold px-1" title={node.label}>
                              {node.label}
                            </span>
                          </div>
                          <div className="space-y-1 border-t border-[rgba(255,255,255,0.04)] pt-2 mt-2 text-[9px] text-gray-500">
                            <div className="flex justify-between">
                              <span>Integrity:</span>
                              <span className="text-[#00FF9D] font-bold">{node.integrity}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Format:</span>
                              <span className="text-gray-400 font-bold truncate max-w-[50px]">{node.mime_type.split("/")[1]}</span>
                            </div>
                          </div>
                        </div>

                        {index < graphData.nodes.length - 1 && (
                          <div className="flex items-center justify-center text-[#00E5FF]">
                            <ArrowRight className="h-5 w-5 transform rotate-90 md:rotate-0" />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Upgraded Investigation Insights with Evidence Matrices */}
          {item.modification_report?.investigation_insights && (
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Info className="h-4 w-4 text-[#00E5FF]" />
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Investigation Insights
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(item.modification_report.investigation_insights).map(([key, insight]: any) => {
                  const getInsightTitle = (k: string, ins: any) => {
                    if (ins && ins.title) return ins.title;
                    const defaults: Record<string, string> = {
                      possible_redistribution: "Platform Redistribution Risk",
                      possible_social_media_repost: "Likely Social Media Repost",
                      possible_messaging_app_recompression: "Messaging App Redistribution Indicators",
                      screenshot_indicators: "Screenshot Indicators",
                      content_stability_assessment: "Content Frame Stability"
                    };
                    return defaults[k] || k.replace(/_/g, " ").toUpperCase();
                  };

                  const titleText = getInsightTitle(key, insight);
                  const statusText = typeof insight === "string" ? "Detected" : (insight?.status || "Detected");
                  const explanationText = typeof insight === "string" ? insight : (insight?.explanation || "");
                  const levelText = typeof insight === "string" ? "High" : (insight?.level || "High");
                  const confidenceText = typeof insight === "string" ? 80 : (insight?.confidence || 50);
                  const badgeColor = getConfidenceBadgeColor(levelText);
                  const evidenceMatrix = typeof insight === "string" ? null : insight?.evidence_matrix;

                  let statusBadgeColor = "bg-[#FFCC00]/10 text-[#FFCC00] border-[#FFCC00]/20";
                  if (statusText === "Detected") {
                    statusBadgeColor = "bg-[#FF3366]/10 text-[#FF3366] border-[rgba(255,51,102,0.3)]";
                  } else if (statusText === "Not Detected") {
                    statusBadgeColor = "bg-[#00FF9D]/10 text-[#00FF9D] border-[rgba(0,255,157,0.3)]";
                  }

                  return (
                    <div
                      key={key}
                      className="border border-[rgba(255,255,255,0.04)] bg-[#07070A]/50 p-4 rounded-lg space-y-3 font-mono flex flex-col justify-between"
                    >
                      <div className="space-y-2 font-mono text-xs">
                        <div className="flex justify-between items-center flex-wrap gap-2">
                          <span className="text-xs font-bold text-cyan-300 uppercase">{titleText}</span>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className={`px-2 py-0.5 rounded text-[8px] font-black tracking-widest uppercase border ${statusBadgeColor}`}>
                              {statusText}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-[8px] font-black tracking-widest uppercase border ${badgeColor}`}>
                              {levelText} ({confidenceText}%)
                            </span>
                            {insight?.evidence_sufficiency && (
                              <span className={`px-2 py-0.5 rounded text-[8px] font-black tracking-widest uppercase border ${insight.evidence_sufficiency === "Strong"
                                  ? "bg-[#00FF9D]/10 text-[#00FF9D] border-[rgba(0,255,157,0.3)]"
                                  : insight.evidence_sufficiency === "Moderate"
                                    ? "bg-[#F59E0B]/10 text-[#F59E0B] border-[rgba(245,158,11,0.3)]"
                                    : "bg-[#FF3366]/10 text-[#FF3366] border-[rgba(255,51,102,0.3)]"
                                }`}>
                                SUFFICIENCY: {insight.evidence_sufficiency}
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="text-[10px] text-gray-300 leading-relaxed">
                          {explanationText}
                        </p>
                        {insight?.alternative_explanation && (
                          <div className="mt-2 text-[8px] text-gray-500 italic leading-relaxed">
                            <span className="font-bold uppercase not-italic">Alt. Explanation:</span> {insight.alternative_explanation}
                          </div>
                        )}
                        {insight?.evidence && insight.evidence.length > 0 && (
                          <div className="mt-2 space-y-1">
                            <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-semibold">Supporting Evidence</span>
                            <ul className="list-disc pl-3 text-[9px] text-gray-400 space-y-0.5">
                              {insight.evidence.map((evItem: string, idx: number) => (
                                <li key={idx}>{evItem}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Evidence Matrix */}
                      {evidenceMatrix && (
                        <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 mt-3">
                          <span className="block text-[8px] text-gray-400 uppercase tracking-widest mb-1.5 font-bold">Evidence Matrix</span>
                          <div className="grid grid-cols-2 gap-2 text-[9px]">
                            {Object.entries(evidenceMatrix).map(([mKey, mVal]: any) => (
                              <div key={mKey} className="flex justify-between items-center bg-[#111116] px-2 py-1 rounded border border-[rgba(255,255,255,0.02)]">
                                <span className="text-gray-400 capitalize">{mKey.replace(/_/g, " ")}:</span>
                                <span className="text-gray-400 font-bold">
                                  {typeof mVal === "boolean" ? (mVal ? "Yes" : "No") : mVal !== null ? String(mVal) : "—"}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Investigator View Panel */}
          <div className="cyber-card p-6 space-y-6">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
              <Eye className="h-4 w-4 text-[#00FF9D]" />
              <h2 className="font-mono text-xs font-bold text-[#00FF9D] tracking-widest uppercase">
                Investigator View
              </h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
              {/* Observations */}
              <div className="md:col-span-1 space-y-2 font-mono text-[10px] text-gray-400 bg-[#0A0A0E] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="block text-[8px] text-[#00E5FF] font-black uppercase tracking-widest border-b border-[rgba(0,229,255,0.1)] pb-1 mb-2">Observations</span>
                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <span>MIME:</span>
                    <span className="text-white font-bold">{item.mime_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Resolution:</span>
                    <span className="text-white font-bold">{item.resolution || "Unknown"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Size:</span>
                    <span className="text-white font-bold">{(item.file_size / 1024).toFixed(1)} KB</span>
                  </div>
                  <div className="border-t border-[rgba(255,255,255,0.04)] pt-2 mt-2">
                    <span className="block text-[7px] uppercase tracking-wider text-gray-500 mb-1">Active Anomalies:</span>
                    {[
                      { l: "Metadata Stripped", k: "metadata_stripped" },
                      { l: "Crop", k: "cropping_detected" },
                      { l: "Resize", k: "resizing_detected" },
                      { l: "Re-encoding", k: "re_encoded" },
                      { l: "Compression", k: "heavy_compression" },
                      { l: "Watermark", k: "watermark_detected" }
                    ].map(f => {
                      const val = item.modification_report?.[f.k] ?? false;
                      if (!val) return null;
                      return (
                        <div key={f.k} className="text-[#FF3366] font-bold text-[8px] uppercase">
                          • {f.l}
                        </div>
                      );
                    })}
                    {!item.modification_report?.metadata_stripped &&
                      !item.modification_report?.cropping_detected &&
                      !item.modification_report?.resizing_detected &&
                      !item.modification_report?.re_encoded &&
                      !item.modification_report?.heavy_compression &&
                      !item.modification_report?.watermark_detected && (
                        <span className="text-[#00FF9D] text-[8px] font-bold">No modification flags</span>
                      )}
                  </div>
                </div>
              </div>

              {/* Evidence */}
              <div className="md:col-span-1 space-y-2 font-mono text-[10px] text-gray-400 bg-[#0A0A0E] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="block text-[8px] text-[#7C3AED] font-black uppercase tracking-widest border-b border-[rgba(124,58,237,0.1)] pb-1 mb-2">Measurable Evidence</span>
                <div className="space-y-1.5">
                  <div className="flex justify-between">
                    <span>EXIF Tags:</span>
                    <span className="text-white font-bold">{item.metadata_sig?.exif ? Object.keys(item.metadata_sig.exif).length : 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>JPEG Quality:</span>
                    <span className="text-white font-bold">{item.metadata_sig?.jpeg_quality ?? "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Blockiness:</span>
                    <span className="text-white font-bold">{item.metadata_sig?.blockiness?.toFixed(2) ?? "1.00"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>pHash Bits:</span>
                    <span className="text-white font-bold truncate max-w-[50px]">{item.phash ? "64-bit" : "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Clip Embed:</span>
                    <span className="text-white font-bold">{item.embedding ? "512-dim" : "N/A"}</span>
                  </div>
                </div>
              </div>

              {/* Conclusions */}
              <div className="md:col-span-1 space-y-2 font-mono text-[10px] text-gray-400 bg-[#0A0A0E] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="block text-[8px] text-[#00FF9D] font-black uppercase tracking-widest border-b border-[rgba(0,255,157,0.1)] pb-1 mb-2">Conclusions</span>
                <div className="space-y-1.5 text-white">
                  <span className="block text-xs font-black text-[#00FF9D]">
                    {item.modification_report?.asset_classification ?? "Unknown"}
                  </span>
                  <p className="text-[9px] text-gray-400 mt-2 leading-relaxed">
                    Derived classification type based on comparison with estimated origin.
                  </p>
                </div>
              </div>

              {/* Confidence */}
              <div className="md:col-span-1 space-y-2 font-mono text-[10px] text-gray-400 bg-[#0A0A0E] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="block text-[8px] text-[#00E5FF] font-black uppercase tracking-widest border-b border-[rgba(0,229,255,0.1)] pb-1 mb-2">Confidence & Sufficiency</span>
                <div className="space-y-2">
                  <div>
                    <span className="block text-[8px] text-gray-500">INVESTIGATION CONFIDENCE</span>
                    <span className="text-sm font-black text-white">
                      {item.modification_report?.overall_investigation_confidence?.score !== undefined
                        ? `${item.modification_report.overall_investigation_confidence.score}%`
                        : "Not Evaluated"}
                    </span>
                    <span className="block text-[8px] text-gray-500 text-[7px]">LEVEL: {item.modification_report?.overall_investigation_confidence?.level ?? "Not Evaluated"}</span>
                  </div>
                  <div className="border-t border-[rgba(255,255,255,0.04)] pt-2 mt-2">
                    <span className="block text-[8px] text-gray-500">EVIDENCE SUFFICIENCY</span>
                    <span className={`inline-block px-1.5 py-0.5 rounded text-[8px] font-black border uppercase mt-1 ${item.modification_report?.overall_investigation_confidence?.sufficiency === "Strong"
                        ? "bg-[#00FF9D]/10 text-[#00FF9D] border-[rgba(0,255,157,0.3)]"
                        : item.modification_report?.overall_investigation_confidence?.sufficiency === "Moderate"
                          ? "bg-[#F59E0B]/10 text-[#F59E0B] border-[rgba(245,158,11,0.3)]"
                          : "bg-[#FF3366]/10 text-[#FF3366] border-[rgba(255,51,102,0.3)]"
                      }`}>
                      {item.modification_report?.overall_investigation_confidence?.sufficiency ?? "Not Evaluated"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Limitations */}
              <div className="md:col-span-1 space-y-2 font-mono text-[10px] text-gray-400 bg-[#0A0A0E] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                <span className="block text-[8px] text-[#FF3366] font-black uppercase tracking-widest border-b border-[rgba(255,51,102,0.1)] pb-1 mb-2">Limitations</span>
                <div className="space-y-1.5 text-[9px]">
                  {!item.metadata_sig?.exif && (
                    <div className="text-[#FF3366]">• Original capture hardware and time are anonymous (EXIF absent).</div>
                  )}
                  {(!graphData || graphData.nodes.length <= 1) && (
                    <div className="text-[#FF3366]">• Standing baseline unverified without visual duplicates cluster.</div>
                  )}
                  <div className="text-gray-500">• Max origin estimation confidence capped at 98%.</div>
                  <div className="text-gray-500">• Forensics filename-agnostic.</div>
                </div>
              </div>
            </div>
          </div>

          {/* Blind Forensic Investigation Report Card */}
          {item.modification_report && (
            <div className="cyber-card p-6 space-y-6">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <FileText className="h-4 w-4 text-[#00E5FF]" />
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Detailed Diagnostic Findings
                </h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Executive Summary & Confidence Factors */}
                <div className="lg:col-span-5 space-y-6">
                  <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-4">
                    <div>
                      <span className="block text-[8px] text-gray-500 uppercase tracking-widest">EXECUTIVE SUMMARY</span>
                      <p className="text-xs text-white leading-relaxed mt-1">
                        {item.modification_report.executive_summary?.findings}
                      </p>
                    </div>
                    {item.modification_report.executive_summary?.conclusions?.length > 0 && (
                      <div>
                        <span className="block text-[8px] text-gray-500 uppercase tracking-widest">KEY CONCLUSIONS</span>
                        <ul className="list-disc pl-4 mt-1 space-y-1 text-xs text-gray-300">
                          {item.modification_report.executive_summary.conclusions.map((conclusion: string, idx: number) => (
                            <li key={idx}>{conclusion}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Overall Confidence & Factors */}
                    <div className="border-t border-[rgba(255,255,255,0.06)] pt-4">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-[9px] text-gray-400 uppercase font-black">Confidence Factor Weights</span>
                        <span className="text-xl font-black text-[#00E5FF] glow-text-cyan">
                          {item.modification_report.executive_summary?.confidence_score}%
                        </span>
                      </div>

                      <div className="space-y-2.5">
                        {item.modification_report.executive_summary?.confidence_factors &&
                          Object.entries(item.modification_report.executive_summary.confidence_factors).map(([factor, score]: any) => (
                            <div key={factor} className="space-y-1">
                              <div className="flex justify-between text-[9px]">
                                <span className="text-gray-500 capitalize">{factor.replace(/_/g, " ")}</span>
                                <span className="text-[#00FF9D] font-bold">{score}%</span>
                              </div>
                              <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-[#00FF9D] rounded-full" style={{ width: `${score}%` }}></div>
                              </div>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                  </div>

                  {/* Relationship Analysis details */}
                  {item.modification_report.relationship_analysis && (
                    <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-3">
                      <span className="block text-[8px] text-[#00FF9D] uppercase tracking-widest">Visual Clustering Details</span>

                      <div className="grid grid-cols-2 gap-4 text-[10px]">
                        <div>
                          <span className="block text-[8px] text-gray-500">Related Assets</span>
                          <span className="text-white font-bold">{item.modification_report.relationship_analysis.related_assets_count}</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Classification Type</span>
                          <span className="text-white font-bold">{item.modification_report.relationship_analysis.relationship_type}</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Probable Origin File</span>
                          <span className="text-white font-bold truncate block" title={item.modification_report.relationship_analysis.probable_origin_asset}>
                            {item.modification_report.relationship_analysis.probable_origin_asset}
                          </span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Origin Sim Confidence</span>
                          <span className="text-[#00FF9D] font-bold">{item.modification_report.relationship_analysis.confidence_score}%</span>
                        </div>
                        {item.modification_report.relationship_analysis.origin_confidence !== undefined && (
                          <>
                            <div>
                              <span className="block text-[8px] text-gray-500">Origin Confidence</span>
                              <span className="text-[#00E5FF] font-bold">{item.modification_report.relationship_analysis.origin_confidence}%</span>
                            </div>
                            <div>
                              <span className="block text-[8px] text-gray-500">Origin Probability</span>
                              <span className="text-[#00FF9D] font-bold">{item.modification_report.relationship_analysis.origin_probability}%</span>
                            </div>
                            <div className="col-span-2">
                              <span className="block text-[8px] text-gray-500">Origin Selection Status</span>
                              <span className={`px-1.5 py-0.5 rounded text-[8px] font-black uppercase tracking-widest border ${item.modification_report.relationship_analysis.origin_undetermined
                                  ? "bg-[#FFCC00]/10 text-[#FFCC00] border-[#FFCC00]/20"
                                  : "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20"
                                }`}>
                                {item.modification_report.relationship_analysis.origin_undetermined ? "Undetermined / Probabilistic" : "Determined Baseline"}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Technical Profile & Forensic Findings list */}
                <div className="lg:col-span-7 space-y-6">
                  {/* Technical Profile Table */}
                  <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-3">
                    <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Technical Profile</span>
                    {item.modification_report.technical_profile && (
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs">
                        <div>
                          <span className="block text-[8px] text-gray-500">Resolution</span>
                          <span className="text-white font-bold">{item.modification_report.technical_profile.resolution}</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Format</span>
                          <span className="text-white font-bold">{item.modification_report.technical_profile.format}</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">File Size</span>
                          <span className="text-white font-bold">{item.modification_report.technical_profile.file_size?.toLocaleString()} bytes</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">EXIF Status</span>
                          <span className="text-white font-bold">{item.modification_report.technical_profile.exif_status}</span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Est. JPEG Quality</span>
                          <span className="text-white font-bold">
                            {item.modification_report.technical_profile.compression_indicators?.jpeg_quality !== null
                              ? `${item.modification_report.technical_profile.compression_indicators?.jpeg_quality}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div>
                          <span className="block text-[8px] text-gray-500">Blockiness Index</span>
                          <span className="text-white font-bold">
                            {item.modification_report.technical_profile.compression_indicators?.blockiness?.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Detailed Forensic Findings */}
                  <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-4">
                    <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Detailed Forensic Findings</span>

                    <div className="space-y-4">
                      {item.modification_report.forensic_findings?.map((finding: any, idx: number) => {
                        let statusColor = "bg-[#FFCC00]/10 text-[#FFCC00] border-[#FFCC00]/20";
                        if (finding.status === "Detected") {
                          statusColor = "bg-[#FF3366]/10 text-[#FF3366] border-[#FF3366]/20";
                        } else if (finding.status === "Not Detected") {
                          statusColor = "bg-[#00FF9D]/10 text-[#00FF9D] border-[#00FF9D]/20";
                        }

                        return (
                          <div key={idx} className="border border-[rgba(255,255,255,0.03)] bg-[#07070A]/50 p-3 rounded space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-bold text-white uppercase">{finding.finding}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] text-gray-500">{finding.confidence}% confidence</span>
                                <span className={`px-2 py-0.5 rounded text-[8px] font-black tracking-widest uppercase border ${statusColor}`}>
                                  {finding.status}
                                </span>
                              </div>
                            </div>

                            {finding.evidence?.length > 0 && (
                              <div className="space-y-1">
                                <span className="block text-[7px] text-gray-500 uppercase">Supporting Evidence</span>
                                <ul className="list-disc pl-4 text-[10px] text-gray-400 space-y-0.5">
                                  {finding.evidence.map((ev: string, eIdx: number) => (
                                    <li key={eIdx}>{ev}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Interactive pHash visualization */}
          {phashSteps && (
            <div className="cyber-card p-6 space-y-6">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 text-center">
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Perceptual Hashing Step-by-Step Generation
                </h2>
                <span className="block text-[8px] font-mono text-gray-500 uppercase tracking-widest mt-0.5">
                  Educational Diagnostics
                </span>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* Steps Stepper */}
                <div className="lg:col-span-2 space-y-2">
                  {stepLabels.map((step) => (
                    <button
                      key={step.id}
                      onClick={() => setActiveVisualizerStep(step.id)}
                      className={`w-full text-left p-3 rounded border font-mono transition-all flex items-center gap-3 ${activeVisualizerStep === step.id
                          ? "border-[#00E5FF] bg-[#00E5FF]/5 shadow-[0_0_8px_rgba(0,229,255,0.2)]"
                          : "border-[rgba(255,255,255,0.04)] bg-[#111] hover:bg-[#151515]"
                        }`}
                    >
                      <div className={`h-6 w-6 rounded flex items-center justify-center text-xs font-bold ${activeVisualizerStep === step.id ? "bg-[#00E5FF] text-black" : "bg-[#1F1F1F] text-gray-400"
                        }`}>
                        {step.id}
                      </div>
                      <div>
                        <span className="block text-xs font-bold text-white">{step.name}</span>
                        <span className="block text-[9px] text-gray-500 mt-0.5">{step.desc}</span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Steps Display Panel */}
                <div className="lg:col-span-3 flex flex-col justify-center items-center border border-[rgba(255,255,255,0.04)] bg-[#0C0C12] rounded p-6 min-h-[300px]">
                  <span className="block font-mono text-[9px] uppercase tracking-widest text-[#00FF9D] mb-4">
                    Step {activeVisualizerStep}: {stepLabels[activeVisualizerStep - 1].name}
                  </span>

                  <div className="relative border border-[rgba(255,255,255,0.06)] bg-[#07070A] rounded p-2 max-w-[280px]">
                    <img
                      src={(phashSteps as any)[`step${activeVisualizerStep}`]}
                      alt={`pHash Stage ${activeVisualizerStep}`}
                      className="max-h-[250px] object-contain w-full rounded"
                    />
                  </div>

                  {activeVisualizerStep === 6 && (
                    <div className="mt-4 font-mono text-[10px] text-gray-400 text-center">
                      <span>Resulting Perceptual Hash String: </span>
                      <span className="block text-[#00E5FF] font-black text-sm glow-text-cyan select-all mt-1">
                        {phashSteps.hash}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Video Keyframe Panel (if Video) */}
          {isVideo && item.keyframes && item.keyframes.length > 0 && (
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Film className="h-4 w-4 text-[#00E5FF]" />
                <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                  Keyframe pHash Timeline Log
                </h2>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-4">
                {item.keyframes.map((kf) => (
                  <div key={kf.id} className="border border-[rgba(255,255,255,0.04)] bg-[#111] rounded p-2 space-y-2">
                    <div className="relative rounded overflow-hidden aspect-video bg-black">
                      <img
                        src={`${backendUrl}${kf.filepath}`}
                        alt={`Frame ${kf.timestamp}`}
                        className="object-cover w-full h-full"
                      />
                      <div className="absolute bottom-1 right-1 bg-black/75 px-1 py-0.5 rounded font-mono text-[8px] text-white">
                        {kf.timestamp.toFixed(1)}s
                      </div>
                    </div>
                    <div className="font-mono text-[8px] text-gray-500 truncate" title={kf.phash}>
                      pHash: <span className="text-white">{kf.phash}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Audio Spectral peaks (if has audio) */}
          {item.audio_fingerprint?.has_audio && (
            <div className="cyber-card p-6 space-y-4">
              <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                <Volume2 className="h-4 w-4 text-[#00FF9D]" />
                <h2 className="font-mono text-xs font-bold text-[#00FF9D] tracking-widest uppercase">
                  Audio Waveform Chroma fingerprint signature
                </h2>
              </div>

              <div className="grid grid-cols-12 gap-1.5 h-20 items-end bg-[#111] p-4 rounded border border-[rgba(255,255,255,0.03)]">
                {item.audio_fingerprint.mean_chroma.map((val: number, idx: number) => (
                  <div key={idx} className="h-full flex flex-col justify-end gap-1.5 text-center">
                    <div
                      className="bg-gradient-to-t from-[#7C3AED] to-[#00FF9D] w-full rounded-t-sm transition-all"
                      style={{ height: `${Math.max(5, val * 100)}%` }}
                    ></div>
                    <span className="font-mono text-[7px] text-gray-600">
                      {["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][idx]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Direct Similarity matches details list */}
          <div className="cyber-card p-6 space-y-4">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
              <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                Cross-Indexed Similarity Variations ({similarItems.length})
              </h2>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs text-gray-400">
                <thead>
                  <tr className="border-b border-[rgba(255,255,255,0.06)] text-[9px] uppercase tracking-widest text-gray-500">
                    <th className="pb-2">Target Name</th>
                    <th className="pb-2">Relationship Type</th>
                    <th className="pb-2 text-center">Visual Match</th>
                    <th className="pb-2 text-center">Audio Match</th>
                    <th className="pb-2 text-center">Semantic Match</th>
                    <th className="pb-2 text-center">Confidence</th>
                    <th className="pb-2 text-right">View DNA</th>
                  </tr>
                </thead>
                <tbody>
                  {similarItems.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-6 text-center text-gray-600">
                        NO CORRELATED VARIATIONS INDEXED IN DATABASE.
                      </td>
                    </tr>
                  ) : (
                    similarItems.map((match) => (
                      <tr
                        key={match.id}
                        className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.01)] transition-colors"
                      >
                        <td className="py-3 font-semibold text-white max-w-[200px] truncate">
                          {match.filename}
                        </td>
                        <td className="py-3 uppercase text-gray-500">
                          {match.relationship_type || "Variant"}
                        </td>
                        <td className="py-3 text-center">
                          {Math.round(match.visual_similarity * 100)}%
                        </td>
                        <td className="py-3 text-center">
                          {match.audio_similarity > 0 ? `${Math.round(match.audio_similarity * 100)}%` : "—"}
                        </td>
                        <td className="py-3 text-center">
                          {match.semantic_similarity > 0 ? `${Math.round(match.semantic_similarity * 100)}%` : "—"}
                        </td>
                        <td className="py-3 text-center">
                          <span className="font-bold text-[#00FF9D] glow-text-green">
                            {Math.round(match.combined_score * 100)}%
                          </span>
                        </td>
                        <td className="py-3 text-right">
                          <button
                            onClick={() => router.push(`/media/${match.id}`)}
                            className="inline-flex items-center gap-1 text-[#00E5FF] hover:underline"
                          >
                            <Eye className="h-3 w-3" />
                            PROFILE
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* OSINT Intelligence Tab Layout */}
      {activeTab === "osint" && (
        <div className="space-y-6 animate-fadeIn">
          {/* Active Provider status log */}
          {item.modification_report?.osint_summary?.provider_status && (
            <div className="bg-[#12121A]/80 border border-[rgba(255,255,255,0.04)] rounded-xl p-4 font-mono text-xs text-gray-400 space-y-3">
              <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Multi-Provider Status Registry</span>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(item.modification_report.osint_summary.provider_status).map(([providerName, provStatus]: any) => {
                  const isOffline = provStatus.includes("Offline") || provStatus.includes("Degraded");
                  return (
                    <div key={providerName} className="p-2 bg-[#09090D] border border-[rgba(255,255,255,0.02)] rounded flex flex-col justify-between">
                      <span className="text-[8px] text-gray-500 font-bold uppercase">{providerName}</span>
                      <span className={`block text-[10px] font-black uppercase mt-1 ${isOffline ? "text-red-500" : "text-[#00FF9D]"}`}>
                        {isOffline ? "OFFLINE / FALLBACK" : "ACTIVE"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* OSINT Scan Control Card */}
          <div className="cyber-card p-6 space-y-6">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div className="space-y-1">
                <h2 className="font-mono text-sm font-bold text-[#7C3AED] tracking-widest uppercase flex items-center gap-2">
                  <Globe className="h-4.5 w-4.5" />
                  OSINT Hunt Control Center
                </h2>
                <p className="text-xs text-gray-500 font-mono">
                  Scan global networks (Reddit, News outlets, Google indexing) for online references to this asset signature.
                </p>
              </div>

              {/* Action Button */}
              {osintScanStatus !== "Running" && osintScanStatus !== "Pending" && (
                <button
                  onClick={handleLaunchScan}
                  className="flex items-center gap-2 rounded bg-gradient-to-r from-[#7C3AED] to-[#00E5FF] hover:opacity-90 px-6 py-2.5 font-mono text-xs text-black font-black transition-all shadow-[0_0_15px_rgba(124,58,237,0.3)] uppercase tracking-wider"
                >
                  <Search className="h-4 w-4" />
                  Investigate Online
                </button>
              )}
            </div>

            {/* Config & Search Parameters (if not currently running) */}
            {osintScanStatus !== "Running" && osintScanStatus !== "Pending" && (
              <div className="bg-[#12121A] p-4 rounded border border-[rgba(255,255,255,0.03)] space-y-3 font-mono text-xs">
                <div>
                  <label className="block text-[8px] uppercase tracking-widest text-gray-500 mb-1">Target Search Query</label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter OSINT search keywords..."
                    className="w-full bg-[#08080C] border border-[rgba(255,255,255,0.08)] rounded px-3 py-2 text-white focus:outline-none focus:border-[#7C3AED] font-mono text-xs"
                  />
                </div>

                {osintTags.length > 0 && (
                  <div>
                    <span className="block text-[8px] uppercase tracking-widest text-gray-500 mb-1">Generated Image Tags</span>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {osintTags.map((tag, idx) => (
                        <span key={idx} className="px-2 py-0.5 rounded-full bg-[#7C3AED]/10 border border-[#7C3AED]/20 text-[9px] text-[#A78BFA] font-bold">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Active Scan Progress Status */}
            {isScanning && (
              <div className="bg-[#0C0C12] border border-[rgba(255,255,255,0.04)] rounded p-6 space-y-6">
                <div className="flex items-center gap-4">
                  <div className="relative h-10 w-10 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border-4 border-[#7C3AED]/20 border-t-[#7C3AED] animate-spin"></div>
                    <Cpu className="h-4 w-4 text-[#7C3AED] animate-pulse" />
                  </div>
                  <div>
                    <h3 className="font-mono text-xs font-bold text-white uppercase tracking-wider">
                      Apify Actor Ingestion Triggered
                    </h3>
                    <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest mt-0.5">
                      Scraping Reddit, Google Search, & News Outlets
                    </span>
                  </div>
                </div>

                {/* Simulated/Real Diagnostic Stepper */}
                <div className="space-y-2.5 font-mono text-[10px]">
                  <div className="flex items-center gap-2 text-[#00FF9D]">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>EXTRACTED DIGITAL FOOTPRINTS & IMAGE SIGNATURES</span>
                  </div>
                  <div className={`flex items-center gap-2 ${osintTags.length > 0 ? "text-[#00FF9D]" : "text-gray-400 animate-pulse"}`}>
                    {osintTags.length > 0 ? <CheckCircle2 className="h-3.5 w-3.5" /> : <div className="h-2 w-2 rounded-full bg-gray-500"></div>}
                    <span>CLASSIFYING IMAGE SUBJECT VIA SEMANTIC ANALYSIS</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-400 animate-pulse">
                    <div className="h-2 w-2 rounded-full bg-purple-500 animate-ping"></div>
                    <span>COMMENCING ONLINE SCRAPE ACTORS (APIFY Google Search Scraper)</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-600">
                    <div className="h-2 w-2 rounded-full bg-gray-700"></div>
                    <span>CALCULATING CONFIDENCE SCORES & EXTRACTING MATCH KEYWORDS</span>
                  </div>
                </div>
              </div>
            )}

            {scanError && (
              <div className="p-3 bg-[#FF3366]/5 border border-[#FF3366]/20 rounded flex items-center gap-2 text-xs text-[#FF3366] font-mono">
                <AlertCircle className="h-4 w-4" />
                <span>OSINT SCANNER DEGRADED: {scanError}</span>
              </div>
            )}
          </div>

          {/* Not Started / Pending Scan placeholder */}
          {(osintScanStatus === "Not Started" || osintScanStatus === "Pending" || osintScanStatus === "Not Searched") && (
            <div className="cyber-card p-12 text-center space-y-4">
              <Globe className="h-8 w-8 text-gray-600 mx-auto" />
              <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest">
                OSINT Hunt Not Executed
              </h3>
              <p className="text-xs text-gray-500 max-w-[420px] mx-auto font-mono leading-relaxed">
                No active web traces or online matches have been verified. Click the <strong>Investigate Online</strong> button to query global indexes for visual duplicates.
              </p>
            </div>
          )}

          {/* Completed scan but no results */}
          {osintScanStatus === "No Matches Found" && (
            <div className="cyber-card p-12 text-center space-y-4">
              <Globe className="h-8 w-8 text-gray-600 mx-auto" />
              <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest">
                No Matches Found
              </h3>
              <p className="text-xs text-gray-500 max-w-[420px] mx-auto font-mono leading-relaxed">
                The visual fingerprints (SHA256, hashes) and metadata tags did not return matching traces on indexed Reddit posts, news channels, or search results.
              </p>
            </div>
          )}

          {/* Scan executed with Provider Unavailable and 0 results */}
          {osintScanStatus === "Provider Unavailable" && osintResults.length === 0 && (
            <div className="cyber-card p-12 text-center space-y-4 border border-[#FFCC00]/20 bg-[#FFCC00]/5">
              <Globe className="h-8 w-8 text-[#FFCC00] mx-auto opacity-80" />
              <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest">
                Provider Unavailable
              </h3>
              <p className="text-xs text-gray-500 max-w-[420px] mx-auto font-mono leading-relaxed">
                The visual search API keys (Google Lens, Bing Visual Search, Yandex, TinEye) are not configured. Standard web intelligence scan could not be executed.
              </p>
            </div>
          )}

          {/* Scan completed with results */}
          {(osintScanStatus === "Verified Matches Found" || (osintScanStatus === "Provider Unavailable" && osintResults.length > 0)) && (
            <>
              {/* Summary Cards Row */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-1">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">
                    {osintScanStatus === "Verified Matches Found" ? "Total Matches Count" : "Simulated Matches Count"}
                  </span>
                  <span className="block text-2xl font-black text-[#00E5FF] glow-text-cyan">
                    {osintResults.length}
                  </span>
                </div>
                <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-1">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">
                    {osintScanStatus === "Verified Matches Found" ? "Reddit Threads" : "Simulated Reddit Threads"}
                  </span>
                  <span className="block text-2xl font-black text-[#7C3AED] glow-text-purple">
                    {osintResults.filter(r => r.source === "Reddit").length}
                  </span>
                </div>
                <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-1">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">
                    {osintScanStatus === "Verified Matches Found" ? "News Websites" : "Simulated News Outlets"}
                  </span>
                  <span className="block text-2xl font-black text-[#00FF9D] glow-text-green">
                    {osintResults.filter(r => r.source === "News").length}
                  </span>
                </div>
                <div className="bg-[#12121A]/50 border border-[rgba(255,255,255,0.04)] rounded p-4 font-mono space-y-1">
                  <span className="block text-[8px] text-gray-500 uppercase tracking-widest">
                    {osintScanStatus === "Verified Matches Found" ? "Google Search References" : "Simulated Web References"}
                  </span>
                  <span className="block text-2xl font-black text-white">
                    {osintResults.filter(r => r.source === "Google Search").length}
                  </span>
                </div>
              </div>

              {/* Timeline (Left) & Results Table (Right) */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Timeline Panel */}
                <div className="cyber-card p-6 space-y-4 lg:col-span-1">
                  <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                    <Clock className="h-4 w-4 text-[#7C3AED]" />
                    <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest">
                      Timeline Log View
                    </h3>
                  </div>

                  <div className="relative pl-6 border-l border-[rgba(255,255,255,0.06)] space-y-6 max-h-[500px] overflow-y-auto pr-2">
                    {osintResults.map((res, idx) => {
                      const isReal = res.source_type === "real_provider" || res.source_type === "apify";
                      return (
                        <div key={idx} className="relative space-y-1.5 animate-fadeIn">
                          {/* Dot */}
                          <div className={`absolute -left-[30px] top-1.5 h-2 w-2 rounded-full border-2 ${isReal ? "bg-[#00FF9D] border-[#00FF9D]" : "bg-[#FF3366] border-[#FF3366]"
                            }`} />

                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="block font-mono text-[8px] text-gray-500 font-bold">
                              {res.publication_date || "N/A Date"}
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-[7px] font-black uppercase tracking-wider ${isReal
                                ? "bg-[#00FF9D]/10 text-[#00FF9D] border border-[#00FF9D]/20"
                                : "bg-[#FF3366]/10 text-[#FF3366] border border-[#FF3366]/20"
                              }`}>
                              {isReal ? "REAL PROVIDER" : "SIMULATED RESULT"}
                            </span>
                            <span className="text-[7.5px] text-gray-500 font-bold font-mono">
                              ({res.source_type === "real_provider" ? res.source : res.source_type === "apify" ? "Apify Search" : "Mock Registry"})
                            </span>
                          </div>

                          <h4 className="font-mono text-xs font-bold text-white hover:text-[#00E5FF] transition-colors leading-tight">
                            <a href={res.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                              {res.title}
                              <ExternalLink className="h-2.5 w-2.5 inline" />
                            </a>
                          </h4>

                          <p className="font-mono text-[9px] text-gray-400 line-clamp-2 leading-relaxed">
                            {res.snippet}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Data Table Panel */}
                <div className="cyber-card p-6 space-y-4 lg:col-span-2">
                  <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                    <h3 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                      OSINT Source Table
                    </h3>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left font-mono text-xs text-gray-400">
                      <thead>
                        <tr className="border-b border-[rgba(255,255,255,0.06)] text-[9px] uppercase tracking-widest text-gray-500">
                          <th className="pb-2">Provider</th>
                          <th className="pb-2">Details</th>
                          <th className="pb-2">Evidence Source</th>
                          <th className="pb-2 text-center">Confidence</th>
                          <th className="pb-2 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {osintResults.map((res, idx) => {
                          const isReal = res.source_type === "real_provider" || res.source_type === "apify";
                          const providerName =
                            res.source_type === "real_provider" ? res.source :
                              res.source_type === "apify" ? "Apify Search" : "Mock Registry";
                          return (
                            <tr key={idx} className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.01)] transition-colors">
                              <td className="py-3.5 pr-2 align-top">
                                <span className="block text-[9px] font-bold text-white uppercase">{providerName}</span>
                                <span className="block text-[8px] text-gray-500 font-mono mt-0.5">{res.source}</span>
                              </td>

                              <td className="py-3.5 pr-4 space-y-1 align-top max-w-[280px]">
                                <span className="block font-semibold text-white leading-snug">{res.title}</span>
                                <span className="block text-[9px] text-gray-500 leading-relaxed line-clamp-3">{res.snippet}</span>
                                <span className="block text-[8px] text-gray-600 font-bold">Published: {res.publication_date}</span>
                              </td>

                              <td className="py-3.5 pr-2 align-top">
                                <span className={`inline-block px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-widest ${isReal ? "bg-[#00FF9D]/10 text-[#00FF9D] border border-[#00FF9D]/20" :
                                    "bg-[#FF3366]/10 text-[#FF3366] border border-[#FF3366]/20"
                                  }`}>
                                  {isReal ? "REAL PROVIDER" : "SIMULATED RESULT"}
                                </span>
                              </td>

                              <td className="py-3.5 text-center align-top whitespace-nowrap">
                                <div className="space-y-1">
                                  <span className={`block font-black text-sm ${res.confidence >= 80 ? "text-[#00FF9D] glow-text-green" :
                                      res.confidence >= 60 ? "text-[#00E5FF] glow-text-cyan" :
                                        "text-yellow-500"
                                    }`}>
                                    {res.confidence}% Match
                                  </span>
                                  {res.reason && (
                                    <span className="block text-[7px] text-gray-500 max-w-[120px] mx-auto text-center leading-tight whitespace-normal">
                                      {res.reason}
                                    </span>
                                  )}
                                </div>
                              </td>

                              <td className="py-3.5 text-right align-top">
                                <a
                                  href={res.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-[#00E5FF] hover:underline"
                                >
                                  <ExternalLink className="h-4 w-4" />
                                </a>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Forensic OSINT Provenance Audit Report */}
              <div className="cyber-card p-6 mt-8 space-y-4 border border-[rgba(0,229,255,0.15)] bg-[#09090D]/80">
                <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-[#00E5FF] glow-text-cyan" />
                  <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest">
                    Forensic OSINT Provenance Audit Report
                  </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-mono text-[10px] text-gray-400">
                  <div className="space-y-3">
                    <div>
                      <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Queried Provider Platforms</span>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {item.modification_report?.osint_summary?.provider_status ? (
                          Object.entries(item.modification_report.osint_summary.provider_status)
                            .filter(([_, status]: any) => !status.includes("Degraded") && !status.includes("Offline"))
                            .map(([providerName]: any) => (
                              <span key={providerName} className="px-2 py-0.5 rounded bg-[#00FF9D]/10 border border-[#00FF9D]/20 text-[#00FF9D] font-bold text-[8px] uppercase">
                                {providerName}
                              </span>
                            ))
                        ) : (
                          <span className="text-gray-600">None</span>
                        )}
                        {(!item.modification_report?.osint_summary?.provider_status ||
                          Object.values(item.modification_report.osint_summary.provider_status).every((status: any) => status.includes("Degraded") || status.includes("Offline"))) && (
                            <span className="text-gray-600">None</span>
                          )}
                      </div>
                    </div>

                    <div>
                      <span className="block text-[8px] text-gray-500 uppercase tracking-widest">Unavailable Provider Platforms (Credentials Missing)</span>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {item.modification_report?.osint_summary?.provider_status ? (
                          Object.entries(item.modification_report.osint_summary.provider_status)
                            .filter(([_, status]: any) => status.includes("Degraded") || status.includes("Offline"))
                            .map(([providerName]: any) => (
                              <span key={providerName} className="px-2 py-0.5 rounded bg-[#FF3366]/10 border border-[#FF3366]/20 text-[#FF3366] font-bold text-[8px] uppercase">
                                {providerName}
                              </span>
                            ))
                        ) : (
                          <span className="text-gray-600">None</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Real Evidence Matches</span>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-gray-300">
                        {osintResults.filter(r => r.source_type === "real_provider" || r.source_type === "apify").length > 0 ? (
                          osintResults
                            .filter(r => r.source_type === "real_provider" || r.source_type === "apify")
                            .map((r, i) => (
                              <li key={i} className="truncate">
                                <span className="text-[#00FF9D] font-bold">[{r.source}]</span> {r.title}
                              </li>
                            ))
                        ) : (
                          <li className="text-gray-600 list-none">No real provider evidence index matched.</li>
                        )}
                      </ul>
                    </div>

                    <div>
                      <span className="block text-[8px] text-gray-500 uppercase tracking-widest font-bold">Simulated Reference Matches</span>
                      <ul className="list-disc list-inside mt-1 space-y-1 text-gray-400">
                        {osintResults.filter(r => r.source_type !== "real_provider" && r.source_type !== "apify").length > 0 ? (
                          osintResults
                            .filter(r => r.source_type !== "real_provider" && r.source_type !== "apify")
                            .map((r, i) => (
                              <li key={i} className="truncate">
                                <span className="text-[#FF3366] font-bold">[{r.source}]</span> {r.title}
                              </li>
                            ))
                        ) : (
                          <li className="text-gray-600 list-none">No simulated reference matches loaded.</li>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
