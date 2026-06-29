"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Database, Shield, Share2, Award, FileText,
  Plus, Calendar, ArrowRight, Eye, Video, Image as ImageIcon,
  GitMerge, AlertCircle, Sparkles
} from "lucide-react";

interface Stats {
  total_indexed: number;
  cases_count: number;
  matches_count: number;
  avg_confidence: number;
  videos_processed: number;
  images_processed: number;
  recent_investigations: Array<{
    id: number;
    filename: string;
    mime_type: string;
    created_at: string;
    risk_score: number;
    integrity_score: number;
  }>;
}

interface Case {
  id: number;
  name: string;
  description: string;
  created_at: string;
  status: string;
}

interface Family {
  cluster_id: string;
  case_id: number;
  family_size: number;
  most_probable_origin: string;
  most_probable_origin_id: number | null;
  variant_distribution: string;
  earliest_known_appearance: string | null;
  investigation_confidence: {
    level: string;
    score: number;
  };
  investigation_narrative: string;
}

interface MergeRecommendation {
  id: number;
  case_id: number;
  source_cluster_id: string;
  target_cluster_id: string;
  confidence: number;
  status: string;
  created_at: string;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [cases, setCases] = useState<Case[]>([]);
  const [families, setFamilies] = useState<Family[]>([]);
  const [merges, setMerges] = useState<MergeRecommendation[]>([]);
  const [activeTab, setActiveTab] = useState<"dashboard" | "families">("dashboard");
  const [loading, setLoading] = useState(true);

  // Case Form State
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [newCaseName, setNewCaseName] = useState("");
  const [newCaseDesc, setNewCaseDesc] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const backendUrl = "http://127.0.0.1:8000";

  const fetchDashboardData = async () => {
    try {
      const [statsRes, casesRes, familiesRes] = await Promise.all([
        fetch(`${backendUrl}/api/dashboard`).catch(() => null),
        fetch(`${backendUrl}/api/cases`).catch(() => null),
        fetch(`${backendUrl}/api/families`).catch(() => null),
      ]);
      
      if (statsRes && statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
      if (casesRes && casesRes.ok) {
        const casesData = await casesRes.json();
        setCases(casesData);
        
        if (casesData.length > 0) {
          const mergePromises = casesData.map((c: Case) =>
            fetch(`${backendUrl}/api/cases/${c.id}/merges`)
              .then((r) => (r.ok ? r.json() : []))
              .catch(() => [])
          );
          const mergeResults = await Promise.all(mergePromises);
          setMerges(mergeResults.flat());
        }
      }
      if (familiesRes && familiesRes.ok) {
        const familiesData = await familiesRes.json();
        setFamilies(familiesData);
      }
    } catch (e) {
      console.error("Connection failed with backend. Make sure FastAPI is running on port 8000.", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCaseName) return;
    setSubmitting(true);
    try {
      const res = await fetch(`${backendUrl}/api/cases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newCaseName, description: newCaseDesc }),
      });
      if (res.ok) {
        setNewCaseName("");
        setNewCaseDesc("");
        setShowCreateCase(false);
        fetchDashboardData();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleApproveMerge = async (id: number) => {
    try {
      const res = await fetch(`${backendUrl}/api/merges/${id}/approve`, {
        method: "POST"
      });
      if (res.ok) {
        fetchDashboardData();
      }
    } catch (err) {
      console.error("Failed to approve merge:", err);
    }
  };

  const handleRejectMerge = async (id: number) => {
    try {
      const res = await fetch(`${backendUrl}/api/merges/${id}/reject`, {
        method: "POST"
      });
      if (res.ok) {
        fetchDashboardData();
      }
    } catch (err) {
      console.error("Failed to reject merge:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-t-transparent border-[#00E5FF]"></div>
        <span className="font-mono text-xs text-gray-500 tracking-widest">INITIALIZING COGNITIVE INTERFACE...</span>
      </div>
    );
  }

  const riskColor = (score: number) => {
    if (score >= 70) return "text-[#FF3366] bg-[rgba(255,51,102,0.1)] border-[rgba(255,51,102,0.3)]";
    if (score >= 35) return "text-[#F59E0B] bg-[rgba(245,158,11,0.1)] border-[rgba(245,158,11,0.3)]";
    return "text-[#00FF9D] bg-[rgba(0,255,157,0.1)] border-[rgba(0,255,157,0.3)]";
  };

  const integrityColor = (score: number) => {
    if (score >= 80) return "text-[#00FF9D] bg-[rgba(0,255,157,0.1)] border-[rgba(0,255,157,0.3)]";
    if (score >= 50) return "text-[#F59E0B] bg-[rgba(245,158,11,0.1)] border-[rgba(245,158,11,0.3)]";
    return "text-[#FF3366] bg-[rgba(255,51,102,0.1)] border-[rgba(255,51,102,0.3)]";
  };

  const getConfidenceBadgeColor = (level: string) => {
    if (level === "High") return "text-[#00FF9D] bg-[rgba(0,255,157,0.08)] border-[rgba(0,255,157,0.2)]";
    if (level === "Medium") return "text-[#F59E0B] bg-[rgba(245,158,11,0.08)] border-[rgba(245,158,11,0.2)]";
    return "text-[#FF3366] bg-[rgba(255,51,102,0.08)] border-[rgba(255,51,102,0.2)]";
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Platform Title Banner */}
      <div className="relative overflow-hidden rounded-xl border border-[rgba(255,255,255,0.05)] bg-gradient-to-r from-[#0C0C14] to-[#0A0A0A] p-6 shadow-2xl">
        <div className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-[#7C3AED]/10 blur-3xl"></div>
        <div className="absolute -bottom-20 -left-20 h-48 w-48 rounded-full bg-[#00E5FF]/10 blur-3xl"></div>

        <div className="relative flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="font-mono text-2xl font-black text-white tracking-wider uppercase">
              Investigation Dashboard
            </h1>
            <p className="mt-1 text-xs text-gray-500 tracking-wide">
              Media Intelligence Platform & Forensic Diagnostics System
            </p>
          </div>
          <div className="flex gap-3">
            <Link
              href="/upload"
              className="cyber-button px-4 py-2 font-mono text-xs tracking-wider font-semibold"
            >
              INGEST ASSET
            </Link>
            <button
              onClick={() => setShowCreateCase(true)}
              className="flex items-center gap-2 rounded-md border border-[rgba(255,255,255,0.1)] bg-[#1A1A1A] px-4 py-2 font-mono text-xs tracking-wider font-semibold text-white hover:bg-[#222]"
            >
              <Plus className="h-4 w-4 text-[#00FF9D]" />
              NEW CASE
            </button>
          </div>
        </div>
      </div>

      {/* Case Creation Overlay Modal */}
      {showCreateCase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-md border border-[rgba(255,255,255,0.1)] bg-[#0C0C0C] p-6 rounded-lg shadow-2xl">
            <h3 className="font-mono text-sm font-bold text-[#00E5FF] tracking-wider uppercase mb-4">
              Open New Investigation Case
            </h3>
            <form onSubmit={handleCreateCase} className="space-y-4">
              <div>
                <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
                  Case Identifier / Name
                </label>
                <input
                  type="text"
                  value={newCaseName}
                  onChange={(e) => setNewCaseName(e.target.value)}
                  placeholder="e.g. Case #2026-ALPHA: Intel Leak"
                  className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2 font-mono text-xs text-white placeholder-gray-600 focus:border-[#00E5FF] focus:outline-none"
                  required
                />
              </div>
              <div>
                <label className="block font-mono text-[10px] text-gray-500 tracking-widest uppercase mb-1">
                  Scope / Description
                </label>
                <textarea
                  value={newCaseDesc}
                  onChange={(e) => setNewCaseDesc(e.target.value)}
                  placeholder="Detail the parameters of the investigation..."
                  className="w-full rounded border border-[rgba(255,255,255,0.08)] bg-[#141414] p-2 font-mono text-xs text-white placeholder-gray-600 focus:border-[#00E5FF] focus:outline-none h-24 resize-none"
                />
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCreateCase(false)}
                  className="rounded border border-[rgba(255,255,255,0.1)] px-4 py-2 font-mono text-xs tracking-wider text-gray-400 hover:text-white"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="cyber-button px-4 py-2 font-mono text-xs tracking-wider"
                >
                  {submitting ? "CREATING..." : "CONFIRM CASE"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Grid Statistics Widgets */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Indexed Assets</span>
            <Database className="h-4 w-4 text-[#00E5FF]" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-white">
            {stats?.total_indexed ?? 0}
          </span>
        </div>
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Active Cases</span>
            <Shield className="h-4 w-4 text-[#7C3AED]" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-white">
            {stats?.cases_count ?? 0}
          </span>
        </div>
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Linked Pairs</span>
            <Share2 className="h-4 w-4 text-[#00FF9D]" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-white">
            {stats?.matches_count ?? 0}
          </span>
        </div>
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Avg Confidence</span>
            <Award className="h-4 w-4 text-[#00E5FF]" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-[#00FF9D] glow-text-green">
            {stats ? `${intPercent(stats.avg_confidence)}%` : "0%"}
          </span>
        </div>
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Images Ingested</span>
            <ImageIcon className="h-4 w-4 text-gray-400" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-white">
            {stats?.images_processed ?? 0}
          </span>
        </div>
        <div className="cyber-card p-4 flex flex-col justify-between min-h-[100px]">
          <div className="flex justify-between items-start">
            <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Videos Ingested</span>
            <Video className="h-4 w-4 text-gray-400" />
          </div>
          <span className="mt-2 text-2xl font-bold font-mono text-white">
            {stats?.videos_processed ?? 0}
          </span>
        </div>
      </div>

      {/* Tabs Menu */}
      <div className="flex border-b border-[rgba(255,255,255,0.08)] bg-[#0A0A0E] p-1 rounded-lg">
        <button
          onClick={() => setActiveTab("dashboard")}
          className={`flex-1 md:flex-initial text-center px-6 py-2.5 font-mono text-xs tracking-wider uppercase font-bold rounded-md transition-all ${
            activeTab === "dashboard"
              ? "bg-[#14141E] text-[#00E5FF] border border-[rgba(0,229,255,0.2)]"
              : "text-gray-500 hover:text-white"
          }`}
        >
          FORENSICS REGISTRY
        </button>
        <button
          onClick={() => setActiveTab("families")}
          className={`flex-1 md:flex-initial text-center px-6 py-2.5 font-mono text-xs tracking-wider uppercase font-bold rounded-md transition-all ${
            activeTab === "families"
              ? "bg-[#14141E] text-[#00E5FF] border border-[rgba(0,229,255,0.2)]"
              : "text-gray-500 hover:text-white"
          }`}
        >
          MEDIA FAMILIES DASHBOARD ({families.length})
        </button>
      </div>

      {activeTab === "dashboard" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Side: Active Cases Index */}
          <div className="cyber-card p-6 flex flex-col gap-4 lg:col-span-1">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
              <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                Investigation cases ({cases.length})
              </h2>
            </div>

            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
              {cases.length === 0 ? (
                <span className="block text-center font-mono text-xs text-gray-600 py-8">NO CASES SEEDED</span>
              ) : (
                cases.map((c) => (
                  <div
                    key={c.id}
                    className="rounded border border-[rgba(255,255,255,0.04)] bg-[#111111] p-3 hover:border-[rgba(0,229,255,0.2)] transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <span className="font-mono text-xs font-bold text-white tracking-wide">{c.name}</span>
                      <span className="rounded bg-[rgba(0,255,157,0.08)] px-1.5 py-0.5 font-mono text-[8px] text-[#00FF9D] border border-[rgba(0,255,157,0.2)] uppercase">{c.status}</span>
                    </div>
                    <p className="mt-1 text-[10px] text-gray-500 line-clamp-2">{c.description || "No description provided."}</p>
                    <div className="mt-3 flex items-center justify-between border-t border-[rgba(255,255,255,0.03)] pt-2 text-[8px] font-mono text-gray-600">
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {new Date(c.created_at).toLocaleDateString()}
                      </span>
                      <Link
                        href={`/upload?caseId=${c.id}`}
                        className="flex items-center gap-1 text-[#00E5FF] hover:underline"
                      >
                        INGEST HERE
                        <ArrowRight className="h-3 w-3" />
                      </Link>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Right Side: Recent Investigations Logs */}
          <div className="cyber-card p-6 flex flex-col gap-4 lg:col-span-2">
            <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
              <h2 className="font-mono text-xs font-bold text-[#00E5FF] tracking-widest uppercase">
                Recent Forensic Registries
              </h2>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs text-gray-400">
                <thead>
                  <tr className="border-b border-[rgba(255,255,255,0.06)] text-[9px] uppercase tracking-widest text-gray-500">
                    <th className="pb-2">Filename</th>
                    <th className="pb-2">Format</th>
                    <th className="pb-2 text-center">Integrity</th>
                    <th className="pb-2 text-center">Risk Score</th>
                    <th className="pb-2 text-right">View DNA</th>
                  </tr>
                </thead>
                <tbody>
                  {!stats || stats.recent_investigations.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-gray-600">
                        NO ASSETS INGESTED YET. UPLOAD AN IMAGE OR VIDEO TO INITIATE FORENSICS.
                      </td>
                    </tr>
                  ) : (
                    stats.recent_investigations.map((file) => (
                      <tr
                        key={file.id}
                        className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.01)] transition-colors"
                      >
                        <td className="py-3 font-semibold text-white max-w-[200px] truncate">
                          {file.filename}
                        </td>
                        <td className="py-3 uppercase text-gray-500">
                          {file.mime_type.split("/")[1]}
                        </td>
                        <td className="py-3 text-center">
                          <span className={`inline-block w-16 text-center rounded border px-1.5 py-0.5 text-[10px] font-bold ${integrityColor(file.integrity_score)}`}>
                            {file.integrity_score}
                          </span>
                        </td>
                        <td className="py-3 text-center">
                          <span className={`inline-block w-16 text-center rounded border px-1.5 py-0.5 text-[10px] font-bold ${riskColor(file.risk_score)}`}>
                            {file.risk_score}
                          </span>
                        </td>
                        <td className="py-3 text-right">
                          <Link
                            href={`/media/${file.id}`}
                            className="inline-flex items-center gap-1.5 text-xs text-[#00E5FF] hover:underline"
                          >
                            <Eye className="h-3 w-3" />
                            PROFILE
                          </Link>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Analyst Pending Merges Approvals Section */}
          {merges.length > 0 && (
            <div className="cyber-card p-6 border-l-4 border-l-[#F59E0B] bg-[rgba(245,158,11,0.02)] space-y-4">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-[#F59E0B]" />
                <h2 className="font-mono text-sm font-bold text-[#F59E0B] tracking-wider uppercase">
                  Analyst Pending Cluster Merges ({merges.length})
                </h2>
              </div>
              <p className="text-xs text-gray-400 max-w-3xl">
                The platform has detected high cross-similarity matches between visual clusters. These clusters are isolated to protect independent evidence tracking. Review the cases below and confirm if they should be merged.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {merges.map((rec) => (
                  <div
                    key={rec.id}
                    className="flex flex-col justify-between p-4 rounded border border-[rgba(245,158,11,0.2)] bg-[#0F0E0A] gap-3"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[10px] font-mono text-gray-500">
                        <span>CASE OBJECT ID: {rec.case_id}</span>
                        <span className="font-bold text-[#00FF9D]">CONFIDENCE: {Math.round(rec.confidence * 100)}%</span>
                      </div>
                      <div className="font-mono text-xs text-white flex items-center gap-2 flex-wrap mt-1">
                        <span className="px-1.5 py-0.5 rounded bg-[#1C160F] text-[#F59E0B] border border-[rgba(245,158,11,0.3)]">{rec.source_cluster_id}</span>
                        <ArrowRight className="h-3.5 w-3.5 text-gray-600" />
                        <span className="px-1.5 py-0.5 rounded bg-[#0F1C16] text-[#00FF9D] border border-[rgba(0,255,157,0.3)]">{rec.target_cluster_id}</span>
                      </div>
                    </div>
                    <div className="flex gap-2 justify-end border-t border-[rgba(255,255,255,0.04)] pt-3">
                      <button
                        onClick={() => handleRejectMerge(rec.id)}
                        className="rounded border border-[rgba(255,255,255,0.1)] hover:bg-[#222] px-3 py-1.5 font-mono text-[10px] tracking-wider text-gray-400 hover:text-white transition-colors"
                      >
                        DISCARD
                      </button>
                      <button
                        onClick={() => handleApproveMerge(rec.id)}
                        className="flex items-center gap-1.5 rounded bg-[#F59E0B] hover:bg-[#D97706] text-black px-4 py-1.5 font-mono text-[10px] tracking-wider font-bold transition-colors"
                      >
                        <GitMerge className="h-3 w-3" />
                        APPROVE MERGE
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Families Visual Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {families.length === 0 ? (
              <div className="cyber-card p-12 text-center text-gray-600 col-span-2 font-mono text-sm">
                NO MEDIA FAMILIES RECONSTRUCTED. INGEST MULTIPLE VISUALLY RELATED ASSETS TO BEGIN CLUSTERING.
              </div>
            ) : (
              families.map((fam) => (
                <div
                  key={fam.cluster_id}
                  className="cyber-card p-6 flex flex-col justify-between gap-5 relative overflow-hidden group hover:border-[rgba(0,229,255,0.2)] transition-all"
                >
                  <div className="absolute right-0 top-0 h-24 w-24 rounded-full bg-[#00E5FF]/5 blur-2xl group-hover:bg-[#00E5FF]/10 transition-colors"></div>

                  <div className="space-y-4">
                    {/* Header */}
                    <div className="flex items-start justify-between border-b border-[rgba(255,255,255,0.06)] pb-3">
                      <div className="space-y-1">
                        <span className="font-mono text-[9px] text-gray-500 tracking-widest uppercase">Visual Cluster Family</span>
                        <h3 className="font-mono text-sm font-bold text-white tracking-wide group-hover:text-[#00E5FF] transition-colors">
                          {fam.cluster_id}
                        </h3>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="rounded bg-[rgba(0,229,255,0.08)] px-2 py-0.5 font-mono text-[10px] text-[#00E5FF] border border-[rgba(0,229,255,0.2)] font-bold">
                          {fam.family_size} Assets
                        </span>
                        {fam.earliest_known_appearance && (
                          <span className="text-[8px] font-mono text-gray-600 uppercase">
                            Earliest: {new Date(fam.earliest_known_appearance).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Properties */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
                      <div className="space-y-0.5 border border-[rgba(255,255,255,0.03)] bg-[#09090E] p-2 rounded">
                        <span className="text-[9px] text-gray-500 uppercase tracking-wider block">Most Probable Origin</span>
                        <span className="text-white font-semibold truncate block">
                          {fam.most_probable_origin}
                        </span>
                      </div>
                      <div className="space-y-0.5 border border-[rgba(255,255,255,0.03)] bg-[#09090E] p-2 rounded">
                        <span className="text-[9px] text-gray-500 uppercase tracking-wider block">Variants Mapped</span>
                        <span className="text-[#00FF9D] font-semibold truncate block">
                          {fam.variant_distribution}
                        </span>
                      </div>
                    </div>

                    {/* overall confidence */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center text-xs font-mono">
                        <span className="text-gray-500 uppercase tracking-wider text-[9px]">Investigation Confidence</span>
                        <span className={`px-1.5 py-0.5 rounded text-[8px] font-bold border uppercase ${getConfidenceBadgeColor(fam.investigation_confidence.level)}`}>
                          {fam.investigation_confidence.level} ({fam.investigation_confidence.score}%)
                        </span>
                      </div>
                      <div className="w-full bg-[#1A1A1E] h-1.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            fam.investigation_confidence.level === "High"
                              ? "bg-[#00FF9D]"
                              : fam.investigation_confidence.level === "Medium"
                              ? "bg-[#F59E0B]"
                              : "bg-[#FF3366]"
                          }`}
                          style={{ width: `${fam.investigation_confidence.score}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Narrative Summary */}
                    {fam.investigation_narrative && (
                      <div className="border border-[rgba(255,255,255,0.05)] bg-[#07070B] p-3 rounded font-mono text-[10px] text-gray-400 relative">
                        <Sparkles className="absolute right-2 top-2 h-3.5 w-3.5 text-gray-600" />
                        <span className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1 font-semibold">INTELLIGENCE NARRATIVE</span>
                        <p className="line-clamp-3 leading-relaxed">
                          {fam.investigation_narrative}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 flex justify-between items-center text-xs font-mono">
                    <span className="text-[10px] text-gray-600">CASE ID: {fam.case_id}</span>
                    {fam.most_probable_origin_id && (
                      <Link
                        href={`/media/${fam.most_probable_origin_id}`}
                        className="inline-flex items-center gap-1 text-[#00E5FF] hover:underline uppercase text-[10px] font-bold"
                      >
                        Explore DNA Profile
                        <ArrowRight className="h-3 w-3" />
                      </Link>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function intPercent(score: number) {
  return Math.round(score * 100);
}
