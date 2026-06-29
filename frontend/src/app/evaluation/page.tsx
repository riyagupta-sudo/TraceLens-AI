"use client";

import React, { useState, useEffect } from "react";
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell 
} from "recharts";
import { 
  Shield, Cpu, AlertTriangle, CheckCircle, BarChart3, Target, RefreshCw, 
  Download, Play, Database, Camera, Layers, Settings, FileText, HelpCircle 
} from "lucide-react";

export default function EvaluationDashboard() {
  const backendUrl = "http://127.0.0.1:8000";
  const [modelVersion, setModelVersion] = useState<"v1" | "v2">("v2");
  const [activeTab, setActiveTab] = useState<"performance" | "calibration" | "dataset" | "benchmark">("performance");
  
  // States for API data
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [benchmarkStats, setBenchmarkStats] = useState<any>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null);
  
  // Loading & Action states
  const [loadingDashboard, setLoadingDashboard] = useState<boolean>(true);
  const [loadingBenchmarkStats, setLoadingBenchmarkStats] = useState<boolean>(true);
  const [runningEvaluation, setRunningEvaluation] = useState<boolean>(false);
  const [seedingDataset, setSeedingDataset] = useState<boolean>(false);
  const [notification, setNotification] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Auto-hide notification
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // Fetch dashboard stats when model version changes
  useEffect(() => {
    async function fetchDashboard() {
      setLoadingDashboard(true);
      try {
        const res = await fetch(`${backendUrl}/api/evaluation/dashboard?model_version=${modelVersion}`);
        if (!res.ok) throw new Error("Failed to fetch evaluation dashboard data.");
        const data = await res.json();
        setDashboardData(data);
      } catch (err: any) {
        setNotification({ type: "error", message: err.message || "Error loading evaluation metrics." });
      } finally {
        setLoadingDashboard(false);
      }
    }
    fetchDashboard();
  }, [modelVersion]);

  // Fetch benchmark stats on mount / tab change
  const fetchBenchmarkStats = async () => {
    setLoadingBenchmarkStats(true);
    try {
      const res = await fetch(`${backendUrl}/api/benchmark/stats`);
      if (!res.ok) throw new Error("Failed to fetch benchmark stats.");
      const data = await res.json();
      setBenchmarkStats(data);
    } catch (err: any) {
      setNotification({ type: "error", message: err.message || "Error loading benchmark stats." });
    } finally {
      setLoadingBenchmarkStats(false);
    }
  };

  useEffect(() => {
    if (activeTab === "benchmark") {
      fetchBenchmarkStats();
    }
  }, [activeTab]);

  // Run Benchmark Seeding
  const handleSeedBenchmark = async () => {
    setSeedingDataset(true);
    try {
      const res = await fetch(`${backendUrl}/api/benchmark/seed`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to seed benchmark dataset.");
      const data = await res.json();
      setNotification({ type: "success", message: data.message || "Benchmark dataset seeded successfully." });
      await fetchBenchmarkStats();
    } catch (err: any) {
      setNotification({ type: "error", message: err.message || "Error seeding benchmark dataset." });
    } finally {
      setSeedingDataset(false);
    }
  };

  // Run Benchmark Evaluation
  const handleEvaluateBenchmark = async () => {
    setRunningEvaluation(true);
    setBenchmarkResults(null);
    try {
      const res = await fetch(`${backendUrl}/api/benchmark/evaluate?model_version=${modelVersion}`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to evaluate benchmark dataset.");
      const data = await res.json();
      setBenchmarkResults(data);
      setNotification({ type: "success", message: `Evaluation completed. Overall Accuracy: ${(data.overall_accuracy * 100).toFixed(2)}%` });
      await fetchBenchmarkStats();
    } catch (err: any) {
      setNotification({ type: "error", message: err.message || "Error evaluating benchmark dataset." });
    } finally {
      setRunningEvaluation(false);
    }
  };

  // Prepare data for reliability diagram
  const getReliabilityData = () => {
    if (!dashboardData?.calibration?.reliability_diagram) return [];
    return dashboardData.calibration.reliability_diagram.map((item: any) => ({
      range: item.range,
      "Expected Accuracy": item.confidence,
      "Actual Accuracy": item.accuracy,
      "Sample Count": item.count
    }));
  };

  // Prepare data for confidence histogram
  const getHistogramData = () => {
    if (!dashboardData?.calibration?.confidence_histogram) return [];
    return dashboardData.calibration.confidence_histogram.map((item: any) => ({
      range: item.range,
      "Frequency": item.count
    }));
  };

  // Prepare data for probability distribution
  const getDistributionData = () => {
    if (!dashboardData?.charts?.probability_distribution) return [];
    const realProbs = dashboardData.charts.probability_distribution.real || [];
    const aiProbs = dashboardData.charts.probability_distribution.ai || [];
    
    // Bin probabilities into 20 bins (0.05 increments)
    const bins = Array.from({ length: 20 }, (_, i) => ({
      threshold: ((i + 1) * 0.05).toFixed(2),
      "Authentic Media (Real)": 0,
      "AI Generated (Fake)": 0
    }));

    realProbs.forEach((p: number) => {
      const binIdx = Math.min(19, Math.floor(p / 0.05));
      bins[binIdx]["Authentic Media (Real)"] += 1;
    });

    aiProbs.forEach((p: number) => {
      const binIdx = Math.min(19, Math.floor(p / 0.05));
      bins[binIdx]["AI Generated (Fake)"] += 1;
    });

    return bins;
  };

  // Dataset Pie Chart data
  const getDatasetPieData = () => {
    if (!dashboardData?.dataset_statistics) return [];
    const stats = dashboardData.dataset_statistics;
    return [
      { name: "Authentic Photos", value: stats.real_count, color: "#00FF9D" },
      { name: "AI-Generated Media", value: stats.ai_count, color: "#FF3366" },
      { name: "Screenshot Detections", value: stats.screenshots_count, color: "#00E5FF" },
      { name: "Edited Variants", value: stats.edited_count, color: "#7C3AED" }
    ];
  };

  const performance = dashboardData?.model_performance;
  const calibration = dashboardData?.calibration;
  const datasetStats = dashboardData?.dataset_statistics;

  return (
    <div className="space-y-6">
      {/* Header and Model Version Selector */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-[rgba(255,255,255,0.06)] pb-4">
        <div>
          <h1 className="font-mono text-2xl font-black tracking-wider text-white flex items-center gap-2">
            <Cpu className="h-6 w-6 text-[#00E5FF] animate-pulse" />
            AI DETECTOR EVALUATION CENTER
          </h1>
          <p className="font-mono text-[10px] text-gray-500 uppercase tracking-widest mt-1">
            Analyze calibrational characteristics, training baselines, and cross-platform benchmark accuracies.
          </p>
        </div>
        
        {/* Model Selector Toggle */}
        <div className="flex items-center gap-1.5 bg-[#121212] border border-[rgba(255,255,255,0.06)] rounded-lg p-1">
          <button
            onClick={() => setModelVersion("v1")}
            className={`font-mono text-xs font-bold tracking-wider px-3.5 py-2 rounded-md transition-all ${
              modelVersion === "v1"
                ? "bg-gradient-to-r from-red-500/20 to-orange-500/20 text-orange-400 border border-orange-500/30 shadow-[0_0_8px_rgba(249,115,22,0.2)]"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            DETECTOR V1 (BASELINE)
          </button>
          <button
            onClick={() => setModelVersion("v2")}
            className={`font-mono text-xs font-bold tracking-wider px-3.5 py-2 rounded-md transition-all ${
              modelVersion === "v2"
                ? "bg-gradient-to-r from-[#7C3AED]/20 to-[#00E5FF]/20 text-[#00E5FF] border border-[#00E5FF]/30 shadow-[0_0_8px_rgba(0,229,255,0.2)]"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            DETECTOR V2 (PRO-LEVEL)
          </button>
        </div>
      </div>

      {/* Notifications */}
      {notification && (
        <div className={`cyber-card p-4 flex items-center gap-3 border ${
          notification.type === "success" 
            ? "border-[#00FF9D]/30 bg-[#00FF9D]/5 text-[#00FF9D]" 
            : "border-[#FF3366]/30 bg-[#FF3366]/5 text-[#FF3366]"
        }`}>
          {notification.type === "success" ? <CheckCircle className="h-5 w-5 shrink-0" /> : <AlertTriangle className="h-5 w-5 shrink-0" />}
          <span className="font-mono text-xs">{notification.message}</span>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex border-b border-[rgba(255,255,255,0.04)] overflow-x-auto gap-2">
        <button
          onClick={() => setActiveTab("performance")}
          className={`font-mono text-xs font-bold tracking-wider px-5 py-3 border-b-2 transition-all ${
            activeTab === "performance"
              ? "border-[#00E5FF] text-white bg-[rgba(0,229,255,0.02)]"
              : "border-transparent text-gray-500 hover:text-gray-300 hover:bg-[rgba(255,255,255,0.01)]"
          }`}
        >
          MODEL PERFORMANCE
        </button>
        <button
          onClick={() => setActiveTab("calibration")}
          className={`font-mono text-xs font-bold tracking-wider px-5 py-3 border-b-2 transition-all ${
            activeTab === "calibration"
              ? "border-[#00FF9D] text-white bg-[rgba(0,255,157,0.02)]"
              : "border-transparent text-gray-500 hover:text-gray-300 hover:bg-[rgba(255,255,255,0.01)]"
          }`}
        >
          CALIBRATION & reliability
        </button>
        <button
          onClick={() => setActiveTab("dataset")}
          className={`font-mono text-xs font-bold tracking-wider px-5 py-3 border-b-2 transition-all ${
            activeTab === "dataset"
              ? "border-[#7C3AED] text-white bg-[rgba(124,58,237,0.02)]"
              : "border-transparent text-gray-500 hover:text-gray-300 hover:bg-[rgba(255,255,255,0.01)]"
          }`}
        >
          DATASET STATISTICS
        </button>
        <button
          onClick={() => setActiveTab("benchmark")}
          className={`font-mono text-xs font-bold tracking-wider px-5 py-3 border-b-2 transition-all ${
            activeTab === "benchmark"
              ? "border-yellow-500 text-white bg-[rgba(234,179,8,0.02)]"
              : "border-transparent text-gray-500 hover:text-gray-300 hover:bg-[rgba(255,255,255,0.01)]"
          }`}
        >
          BENCHMARK MANAGER
        </button>
      </div>

      {/* Main Content Area */}
      {loadingDashboard && activeTab !== "benchmark" ? (
        <div className="cyber-card p-12 flex flex-col items-center justify-center gap-4">
          <RefreshCw className="h-8 w-8 text-[#00E5FF] animate-spin" />
          <p className="font-mono text-xs text-gray-400">Loading neural evaluation database...</p>
        </div>
      ) : (
        <>
          {/* TAB 1: MODEL PERFORMANCE */}
          {activeTab === "performance" && performance && (
            <div className="space-y-6">
              {/* Top Level Summary Cards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="cyber-card p-5 space-y-2 relative overflow-hidden">
                  <div className="absolute right-3 top-3 h-10 w-10 text-[rgba(255,255,255,0.02)] flex items-center justify-center">
                    <Target className="h-8 w-8 text-gray-800" />
                  </div>
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">ACCURACY</span>
                  <span className="block font-mono text-3xl font-black text-white glow-text-cyan">
                    {(performance.accuracy * 100).toFixed(1)}%
                  </span>
                  <span className="block font-mono text-[8px] text-emerald-400">
                    {modelVersion === "v2" ? "+9.5% improvement" : "Baseline accuracy"}
                  </span>
                </div>

                <div className="cyber-card p-5 space-y-2 relative overflow-hidden">
                  <div className="absolute right-3 top-3 h-10 w-10 text-[rgba(255,255,255,0.02)] flex items-center justify-center">
                    <Shield className="h-8 w-8 text-gray-800" />
                  </div>
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">F1-SCORE</span>
                  <span className="block font-mono text-3xl font-black text-[#00FF9D] glow-text-green">
                    {(performance.f1_score * 100).toFixed(1)}%
                  </span>
                  <span className="block font-mono text-[8px] text-[#00FF9D]">
                    Harmonic mean of precision & recall
                  </span>
                </div>

                <div className="cyber-card p-5 space-y-2 relative overflow-hidden">
                  <div className="absolute right-3 top-3 h-10 w-10 text-[rgba(255,255,255,0.02)] flex items-center justify-center">
                    <BarChart3 className="h-8 w-8 text-gray-800" />
                  </div>
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">ROC AUC</span>
                  <span className="block font-mono text-3xl font-black text-[#7C3AED]">
                    {(performance.roc_auc * 100).toFixed(1)}%
                  </span>
                  <span className="block font-mono text-[8px] text-gray-400">
                    Discriminative capabilities
                  </span>
                </div>

                <div className="cyber-card p-5 space-y-2 relative overflow-hidden">
                  <div className="absolute right-3 top-3 h-10 w-10 text-[rgba(255,255,255,0.02)] flex items-center justify-center">
                    <AlertTriangle className="h-8 w-8 text-gray-800" />
                  </div>
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">FALSE POSITIVES</span>
                  <span className="block font-mono text-3xl font-black text-[#FF3366]">
                    {(performance.fpr * 100).toFixed(1)}%
                  </span>
                  <span className="block font-mono text-[8px] text-red-400">
                    {modelVersion === "v2" ? "-8.0% FPR Reduction" : "False alarm rate"}
                  </span>
                </div>
              </div>

              {/* ROC Curve Graph & Confusion Matrix */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* ROC Curve Chart */}
                <div className="cyber-card p-6 lg:col-span-2 space-y-4">
                  <div className="flex items-center justify-between border-b border-[rgba(255,255,255,0.06)] pb-3">
                    <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                      <Target className="h-4 w-4 text-[#00E5FF]" />
                      Receiver Operating Characteristic (ROC) Curve
                    </h3>
                  </div>
                  <div className="h-80 w-full font-mono text-[10px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={dashboardData?.charts?.roc_curve || []}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient id="rocColor" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.4}/>
                            <stop offset="95%" stopColor="#00E5FF" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                        <XAxis 
                          dataKey="fpr" 
                          stroke="#555" 
                          label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -5, fill: '#888' }} 
                        />
                        <YAxis 
                          stroke="#555" 
                          label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', offset: 10, fill: '#888' }} 
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: "#121212", borderColor: "rgba(255,255,255,0.08)", color: "#fff" }}
                          itemStyle={{ color: "#00E5FF" }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="tpr" 
                          stroke="#00E5FF" 
                          fillOpacity={1} 
                          fill="url(#rocColor)" 
                          strokeWidth={2}
                          name="True Positive Rate"
                        />
                        {/* Reference Line */}
                        <Line 
                          type="monotone" 
                          dataKey={(d) => d.fpr} 
                          stroke="#FF3366" 
                          strokeDasharray="5 5" 
                          dot={false}
                          name="Random Guess" 
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Confusion Matrix Card */}
                <div className="cyber-card p-6 space-y-4">
                  <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                    <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                      <Layers className="h-4 w-4 text-[#00FF9D]" />
                      Confusion Matrix
                    </h3>
                  </div>

                  {dashboardData?.charts?.confusion_matrix && (
                    <div className="space-y-4 pt-2">
                      <div className="grid grid-cols-3 gap-2 text-center font-mono text-[9px] text-gray-500 font-bold uppercase">
                        <div></div>
                        <div>PREDICTED REAL</div>
                        <div>PREDICTED FAKE</div>
                      </div>

                      {/* Row 1: Actual Real */}
                      <div className="grid grid-cols-3 gap-2 items-center text-center">
                        <div className="font-mono text-[9px] font-bold text-gray-500 uppercase text-left">ACTUAL REAL</div>
                        <div className="bg-[#00FF9D]/5 border border-[#00FF9D]/20 rounded-md p-4">
                          <span className="block font-mono text-lg font-black text-[#00FF9D]">
                            {dashboardData.charts.confusion_matrix[0][0]}
                          </span>
                          <span className="block font-mono text-[7px] text-gray-500 uppercase mt-0.5">True Negative</span>
                        </div>
                        <div className="bg-[#FF3366]/5 border border-[#FF3366]/20 rounded-md p-4">
                          <span className="block font-mono text-lg font-black text-[#FF3366]">
                            {dashboardData.charts.confusion_matrix[0][1]}
                          </span>
                          <span className="block font-mono text-[7px] text-gray-500 uppercase mt-0.5">False Positive</span>
                        </div>
                      </div>

                      {/* Row 2: Actual Fake */}
                      <div className="grid grid-cols-3 gap-2 items-center text-center">
                        <div className="font-mono text-[9px] font-bold text-gray-500 uppercase text-left">ACTUAL FAKE</div>
                        <div className="bg-[#FF3366]/5 border border-[#FF3366]/20 rounded-md p-4">
                          <span className="block font-mono text-lg font-black text-[#FF3366]">
                            {dashboardData.charts.confusion_matrix[1][0]}
                          </span>
                          <span className="block font-mono text-[7px] text-gray-500 uppercase mt-0.5">False Negative</span>
                        </div>
                        <div className="bg-[#00FF9D]/5 border border-[#00FF9D]/20 rounded-md p-4">
                          <span className="block font-mono text-lg font-black text-[#00FF9D]">
                            {dashboardData.charts.confusion_matrix[1][1]}
                          </span>
                          <span className="block font-mono text-[7px] text-gray-500 uppercase mt-0.5">True Positive</span>
                        </div>
                      </div>

                      <div className="border-t border-[rgba(255,255,255,0.04)] pt-3 text-[10px] text-gray-500 font-mono space-y-1.5">
                        <div className="flex justify-between">
                          <span>Sensitivity / Recall:</span>
                          <span className="text-white font-bold">{(performance.recall * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Specificity:</span>
                          <span className="text-white font-bold">{((1 - performance.fpr) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>False Negative Rate:</span>
                          <span className="text-[#FF3366] font-bold">{(performance.fnr * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: CALIBRATION & RELIABILITY */}
          {activeTab === "calibration" && calibration && (
            <div className="space-y-6">
              {/* Calibration Stats Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="cyber-card p-5 space-y-2">
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">Expected Calibration Error (ECE)</span>
                  <span className="block font-mono text-3xl font-black text-[#00FF9D] glow-text-green">
                    {(calibration.ece * 100).toFixed(3)}%
                  </span>
                  <p className="font-mono text-[9px] text-gray-400 leading-relaxed">
                    Quantifies the difference between confidence levels and actual accuracy scores. 
                    Lower ECE values indicate highly trustworthy, calibrated probabilities.
                  </p>
                </div>

                <div className="cyber-card p-5 space-y-2">
                  <span className="block font-mono text-[9px] text-gray-500 uppercase tracking-widest font-black">Brier Score</span>
                  <span className="block font-mono text-3xl font-black text-[#00E5FF] glow-text-cyan">
                    {calibration.brier_score.toFixed(4)}
                  </span>
                  <p className="font-mono text-[9px] text-gray-400 leading-relaxed">
                    Measures the mean squared difference between predicted probabilities and target outcomes. 
                    An indicator of both accuracy and confidence calibration.
                  </p>
                </div>
              </div>

              {/* Reliability Diagram & Prediction Histograms */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Reliability Diagram */}
                <div className="cyber-card p-6 space-y-4">
                  <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                    <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                      <Target className="h-4 w-4 text-[#00FF9D]" />
                      Reliability Diagram (Calibrational Alignment)
                    </h3>
                  </div>
                  <div className="h-72 w-full font-mono text-[10px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={getReliabilityData()}
                        margin={{ top: 10, right: 30, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                        <XAxis dataKey="range" stroke="#555" />
                        <YAxis stroke="#555" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip
                          contentStyle={{ backgroundColor: "#121212", borderColor: "rgba(255,255,255,0.08)", color: "#fff" }}
                        />
                        <Legend verticalAlign="top" height={36} iconType="rect" />
                        <Bar dataKey="Expected Accuracy" fill="#1f2937" name="Target Accuracy (Ideal)" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Actual Accuracy" fill="#00FF9D" name="Actual Accuracy (Model)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Probability Distribution Graph */}
                <div className="cyber-card p-6 space-y-4">
                  <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                    <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                      <Layers className="h-4 w-4 text-[#7C3AED]" />
                      Probability Density Distributions
                    </h3>
                  </div>
                  <div className="h-72 w-full font-mono text-[10px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={getDistributionData()}
                        margin={{ top: 10, right: 30, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                        <XAxis dataKey="threshold" stroke="#555" label={{ value: 'Neural AI Confidence Score', position: 'insideBottom', offset: -5, fill: '#888' }} />
                        <YAxis stroke="#555" />
                        <Tooltip
                          contentStyle={{ backgroundColor: "#121212", borderColor: "rgba(255,255,255,0.08)", color: "#fff" }}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Area type="monotone" dataKey="Authentic Media (Real)" stroke="#00FF9D" fill="#00FF9D" fillOpacity={0.1} strokeWidth={2} />
                        <Area type="monotone" dataKey="AI Generated (Fake)" stroke="#FF3366" fill="#FF3366" fillOpacity={0.1} strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Confidence Frequency Histogram */}
              <div className="cyber-card p-6 space-y-4">
                <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                  <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-[#00E5FF]" />
                    Confidence Bin Frequency Distribution
                  </h3>
                </div>
                <div className="h-64 w-full font-mono text-[10px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={getHistogramData()}
                      margin={{ top: 10, right: 30, left: -20, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                      <XAxis dataKey="range" stroke="#555" />
                      <YAxis stroke="#555" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#121212", borderColor: "rgba(255,255,255,0.08)", color: "#fff" }}
                        itemStyle={{ color: "#00E5FF" }}
                      />
                      <Bar dataKey="Frequency" fill="#00E5FF" radius={[4, 4, 0, 0]} name="Samples in Bin" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: DATASET STATISTICS */}
          {activeTab === "dataset" && datasetStats && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Dataset Distribution Pie Chart */}
              <div className="cyber-card p-6 lg:col-span-2 space-y-4">
                <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                  <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                    <Database className="h-4 w-4 text-[#7C3AED]" />
                    Neural Evaluation Library Distribution
                  </h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={getDatasetPieData()}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={90}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {getDatasetPieData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: "#121212", borderColor: "rgba(255,255,255,0.08)", color: "#fff" }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="space-y-4">
                    <h4 className="font-mono text-[10px] font-bold text-gray-500 uppercase">Core Dataset Classes</h4>
                    <div className="space-y-3 font-mono text-xs">
                      {getDatasetPieData().map((entry, idx) => (
                        <div key={idx} className="flex items-center justify-between border-b border-[rgba(255,255,255,0.02)] pb-2">
                          <div className="flex items-center gap-2">
                            <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: entry.color }} />
                            <span className="text-gray-300 font-semibold">{entry.name}</span>
                          </div>
                          <span className="text-white font-black">{entry.value.toLocaleString()} files</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Camera Model Sources */}
              <div className="cyber-card p-6 space-y-4">
                <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                  <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                    <Camera className="h-4 w-4 text-[#00FF9D]" />
                    Metadata Camera Profiles
                  </h3>
                </div>

                <div className="space-y-3 font-mono text-xs max-h-[300px] overflow-y-auto pr-1">
                  {Object.entries(datasetStats.camera_sources).map(([camera, count]: any, idx) => (
                    <div key={idx} className="flex items-center justify-between border-b border-[rgba(255,255,255,0.03)] pb-2 hover:bg-[rgba(255,255,255,0.01)] p-1 rounded transition-colors">
                      <span className="text-gray-400 font-bold truncate max-w-[190px]">{camera}</span>
                      <span className="text-[#00FF9D] font-black">{count} files</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: BENCHMARK MANAGER */}
          {activeTab === "benchmark" && (
            <div className="space-y-6">
              {/* Controls bar */}
              <div className="cyber-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="font-mono text-sm font-bold text-white uppercase">Forensic Benchmark Manager</h3>
                  <p className="font-mono text-[9px] text-gray-500 uppercase mt-1">
                    Evaluate baseline models against isolated camera captures, screenshot variants, and AI generators.
                  </p>
                </div>
                
                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    onClick={handleSeedBenchmark}
                    disabled={seedingDataset || runningEvaluation}
                    className="font-mono text-xs font-bold tracking-widest bg-gray-900 border border-[rgba(255,255,255,0.08)] text-gray-300 hover:bg-gray-800 disabled:opacity-50 px-4 py-2.5 rounded-md flex items-center gap-1.5 transition-all"
                  >
                    {seedingDataset ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Database className="h-3.5 w-3.5" />}
                    SEED IMAGES
                  </button>
                  <button
                    onClick={handleEvaluateBenchmark}
                    disabled={runningEvaluation || seedingDataset}
                    className="font-mono text-xs font-bold tracking-widest bg-gradient-to-r from-[#7C3AED] to-[#00E5FF] hover:opacity-90 disabled:opacity-50 px-4 py-2.5 rounded-md flex items-center gap-1.5 shadow-[0_0_10px_rgba(0,229,255,0.3)] transition-all"
                  >
                    {runningEvaluation ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                    RUN BENCHMARK
                  </button>
                  <a
                    href={`${backendUrl}/api/benchmark/report?model_version=${modelVersion}`}
                    className="font-mono text-xs font-bold tracking-widest bg-[#121212] border border-[rgba(0,229,255,0.2)] text-[#00E5FF] hover:bg-[#00E5FF]/10 px-4 py-2.5 rounded-md flex items-center gap-1.5 transition-all"
                  >
                    <Download className="h-3.5 w-3.5" />
                    EXPORT REPORT
                  </a>
                </div>
              </div>

              {loadingBenchmarkStats ? (
                <div className="cyber-card p-12 flex flex-col items-center justify-center gap-4">
                  <RefreshCw className="h-8 w-8 text-[#00E5FF] animate-spin" />
                  <p className="font-mono text-xs text-gray-400">Scanning benchmark folder structures...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Category Counts and evaluation details */}
                  <div className="cyber-card p-6 lg:col-span-2 space-y-4">
                    <div className="border-b border-[rgba(255,255,255,0.06)] pb-3 flex justify-between items-center">
                      <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                        <Layers className="h-4 w-4 text-[#00E5FF]" />
                        Benchmark Folder Directories
                      </h3>
                      <span className="font-mono text-[9px] font-black uppercase text-gray-500 bg-[#121212] border border-[rgba(255,255,255,0.04)] px-2 py-0.5 rounded">
                        Total Files: {benchmarkStats?.total_images ?? 0}
                      </span>
                    </div>

                    <div className="space-y-4 pt-2">
                      {benchmarkStats?.categories && Object.entries(benchmarkStats.categories).map(([path, cat]: any, idx: number) => {
                        const score = benchmarkResults?.categories?.[path]?.accuracy ?? null;
                        return (
                          <div key={idx} className="space-y-1.5">
                            <div className="flex items-center justify-between font-mono text-[10px]">
                              <span className="text-white font-bold">{cat.name} <span className="text-gray-600 font-mono">({path})</span></span>
                              <div className="flex items-center gap-2 font-mono text-[9px]">
                                <span className="text-gray-500 uppercase">{cat.count} Files</span>
                                {score !== null && (
                                  <span className={`font-black uppercase tracking-wider ${
                                    score >= 0.9 ? "text-[#00FF9D]" : score >= 0.7 ? "text-[#00E5FF]" : "text-yellow-500"
                                  }`}>
                                    Accuracy: {(score * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="relative h-2 w-full bg-[#121212] border border-[rgba(255,255,255,0.03)] rounded-full overflow-hidden">
                              <div 
                                className={`absolute h-full rounded-full transition-all duration-500 ${
                                  score !== null 
                                    ? score >= 0.9 ? "bg-[#00FF9D]" : score >= 0.7 ? "bg-[#00E5FF]" : "bg-yellow-500"
                                    : "bg-gray-800"
                                }`}
                                style={{ width: score !== null ? `${score * 100}%` : `${Math.min(100, (cat.count / 10) * 100)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Benchmark Overview & Results */}
                  <div className="cyber-card p-6 space-y-4">
                    <div className="border-b border-[rgba(255,255,255,0.06)] pb-3">
                      <h3 className="font-mono text-xs font-black tracking-widest text-white uppercase flex items-center gap-2">
                        <FileText className="h-4 w-4 text-yellow-500" />
                        Active Benchmark Results
                      </h3>
                    </div>

                    {runningEvaluation ? (
                      <div className="flex flex-col items-center justify-center p-12 text-center space-y-4">
                        <RefreshCw className="h-8 w-8 text-yellow-500 animate-spin" />
                        <span className="block font-mono text-xs text-white font-bold">Evaluating Media Files...</span>
                        <span className="block font-mono text-[9px] text-gray-500 max-w-[200px] leading-relaxed">
                          Running calibrated AI detector neural inference and features heuristics mapping.
                        </span>
                      </div>
                    ) : benchmarkResults ? (
                      <div className="space-y-4">
                        <div className="text-center bg-gray-900/40 border border-[rgba(255,255,255,0.04)] rounded-md p-6 space-y-2">
                          <span className="block font-mono text-[8px] text-gray-500 uppercase tracking-widest font-black">OVERALL BENCHMARK ACCURACY</span>
                          <span className="block font-mono text-4xl font-black text-[#00FF9D] glow-text-green">
                            {(benchmarkResults.overall_accuracy * 100).toFixed(2)}%
                          </span>
                          <span className="block font-mono text-[9px] text-gray-500 uppercase">
                            Evaluated {benchmarkResults.total_images} files in {Object.keys(benchmarkResults.categories).length} categories
                          </span>
                        </div>

                        <div className="font-mono text-xs space-y-2">
                          <div className="flex justify-between border-b border-[rgba(255,255,255,0.02)] pb-2 text-[10px] text-gray-500">
                            <span>CATEGORY</span>
                            <span>ACCURACY</span>
                          </div>
                          {Object.values(benchmarkResults.categories).map((cat: any, idx) => (
                            <div key={idx} className="flex justify-between border-b border-[rgba(255,255,255,0.02)] pb-2">
                              <span className="text-gray-300 font-bold truncate max-w-[170px]">{cat.name}</span>
                              <span className={`font-black ${
                                cat.accuracy >= 0.9 ? "text-[#00FF9D]" : cat.accuracy >= 0.7 ? "text-[#00E5FF]" : "text-yellow-500"
                              }`}>
                                {(cat.accuracy * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center p-12 text-center text-gray-600">
                        <HelpCircle className="h-10 w-10 text-gray-700 mb-2" />
                        <span className="block font-mono text-xs font-semibold uppercase">No active evaluation</span>
                        <span className="block font-mono text-[9px] text-gray-500 mt-1 leading-relaxed">
                          Click "RUN BENCHMARK" above to analyze the loaded benchmark folders on the active model.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
