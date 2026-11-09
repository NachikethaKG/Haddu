"use client";

import React, { useState } from "react";
import {
  X,
  FileDown,
  ExternalLink,
  Maximize2,
  ShieldAlert,
  Calendar,
  MapPin,
  Ruler,
  TreePine,
  Waves,
  Radio,
  TrendingDown,
  Copy,
  Check,
  Scale,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { IncidentProperties, getDossierDownloadUrl } from "@/lib/api";

export interface IncidentDrawerProps {
  incident: IncidentProperties | null;
  onClose: () => void;
  onZoomToExtent?: (lat: number, lon: number, incidentId?: string) => void;
}

export default function IncidentDrawer({
  incident,
  onClose,
  onZoomToExtent,
}: IncidentDrawerProps) {
  const [copiedId, setCopiedId] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<"idle" | "success" | "error">("idle");
  const [downloadFeedback, setDownloadFeedback] = useState<string>("");

  if (!incident) return null;

  const tier = (incident.violation_tier || "MODERATE").toUpperCase();

  const tierBadgeConfig: Record<
    string,
    { label: string; bg: string; text: string; border: string; desc: string }
  > = {
    CRITICAL: {
      label: "CRITICAL BREACH",
      bg: "bg-red-500/20",
      text: "text-red-400",
      border: "border-red-500/40",
      desc: "Reserve Forest Boundary Intrusion",
    },
    HIGH: {
      label: "HIGH SEVERITY",
      bg: "bg-orange-500/20",
      text: "text-orange-400",
      border: "border-orange-500/40",
      desc: "50m Riparian Stream Buffer Breach",
    },
    MODERATE: {
      label: "MODERATE DISTURBANCE",
      bg: "bg-amber-500/20",
      text: "text-amber-400",
      border: "border-amber-500/40",
      desc: "Vegetation Loss In Buffer Corridor",
    },
  };

  const badge = tierBadgeConfig[tier] || tierBadgeConfig.MODERATE;
  const acres = Number(incident.area_acres || 0);
  const guntas = acres * 40; // 1 gunta = 101.17 sqm, 1 acre = 40 guntas
  const sqm = Number(incident.area_sqm || acres * 4046.86);
  const confidence = incident.confidence_score
    ? Math.round(Number(incident.confidence_score) * 100)
    : 85;

  const copyIncidentId = () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(incident.id);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    }
  };

  const lat = Number(incident.centroid_lat || 13.70);
  const lon = Number(incident.centroid_lon || 75.25);
  const googleMapsUrl = `https://www.google.com/maps?q=${lat.toFixed(6)},${lon.toFixed(6)}`;

  const handleDownloadDossier = async () => {
    setIsDownloading(true);
    setDownloadStatus("idle");
    setDownloadFeedback("");
    const downloadUrl = getDossierDownloadUrl(incident.id);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
      const res = await fetch(downloadUrl, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (res.ok) {
        const blob = await res.blob();
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = blobUrl;
        a.download = `haddu_dossier_${incident.id.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(blobUrl);

        setDownloadStatus("success");
        setDownloadFeedback("Legal evidence dossier downloaded successfully.");
        setTimeout(() => setDownloadStatus("idle"), 4000);
      } else if (res.status === 404) {
        setDownloadStatus("error");
        setDownloadFeedback(
          "Incident record not found on the server. Dossiers can only be generated for active database records."
        );
      } else {
        setDownloadStatus("error");
        setDownloadFeedback(`Server returned status ${res.status}. Could not generate PDF dossier.`);
      }
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      setDownloadStatus("error");
      if (err instanceof Error && err.name === "AbortError") {
        setDownloadFeedback("Dossier generation timed out after 25s. Please check backend worker.");
      } else {
        setDownloadFeedback("Could not connect to FastAPI server at http://localhost:8000.");
      }
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <>
      {/* Mobile Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-20 md:hidden"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over Inspection Panel */}
      <aside
        className={cn(
          "fixed top-0 right-0 z-30 h-full w-full sm:w-[440px]",
          "bg-slate-900/95 backdrop-blur-xl border-l border-slate-800 shadow-2xl",
          "flex flex-col transition-transform duration-300 ease-in-out",
          "text-slate-100 font-sans"
        )}
      >
        {/* Drawer Header */}
        <div className="p-5 border-b border-slate-800 flex items-start justify-between gap-4 bg-slate-950/40">
          <div className="space-y-1.5 flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border",
                  badge.bg,
                  badge.text,
                  badge.border
                )}
              >
                {badge.label}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {incident.post_date}
              </span>
            </div>

            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2 truncate">
              <ShieldAlert className="w-5 h-5 text-red-400 shrink-0" />
              <span>Canopy Disturbance Incident</span>
            </h2>

            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
              <span className="truncate">UUID: {incident.id}</span>
              <button
                type="button"
                onClick={copyIncidentId}
                className="hover:text-emerald-400 p-1 rounded transition-colors"
                title="Copy Incident UUID"
              >
                {copiedId ? (
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 text-slate-400 hover:text-white transition-colors"
            title="Close Drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Drawer Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Disturbance Area Metric Banner */}
          <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/90 border border-slate-700/80 rounded-2xl p-4 shadow-lg">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
              <span className="flex items-center gap-1.5 font-medium">
                <Ruler className="w-4 h-4 text-emerald-400" />
                Total Cleared Extent
              </span>
              <span className="font-mono text-emerald-400 text-xs">
                {sqm.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²
              </span>
            </div>

            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-3xl font-extrabold text-white tracking-tight">
                {acres.toFixed(2)}
              </span>
              <span className="text-sm font-semibold text-slate-300">acres</span>
              <span className="text-sm text-slate-500">/</span>
              <span className="text-xl font-bold text-emerald-400">
                {guntas.toFixed(1)}
              </span>
              <span className="text-sm font-semibold text-slate-300">guntas</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Standard Karnataka cadastral conversion: 1 gunta = 101.17 m² (40 guntas = 1 acre)
            </p>
          </div>

          {/* Incident Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            {/* Geographic Coordinates */}
            <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
                <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                <span>Coordinates</span>
              </div>
              <div className="text-xs font-mono text-white font-semibold">
                {incident.centroid_lat.toFixed(4)}° N
              </div>
              <div className="text-xs font-mono text-slate-400">
                {incident.centroid_lon.toFixed(4)}° E
              </div>
            </div>

            {/* Confidence Score */}
            <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
                <Radio className="w-3.5 h-3.5 text-emerald-400" />
                <span>Confidence</span>
              </div>
              <div className="text-base font-bold text-white font-mono">
                {confidence}%
              </div>
              <div className="text-[10px] text-slate-400">Statistical Persistence</div>
            </div>

            {/* Pre & Post Dates */}
            <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
                <Calendar className="w-3.5 h-3.5 text-amber-400" />
                <span>Pass Interval</span>
              </div>
              <div className="text-[11px] text-slate-300 font-mono">
                Pre: <strong className="text-white">{incident.pre_date}</strong>
              </div>
              <div className="text-[11px] text-slate-300 font-mono">
                Post: <strong className="text-white">{incident.post_date}</strong>
              </div>
            </div>

            {/* Remote Sensor & NDVI Drop */}
            <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-3">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-1">
                <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                <span>Sensor & Delta</span>
              </div>
              <div className="text-[11px] font-semibold text-white truncate">
                {incident.sensor || "Sentinel-2 MSI"}
              </div>
              <div className="text-[11px] font-mono text-red-400">
                Δ NDVI: {incident.mean_delta_ndvi ? incident.mean_delta_ndvi.toFixed(2) : "-0.32"}
              </div>
            </div>
          </div>

          {/* Legal Boundary Intersection Details */}
          <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 space-y-2">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <TreePine className="w-4 h-4 text-emerald-400" />
              <span>Boundary Classification</span>
            </h3>

            <div className="text-xs text-slate-300 space-y-1.5 pt-1">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Reserve Forest:</span>
                <span className="font-semibold text-white">
                  {incident.encroached_rf_name || "None (General Private/Revenue Tract)"}
                </span>
              </div>

              {incident.rf_area_acres && (
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">RF Encroached Area:</span>
                  <span className="font-mono text-red-400 font-medium">
                    {Number(incident.rf_area_acres).toFixed(2)} acres
                  </span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-slate-400">Riparian Buffer Breach:</span>
                <span
                  className={cn(
                    "font-medium",
                    incident.riparian_breach ? "text-orange-400 font-bold" : "text-slate-400"
                  )}
                >
                  {incident.riparian_breach ? "YES (Within 50m stream zone)" : "NO"}
                </span>
              </div>
            </div>

            {incident.riparian_breach && (
              <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-2.5 text-xs text-orange-300 flex items-start gap-2 mt-2">
                <Waves className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" />
                <p className="text-[11px] leading-relaxed">
                  Violation includes direct clearing within the 50m statutory riparian buffer of the Tunga River Basin watershed.
                </p>
              </div>
            )}
          </div>

          {/* Statutory Complaint Preview */}
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 uppercase tracking-wider">
              <Scale className="w-4 h-4 text-emerald-400" />
              <span>Enforcement Notice Citation</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed italic bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80">
              &ldquo;Formal complaint submitted under Section 64A of the Karnataka Forest Act, 1963.
              Unlawful felling, canopy disturbance, or encroachment upon gazetted forest tract is a cognizable
              offence liable for immediate eviction and recovery of restoration costs.&rdquo;
            </p>
          </div>
        </div>

        {/* Action Buttons Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/70 space-y-2.5">
          {/* Download Feedback Banner */}
          {downloadStatus !== "idle" && (
            <div
              className={cn(
                "p-2.5 rounded-xl text-xs flex items-start gap-2 border animate-in fade-in duration-200",
                downloadStatus === "success"
                  ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
                  : "bg-red-500/15 border-red-500/30 text-red-300"
              )}
            >
              <span className="text-sm shrink-0">
                {downloadStatus === "success" ? "✓" : "⚠"}
              </span>
              <p className="leading-snug">{downloadFeedback}</p>
            </div>
          )}

          {/* Primary: Download PDF Dossier */}
          <button
            type="button"
            onClick={handleDownloadDossier}
            disabled={isDownloading}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-semibold text-sm",
              "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500",
              "text-white shadow-lg shadow-emerald-950/50 transition-all",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <FileDown className="w-4 h-4" />
            <span>{isDownloading ? "Compiling Dossier PDF..." : "Download Legal Dossier (PDF)"}</span>
          </button>

          <div className="grid grid-cols-2 gap-2">
            {/* Google Maps Directions */}
            <a
              href={googleMapsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                "flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl",
                "bg-slate-800 hover:bg-slate-700 border border-slate-700/80 text-xs font-medium text-slate-200 hover:text-white",
                "transition-colors"
              )}
            >
              <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
              <span>Google Maps</span>
            </a>

            {/* Zoom to Extent */}
            <button
              type="button"
              onClick={() => {
                if (onZoomToExtent) {
                  onZoomToExtent(lat, lon, incident.id);
                }
              }}
              className={cn(
                "flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl",
                "bg-slate-800 hover:bg-slate-700 border border-slate-700/80 text-xs font-medium text-slate-200 hover:text-white",
                "transition-colors"
              )}
            >
              <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Zoom to Extent</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
