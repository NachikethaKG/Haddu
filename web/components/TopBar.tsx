"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  TreePine,
  AlertTriangle,
  Calendar,
  Filter,
  X,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type TierFilterOption = "ALL" | "CRITICAL_ONLY" | "HIGH_PLUS_CRITICAL";

export interface TopBarProps {
  totalAcres: number;
  activeAlerts: number;
  monitoredAreaKm2?: number;
  selectedTier: TierFilterOption;
  onTierChange: (tier: TierFilterOption) => void;
  startDate: string;
  onStartDateChange: (date: string) => void;
  endDate: string;
  onEndDateChange: (date: string) => void;
  onResetFilters: () => void;
  rasterToggleSlot?: React.ReactNode;
}

export default function TopBar({
  totalAcres,
  activeAlerts,
  monitoredAreaKm2 = 1255.64,
  selectedTier,
  onTierChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  onResetFilters,
  rasterToggleSlot,
}: TopBarProps) {
  const [isFilterDropdownOpen, setIsFilterDropdownOpen] = useState(false);
  const [isDatePopoverOpen, setIsDatePopoverOpen] = useState(false);

  const tierLabels: Record<TierFilterOption, string> = {
    ALL: "All Violations",
    CRITICAL_ONLY: "Critical Only (Reserve Forest)",
    HIGH_PLUS_CRITICAL: "High + Critical",
  };

  const hasActiveFilters =
    selectedTier !== "ALL" || startDate !== "" || endDate !== "";

  return (
    <header className="absolute top-4 inset-x-4 z-10 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
      {/* Brand Badge (Left) */}
      <div className="flex items-center gap-3 pointer-events-auto">
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-2xl px-4 py-2.5 shadow-2xl flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-lg shadow-emerald-950/50">
            <ShieldAlert className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xs font-bold tracking-wider text-white uppercase">
                HADDU <span className="text-emerald-400 font-mono text-[10px] font-normal">v1.0</span>
              </h1>
              <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                LIVE
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-medium truncate max-w-[160px] sm:max-w-none">
              Thirthahalli Forest Encroachment Monitor
            </p>
          </div>
        </div>
      </div>

      {/* KPI Stats Badges (Center) */}
      <div className="hidden lg:flex items-center gap-2 pointer-events-auto">
        {/* KPI: Total Cleared Acres */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl px-3.5 py-2 shadow-xl flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
            <AlertTriangle className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-semibold tracking-wider text-slate-400">
              Total Cleared
            </div>
            <div className="text-xs font-bold font-mono text-white flex items-baseline gap-1">
              <span>{totalAcres.toFixed(1)} ac</span>
              <span className="text-[10px] font-normal text-slate-400">
                ({(totalAcres * 40).toFixed(0)} gt)
              </span>
            </div>
          </div>
        </div>

        {/* KPI: Active Disturbance Alerts */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl px-3.5 py-2 shadow-xl flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400">
            <ShieldAlert className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-semibold tracking-wider text-slate-400">
              Active Alerts
            </div>
            <div className="text-xs font-bold font-mono text-amber-300 flex items-baseline gap-1">
              <span>{activeAlerts}</span>
              <span className="text-[10px] font-normal text-slate-400">polygons</span>
            </div>
          </div>
        </div>

        {/* KPI: Monitored Forest Extent */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl px-3.5 py-2 shadow-xl flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <TreePine className="w-3.5 h-3.5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-semibold tracking-wider text-slate-400">
              Monitored Area
            </div>
            <div className="text-xs font-bold font-mono text-emerald-300">
              {monitoredAreaKm2.toLocaleString(undefined, { maximumFractionDigits: 1 })} km²
            </div>
          </div>
        </div>
      </div>

      {/* Filter Bar & Controls (Right) */}
      <div className="flex items-center gap-2 pointer-events-auto">
        {/* Violation Tier Filter Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setIsFilterDropdownOpen((prev) => !prev)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium backdrop-blur-md transition-all shadow-xl border",
              selectedTier !== "ALL"
                ? "bg-emerald-600/30 text-emerald-300 border-emerald-500/50"
                : "bg-slate-900/90 text-slate-200 border-slate-700/80 hover:bg-slate-800"
            )}
            title="Filter by Violation Tier"
          >
            <Filter className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold">{tierLabels[selectedTier]}</span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {isFilterDropdownOpen && (
            <div className="absolute top-full mt-2 right-0 w-60 bg-slate-900/95 backdrop-blur-xl border border-slate-700/90 rounded-2xl p-1.5 shadow-2xl z-20 space-y-1">
              <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-800">
                Violation Severity
              </div>

              {(["ALL", "CRITICAL_ONLY", "HIGH_PLUS_CRITICAL"] as TierFilterOption[]).map(
                (option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => {
                      onTierChange(option);
                      setIsFilterDropdownOpen(false);
                    }}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg text-xs transition-colors flex items-center justify-between",
                      selectedTier === option
                        ? "bg-emerald-600/30 text-emerald-300 font-medium"
                        : "text-slate-300 hover:bg-slate-800 hover:text-white"
                    )}
                  >
                    <span>{tierLabels[option]}</span>
                    {selectedTier === option && (
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    )}
                  </button>
                )
              )}
            </div>
          )}
        </div>

        {/* Date Range Selector */}
        <div className="relative">
          <button
            type="button"
            onClick={() => setIsDatePopoverOpen((prev) => !prev)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium backdrop-blur-md transition-all shadow-xl border",
              startDate || endDate
                ? "bg-cyan-600/30 text-cyan-300 border-cyan-500/50"
                : "bg-slate-900/90 text-slate-200 border-slate-700/80 hover:bg-slate-800"
            )}
            title="Filter by Detection Date Range"
          >
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold">
              {startDate || endDate
                ? `${startDate || "Start"} → ${endDate || "Now"}`
                : "Date Range"}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {isDatePopoverOpen && (
            <div className="absolute top-full mt-2 right-0 w-64 bg-slate-900/95 backdrop-blur-xl border border-slate-700/90 rounded-2xl p-4 shadow-2xl z-20 space-y-3 text-xs text-slate-200">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-800 font-semibold text-white">
                <span className="flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Detection Window</span>
                </span>
                {(startDate || endDate) && (
                  <button
                    type="button"
                    onClick={() => {
                      onStartDateChange("");
                      onEndDateChange("");
                    }}
                    className="text-[10px] text-cyan-400 hover:underline"
                  >
                    Clear Dates
                  </button>
                )}
              </div>

              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-semibold text-slate-400">
                  From (Post Date &ge;)
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => onStartDateChange(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-[10px] uppercase font-semibold text-slate-400">
                  To (Post Date &le;)
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => onEndDateChange(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Quick Presets */}
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between gap-1 text-[10px]">
                <button
                  type="button"
                  onClick={() => {
                    onStartDateChange("2026-01-01");
                    onEndDateChange("2026-03-31");
                    setIsDatePopoverOpen(false);
                  }}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
                >
                  Dry 2026
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onStartDateChange("2026-06-01");
                    onEndDateChange("2026-09-30");
                    setIsDatePopoverOpen(false);
                  }}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
                >
                  Monsoon
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onStartDateChange("");
                    onEndDateChange("");
                    setIsDatePopoverOpen(false);
                  }}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-300"
                >
                  All Time
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Clear Filters Button (shown only when filters active) */}
        {hasActiveFilters && (
          <button
            type="button"
            onClick={onResetFilters}
            className="p-2 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700/80 transition-colors shadow-lg"
            title="Reset Filters"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}

        {/* TiTiler Raster Layer Toggle Slot */}
        {rasterToggleSlot}
      </div>
    </header>
  );
}
