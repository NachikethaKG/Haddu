"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import type { Map as MapLibreMap } from "maplibre-gl";
import {
  Satellite,
  Sliders,
  Eye,
  EyeOff,
  RotateCcw,
  Sparkles,
  Layers,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { API_BASE_URL } from "@/lib/api";

export interface RasterLayerToggleProps {
  map: MapLibreMap | null;
  className?: string;
}

export interface RasterPreset {
  id: string;
  name: string;
  badge: string;
  description: string;
  url: string;
}

export const RASTER_PRESETS: RasterPreset[] = [
  {
    id: "sentinel2-tci",
    name: "Sentinel-2 True Color (TCI)",
    badge: "Optical RGB",
    description: "10m natural color composite for ground verification.",
    url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/ER/2026/2/S2B_43PER_20260204_0_L2A/TCI.tif",
  },
  {
    id: "sentinel2-b08",
    name: "NDVI Near-Infrared (Band 8)",
    badge: "Canopy Index",
    description: "High biomass reflectance; sharp drop indicates clearing.",
    url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/ER/2026/2/S2B_43PER_20260204_0_L2A/B08.tif",
  },
  {
    id: "sentinel2-b04",
    name: "Red Spectral Band (Band 4)",
    badge: "Soil Absorption",
    description: "Strong absorption by chlorophyll; exposes bare soil.",
    url: "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/ER/2026/2/S2B_43PER_20260204_0_L2A/B04.tif",
  },
];

export const DEFAULT_COG_URL = RASTER_PRESETS[0].url;

const RASTER_SOURCE_ID = "titiler-raster-source";
const RASTER_LAYER_ID = "titiler-raster-layer";

export default function RasterLayerToggle({ map, className }: RasterLayerToggleProps) {
  const [isEnabled, setIsEnabled] = useState(false);
  const [opacity, setOpacity] = useState(0.75);
  const [cogUrl, setCogUrl] = useState(DEFAULT_COG_URL);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const opacityRef = useRef(0.75);
  const activeCogUrlRef = useRef<string | null>(null);

  // Sync opacityRef
  useEffect(() => {
    opacityRef.current = opacity;
  }, [opacity]);

  // Handle click outside to close flyout panel
  useEffect(() => {
    if (!isPanelOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsPanelOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsPanelOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isPanelOpen]);

  // Clean, robust source recreation & visibility management
  const updateRasterLayer = useCallback(() => {
    if (!map) return;

    if (!isEnabled) {
      if (map.getLayer(RASTER_LAYER_ID)) {
        map.setLayoutProperty(RASTER_LAYER_ID, "visibility", "none");
      }
      return;
    }

    const tileUrl = `${API_BASE_URL}/api/v1/tiles/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=${encodeURIComponent(
      cogUrl.trim()
    )}`;

    const needsSourceRebuild =
      activeCogUrlRef.current !== cogUrl.trim() ||
      !map.getSource(RASTER_SOURCE_ID);

    if (needsSourceRebuild) {
      // 1. Remove existing layer first if present
      if (map.getLayer(RASTER_LAYER_ID)) {
        map.removeLayer(RASTER_LAYER_ID);
      }
      // 2. Remove existing source if present
      if (map.getSource(RASTER_SOURCE_ID)) {
        map.removeSource(RASTER_SOURCE_ID);
      }

      // 3. Add fresh raster tile source
      map.addSource(RASTER_SOURCE_ID, {
        type: "raster",
        tiles: [tileUrl],
        tileSize: 256,
        attribution: "Sentinel-2 MSI / TiTiler Dynamic Stream",
      });

      // 4. Determine ordering: sit beneath vector boundaries and incidents
      const beforeId = map.getLayer("forest-boundaries-fill")
        ? "forest-boundaries-fill"
        : map.getLayer("incidents-fill")
        ? "incidents-fill"
        : undefined;

      map.addLayer(
        {
          id: RASTER_LAYER_ID,
          type: "raster",
          source: RASTER_SOURCE_ID,
          layout: {
            visibility: "visible",
          },
          paint: {
            "raster-opacity": opacityRef.current,
          },
        },
        beforeId
      );

      activeCogUrlRef.current = cogUrl.trim();
    } else {
      // Source already matches; ensure layer is visible
      if (map.getLayer(RASTER_LAYER_ID)) {
        map.setLayoutProperty(RASTER_LAYER_ID, "visibility", "visible");
      }
    }
  }, [map, isEnabled, cogUrl]);

  // Trigger layer sync when map is ready or when isEnabled/cogUrl changes
  useEffect(() => {
    if (!map) return;
    if (map.isStyleLoaded()) {
      updateRasterLayer();
    } else {
      map.once("load", updateRasterLayer);
    }
  }, [map, updateRasterLayer]);

  const handleToggle = () => {
    setIsEnabled((prev) => !prev);
  };

  // High-performance decoupled opacity adjustment
  const handleOpacityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setOpacity(val);
    opacityRef.current = val;
    if (map && map.getLayer(RASTER_LAYER_ID)) {
      map.setPaintProperty(RASTER_LAYER_ID, "raster-opacity", val);
    }
  };

  const handleSelectPreset = (preset: RasterPreset) => {
    setCogUrl(preset.url);
    if (!isEnabled) {
      setIsEnabled(true);
    }
  };

  const handleResetUrl = () => {
    setCogUrl(DEFAULT_COG_URL);
  };

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div className="flex items-center gap-1.5 bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl shadow-xl p-1">
        {/* Toggle Checkbox Button */}
        <button
          type="button"
          onClick={handleToggle}
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
            isEnabled
              ? "bg-emerald-600/30 text-emerald-300 border border-emerald-500/40"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          )}
          title="Toggle TiTiler Satellite Difference Raster Layer"
        >
          {isEnabled ? (
            <Eye className="w-3.5 h-3.5 text-emerald-400" />
          ) : (
            <EyeOff className="w-3.5 h-3.5 text-slate-500" />
          )}
          <Satellite className="w-3.5 h-3.5 text-cyan-400" />
          <span>TiTiler Raster Overlay</span>
          {isEnabled && (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-0.5" />
          )}
        </button>

        {/* Options / Sliders Popover Toggle */}
        <button
          type="button"
          onClick={() => setIsPanelOpen((prev) => !prev)}
          className={cn(
            "p-1.5 rounded-lg transition-colors",
            isPanelOpen
              ? "bg-slate-800 text-white"
              : "text-slate-400 hover:text-white hover:bg-slate-800/80"
          )}
          title="Configure Raster Opacity and COG Source"
        >
          <Sliders className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Flyout Control Panel */}
      {isPanelOpen && (
        <div className="absolute top-full mt-2 right-0 w-80 bg-slate-900/95 backdrop-blur-xl border border-slate-700/90 rounded-2xl p-4 shadow-2xl z-20 space-y-4 text-xs text-slate-200">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 font-semibold text-white">
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span>TiTiler COG Stream</span>
            </span>
            <span className="text-[10px] text-emerald-400 font-mono bg-emerald-950/70 border border-emerald-800/60 px-1.5 py-0.5 rounded">
              XYZ Tile Server
            </span>
          </div>

          {/* Opacity Slider Control */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300">Raster Opacity</span>
              <span className="font-mono text-emerald-400 font-semibold">
                {Math.round(opacity * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={handleOpacityChange}
              disabled={!isEnabled}
              className={cn(
                "w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500",
                !isEnabled && "opacity-40 cursor-not-allowed"
              )}
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>0% (Transparent)</span>
              <span>100% (Solid)</span>
            </div>
          </div>

          {/* Preset Selection Buttons */}
          <div className="space-y-1.5 pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                <Layers className="w-3 h-3 text-cyan-400" />
                <span>Satellite Band Presets</span>
              </span>
            </div>
            <div className="space-y-1.5">
              {RASTER_PRESETS.map((preset) => {
                const isSelected = cogUrl.trim() === preset.url;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => handleSelectPreset(preset)}
                    className={cn(
                      "w-full text-left p-2 rounded-lg border transition-all flex flex-col gap-0.5",
                      isSelected
                        ? "bg-emerald-950/40 border-emerald-500/60 text-white"
                        : "bg-slate-950/60 border-slate-800/80 hover:bg-slate-800/60 text-slate-300"
                    )}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="font-medium text-[11px]">{preset.name}</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                          {preset.badge}
                        </span>
                        {isSelected && <Check className="w-3 h-3 text-emerald-400" />}
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 leading-normal">
                      {preset.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Custom COG Source URL */}
          <div className="space-y-1.5 pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Custom Cloud-Optimized GeoTIFF</span>
              <button
                type="button"
                onClick={handleResetUrl}
                className="text-emerald-400 hover:text-emerald-300 flex items-center gap-1 text-[10px]"
                title="Reset to default Sentinel-2 COG"
              >
                <RotateCcw className="w-2.5 h-2.5" />
                <span>Reset</span>
              </button>
            </div>
            <input
              type="text"
              value={cogUrl}
              onChange={(e) => setCogUrl(e.target.value)}
              placeholder="https://.../cog.tif"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-[10px] font-mono text-slate-300 focus:outline-none focus:border-emerald-500"
            />
          </div>

          <p className="text-[10px] text-slate-400 leading-relaxed">
            Dynamic tiles rendered on-the-fly from the FastAPI backend at{" "}
            <code className="text-emerald-400">/api/v1/tiles</code>.
          </p>
        </div>
      )}
    </div>
  );
}
