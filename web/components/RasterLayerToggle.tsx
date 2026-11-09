"use client";

import React, { useEffect, useState, useCallback } from "react";
import type { Map as MapLibreMap } from "maplibre-gl";
import {
  Satellite,
  Sliders,
  Eye,
  EyeOff,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { API_BASE_URL } from "@/lib/api";

export interface RasterLayerToggleProps {
  map: MapLibreMap | null;
  className?: string;
}

export const DEFAULT_COG_URL =
  "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/ER/2026/2/S2B_43PER_20260204_0_L2A/TCI.tif";

export default function RasterLayerToggle({ map, className }: RasterLayerToggleProps) {
  const [isEnabled, setIsEnabled] = useState(false);
  const [opacity, setOpacity] = useState(0.75);
  const [cogUrl, setCogUrl] = useState(DEFAULT_COG_URL);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const updateRasterLayer = useCallback(() => {
    if (!map) return;

    const sourceId = "titiler-raster-source";
    const layerId = "titiler-raster-layer";

    if (!isEnabled) {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", "none");
      }
      return;
    }

    const tileUrl = `${API_BASE_URL}/api/v1/tiles/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=${encodeURIComponent(
      cogUrl
    )}`;

    if (!map.getSource(sourceId)) {
      map.addSource(sourceId, {
        type: "raster",
        tiles: [tileUrl],
        tileSize: 256,
        attribution: "Sentinel-2 MSI / TiTiler Dynamic Stream",
      });
    }

    if (!map.getLayer(layerId)) {
      // Place raster layer underneath vector boundaries and incidents
      const beforeId = map.getLayer("forest-boundaries-fill")
        ? "forest-boundaries-fill"
        : map.getLayer("incidents-fill")
        ? "incidents-fill"
        : undefined;

      map.addLayer(
        {
          id: layerId,
          type: "raster",
          source: sourceId,
          layout: {
            visibility: "visible",
          },
          paint: {
            "raster-opacity": opacity,
          },
        },
        beforeId
      );
    } else {
      map.setLayoutProperty(layerId, "visibility", "visible");
      map.setPaintProperty(layerId, "raster-opacity", opacity);
    }
  }, [map, isEnabled, opacity, cogUrl]);

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

  const handleOpacityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setOpacity(val);
    if (map && map.getLayer("titiler-raster-layer")) {
      map.setPaintProperty("titiler-raster-layer", "raster-opacity", val);
    }
  };

  const handleResetUrl = () => {
    setCogUrl(DEFAULT_COG_URL);
  };

  return (
    <div className={cn("relative", className)}>
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
        <div className="absolute top-full mt-2 right-0 w-72 bg-slate-900/95 backdrop-blur-xl border border-slate-700/90 rounded-2xl p-4 shadow-2xl z-20 space-y-4 text-xs text-slate-200">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800 font-semibold text-white">
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span>TiTiler COG Stream</span>
            </span>
            <span className="text-[10px] text-emerald-400 font-mono">
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

          {/* COG Source URL */}
          <div className="space-y-1.5 pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Cloud-Optimized GeoTIFF</span>
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
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-[11px] font-mono text-slate-300 focus:outline-none focus:border-emerald-500"
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
