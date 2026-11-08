"use client";

import React, { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { Layers, Satellite, Map as MapIcon, Globe, Navigation } from "lucide-react";
import { cn } from "@/lib/utils";

export type BaseLayerId = "satellite" | "carto" | "osm";

interface BaseLayerConfig {
  id: BaseLayerId;
  name: string;
  sourceId: string;
  layerId: string;
  tiles: string[];
  attribution: string;
  maxzoom: number;
}

export const BASE_LAYERS: Record<BaseLayerId, BaseLayerConfig> = {
  satellite: {
    id: "satellite",
    name: "Esri Satellite",
    sourceId: "source-esri-satellite",
    layerId: "layer-esri-satellite",
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
    maxzoom: 19,
  },
  carto: {
    id: "carto",
    name: "Carto Positron",
    sourceId: "source-carto-positron",
    layerId: "layer-carto-positron",
    tiles: [
      "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
      "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    ],
    attribution: "© OpenStreetMap contributors, © CARTO",
    maxzoom: 19,
  },
  osm: {
    id: "osm",
    name: "OpenStreetMap",
    sourceId: "source-osm-standard",
    layerId: "layer-osm-standard",
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: "© OpenStreetMap contributors",
    maxzoom: 19,
  },
};

export const THIRTHAHALLI_CENTER: [number, number] = [75.25, 13.70];
export const DEFAULT_ZOOM = 11;

export interface MapComponentProps {
  onMapLoad?: (map: maplibregl.Map) => void;
  className?: string;
  children?: React.ReactNode;
}

export default function Map({ onMapLoad, className, children }: MapComponentProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<maplibregl.Map | null>(null);
  const [activeBaseLayer, setActiveBaseLayer] = useState<BaseLayerId>("satellite");
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Build the composite style containing all base layers
    const sources: Record<string, maplibregl.SourceSpecification> = {};
    const layers: maplibregl.LayerSpecification[] = [];

    (Object.keys(BASE_LAYERS) as BaseLayerId[]).forEach((key) => {
      const cfg = BASE_LAYERS[key];
      sources[cfg.sourceId] = {
        type: "raster",
        tiles: cfg.tiles,
        tileSize: 256,
        attribution: cfg.attribution,
        maxzoom: cfg.maxzoom,
      };

      layers.push({
        id: cfg.layerId,
        type: "raster",
        source: cfg.sourceId,
        layout: {
          visibility: key === "satellite" ? "visible" : "none",
        },
        minzoom: 0,
        maxzoom: 22,
      });
    });

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: {
        version: 8,
        sources,
        layers,
      },
      center: THIRTHAHALLI_CENTER,
      zoom: DEFAULT_ZOOM,
      minZoom: 4,
      maxZoom: 20,
      pitch: 0,
      bearing: 0,
    });

    // Add navigation controls (zoom, pitch, compass)
    map.addControl(
      new maplibregl.NavigationControl({
        showCompass: true,
        showZoom: true,
        visualizePitch: true,
      }),
      "top-right"
    );

    // Scale bar
    map.addControl(
      new maplibregl.ScaleControl({
        maxWidth: 150,
        unit: "metric",
      }),
      "bottom-right"
    );

    map.on("load", () => {
      mapInstanceRef.current = map;
      setMapLoaded(true);
      if (onMapLoad) {
        onMapLoad(map);
      }
    });

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [onMapLoad]);

  const switchBaseLayer = (layerId: BaseLayerId) => {
    const map = mapInstanceRef.current;
    if (!map) return;

    (Object.keys(BASE_LAYERS) as BaseLayerId[]).forEach((key) => {
      const cfg = BASE_LAYERS[key];
      if (map.getLayer(cfg.layerId)) {
        map.setLayoutProperty(
          cfg.layerId,
          "visibility",
          key === layerId ? "visible" : "none"
        );
      }
    });

    setActiveBaseLayer(layerId);
  };

  const resetToCenter = () => {
    if (!mapInstanceRef.current) return;
    mapInstanceRef.current.flyTo({
      center: THIRTHAHALLI_CENTER,
      zoom: DEFAULT_ZOOM,
      pitch: 0,
      bearing: 0,
      essential: true,
    });
  };

  return (
    <div className={cn("relative w-full h-full overflow-hidden", className)}>
      <div ref={mapContainerRef} className="absolute inset-0 w-full h-full" />

      {/* Base Layer Switcher & Re-center Controls */}
      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-2">
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/70 rounded-xl shadow-2xl p-1.5 flex flex-col gap-1 transition-all">
          <button
            type="button"
            onClick={() => setIsSwitcherOpen((prev) => !prev)}
            className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-slate-200 hover:text-white rounded-lg hover:bg-slate-800/80 transition-colors"
            title="Switch Map Base Layer"
          >
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>
              Base Layer:{" "}
              <strong className="text-white font-semibold">
                {BASE_LAYERS[activeBaseLayer].name}
              </strong>
            </span>
          </button>

          {isSwitcherOpen && (
            <div className="flex flex-col gap-1 pt-1 border-t border-slate-800">
              <button
                type="button"
                onClick={() => {
                  switchBaseLayer("satellite");
                  setIsSwitcherOpen(false);
                }}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg transition-colors text-left",
                  activeBaseLayer === "satellite"
                    ? "bg-emerald-600/30 text-emerald-300 font-medium border border-emerald-500/40"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                )}
              >
                <Satellite className="w-3.5 h-3.5 text-cyan-400" />
                <span>Esri World Imagery (Satellite)</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  switchBaseLayer("carto");
                  setIsSwitcherOpen(false);
                }}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg transition-colors text-left",
                  activeBaseLayer === "carto"
                    ? "bg-emerald-600/30 text-emerald-300 font-medium border border-emerald-500/40"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                )}
              >
                <MapIcon className="w-3.5 h-3.5 text-amber-400" />
                <span>Carto Positron (Light Vector)</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  switchBaseLayer("osm");
                  setIsSwitcherOpen(false);
                }}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg transition-colors text-left",
                  activeBaseLayer === "osm"
                    ? "bg-emerald-600/30 text-emerald-300 font-medium border border-emerald-500/40"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                )}
              >
                <Globe className="w-3.5 h-3.5 text-blue-400" />
                <span>OpenStreetMap (Standard)</span>
              </button>
            </div>
          )}
        </div>

        {/* Quick Reset to Thirthahalli */}
        <button
          type="button"
          onClick={resetToCenter}
          className="self-start bg-slate-900/90 backdrop-blur-md border border-slate-700/70 hover:border-emerald-500/50 rounded-xl shadow-xl px-3 py-2 text-xs font-medium text-slate-200 hover:text-white flex items-center gap-2 transition-all hover:bg-slate-800/80"
          title="Reset View to Thirthahalli Forest Center"
        >
          <Navigation className="w-3.5 h-3.5 text-emerald-400" />
          <span>Thirthahalli Extent</span>
        </button>
      </div>

      {/* Render optional child overlays passed from parent page */}
      {mapLoaded && children}
    </div>
  );
}
