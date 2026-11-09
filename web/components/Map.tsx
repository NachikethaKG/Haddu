"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { FeatureCollection, Geometry } from "geojson";
import {
  Layers,
  Satellite,
  Map as MapIcon,
  Globe,
  Navigation,
  TreePine,
  AlertTriangle,
  Eye,
  EyeOff,
} from "lucide-react";
import { cn, computeGeometryBbox } from "@/lib/utils";
import {
  fetchIncidentsGeoJSON,
  fetchForestBoundariesGeoJSON,
  IncidentProperties,
  ForestBoundaryProperties,
  DEFAULT_INCIDENTS,
  DEFAULT_FOREST_BOUNDARIES,
} from "@/lib/api";

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
  incidentsData?: FeatureCollection<Geometry, IncidentProperties> | null;
  forestBoundariesData?: FeatureCollection<Geometry, ForestBoundaryProperties> | null;
  selectedIncidentId?: string | null;
  onIncidentSelect?: (incident: IncidentProperties) => void;
  children?: React.ReactNode;
}

export default function Map({
  onMapLoad,
  className,
  incidentsData,
  forestBoundariesData,
  selectedIncidentId,
  onIncidentSelect,
  children,
}: MapComponentProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const hoveredIncidentIdRef = useRef<string | number | null>(null);

  // Keep callback and data refs updated so the map initialization effect doesn't re-run
  const onMapLoadRef = useRef(onMapLoad);
  useEffect(() => {
    onMapLoadRef.current = onMapLoad;
  }, [onMapLoad]);

  const onIncidentSelectRef = useRef(onIncidentSelect);
  useEffect(() => {
    onIncidentSelectRef.current = onIncidentSelect;
  }, [onIncidentSelect]);

  const [activeBaseLayer, setActiveBaseLayer] = useState<BaseLayerId>("satellite");
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Layer toggles
  const [showIncidents, setShowIncidents] = useState(true);
  const [showForests, setShowForests] = useState(true);

  // Fallback state if parent does not supply data
  const [fallbackIncidents, setFallbackIncidents] = useState<FeatureCollection<Geometry, IncidentProperties>>(
    DEFAULT_INCIDENTS
  );
  const [fallbackForests, setFallbackForests] = useState<FeatureCollection<Geometry, ForestBoundaryProperties>>(
    DEFAULT_FOREST_BOUNDARIES
  );

  const activeIncidents = incidentsData || fallbackIncidents;
  const activeForests = forestBoundariesData || fallbackForests;

  const incidentsRef = useRef(activeIncidents);
  incidentsRef.current = activeIncidents;

  const forestsRef = useRef(activeForests);
  forestsRef.current = activeForests;

  // Load fallback data only once if parent didn't provide props
  useEffect(() => {
    if (!incidentsData) {
      fetchIncidentsGeoJSON().then((data) => setFallbackIncidents(data));
    }
    if (!forestBoundariesData) {
      fetchForestBoundariesGeoJSON().then((data) => setFallbackForests(data));
    }
  }, [incidentsData, forestBoundariesData]);

  // Helper to format incident tooltip HTML
  const formatIncidentTooltip = useCallback((props: IncidentProperties) => {
    const tier = (props.violation_tier || "MODERATE").toUpperCase();
    const badgeColors: Record<string, string> = {
      CRITICAL: "bg-red-500/20 text-red-400 border border-red-500/40",
      HIGH: "bg-orange-500/20 text-orange-400 border border-orange-500/40",
      MODERATE: "bg-amber-500/20 text-amber-400 border border-amber-500/40",
    };
    const badgeClass = badgeColors[tier] || badgeColors.MODERATE;
    const acres = Number(props.area_acres || 0).toFixed(2);
    const guntas = (Number(props.area_acres || 0) * 40).toFixed(1);
    const dateStr = props.post_date || "N/A";
    const forestStr = props.encroached_rf_name
      ? `<div class="text-[11px] text-slate-300 flex items-center justify-between gap-2">
           <span class="text-slate-400">Target Forest:</span>
           <span class="font-medium text-slate-200 text-right truncate max-w-[150px]">${props.encroached_rf_name}</span>
         </div>`
      : "";
    const riparianStr = props.riparian_breach
      ? `<div class="text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/30 font-medium">
           ⚠ Riparian Buffer Breach
         </div>`
      : "";

    return `
      <div class="p-3 font-sans text-xs bg-slate-900/95 text-slate-100 rounded-xl border border-slate-700/80 shadow-2xl backdrop-blur-md min-w-[220px]">
        <div class="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-slate-800">
          <span class="px-2 py-0.5 rounded font-bold uppercase tracking-wider text-[10px] ${badgeClass}">
            ${tier}
          </span>
          <span class="text-[11px] text-slate-400 font-mono">${dateStr}</span>
        </div>
        <div class="space-y-1.5">
          <div class="flex justify-between items-center text-slate-300 text-[11px]">
            <span class="text-slate-400">Disturbed Area:</span>
            <span class="font-bold text-white font-mono">${acres} ac <span class="text-slate-400 font-normal">(${guntas} gt)</span></span>
          </div>
          ${forestStr}
          ${riparianStr}
        </div>
        <div class="mt-2.5 pt-1.5 border-t border-slate-800 text-[10px] text-emerald-400 font-medium flex items-center justify-between">
          <span>Click polygon to inspect</span>
          <span class="text-slate-500 font-mono">${props.sensor || "S2"}</span>
        </div>
      </div>
    `;
  }, []);

  // Helper to format forest boundary tooltip HTML
  const formatForestTooltip = useCallback((props: ForestBoundaryProperties) => {
    const cat = (props.category || "RF").toUpperCase();
    const acres = props.area_acres
      ? `${Number(props.area_acres).toFixed(1)} acres`
      : "Monitored Zone";
    return `
      <div class="p-2.5 font-sans text-xs bg-slate-900/95 text-slate-100 rounded-xl border border-emerald-600/50 shadow-2xl backdrop-blur-md min-w-[190px]">
        <div class="flex items-center justify-between gap-2 mb-1.5 pb-1 border-b border-slate-800">
          <span class="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
            ${cat === "RF" ? "Reserve Forest" : cat}
          </span>
          <span class="text-[10px] text-slate-400 font-mono">${acres}</span>
        </div>
        <div class="font-semibold text-emerald-300 text-[11px]">${props.name || "Forest Boundary"}</div>
      </div>
    `;
  }, []);

  // Initialize Map ONCE on mount
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Build base raster sources and layers
    const sources: Record<string, maplibregl.SourceSpecification> = {};
    const layers: maplibregl.LayerSpecification[] = [
      {
        id: "haddu-background",
        type: "background",
        paint: {
          "background-color": "#090d16",
        },
      },
    ];

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

    mapInstanceRef.current = map;

    // Navigation and Scale controls
    map.addControl(
      new maplibregl.NavigationControl({
        showCompass: true,
        showZoom: true,
        visualizePitch: true,
      }),
      "top-right"
    );

    map.addControl(
      new maplibregl.ScaleControl({
        maxWidth: 150,
        unit: "metric",
      }),
      "bottom-right"
    );

    // Reusable popup instance
    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 12,
      className: "haddu-map-popup",
    });
    popupRef.current = popup;

    map.on("load", () => {
      // Force initial size calculation
      map.resize();
      if (typeof window !== "undefined") {
        requestAnimationFrame(() => map.resize());
        setTimeout(() => map.resize(), 150);
      }

      // 1. Add Reserve Forest Boundaries Source & Layers
      map.addSource("forest-boundaries-source", {
        type: "geojson",
        data: forestsRef.current as unknown as GeoJSON.GeoJSON,
        generateId: true,
      });

      // Semi-transparent green fill
      map.addLayer({
        id: "forest-boundaries-fill",
        type: "fill",
        source: "forest-boundaries-source",
        paint: {
          "fill-color": "#10B981",
          "fill-opacity": 0.08,
        },
      });

      // Green reference outline
      map.addLayer({
        id: "forest-boundaries-line",
        type: "line",
        source: "forest-boundaries-source",
        paint: {
          "line-color": "#059669",
          "line-width": 1.5,
          "line-dasharray": [4, 2],
        },
      });

      // 2. Add Detected Incidents Source & Layers
      map.addSource("incidents-source", {
        type: "geojson",
        data: incidentsRef.current as unknown as GeoJSON.GeoJSON,
        generateId: true,
      });

      // Disturbance polygon fill with dynamic tier color styling
      map.addLayer({
        id: "incidents-fill",
        type: "fill",
        source: "incidents-source",
        paint: {
          "fill-color": [
            "match",
            ["upcase", ["coalesce", ["get", "violation_tier"], "MODERATE"]],
            "CRITICAL",
            "#EF4444", // Red
            "HIGH",
            "#F97316", // Orange
            "MODERATE",
            "#EAB308", // Yellow
            "#10B981", // Fallback
          ],
          "fill-opacity": [
            "case",
            ["boolean", ["feature-state", "hover"], false],
            0.85,
            0.55,
          ],
        },
      });

      // Disturbance polygon outline
      map.addLayer({
        id: "incidents-line",
        type: "line",
        source: "incidents-source",
        paint: {
          "line-color": [
            "match",
            ["upcase", ["coalesce", ["get", "violation_tier"], "MODERATE"]],
            "CRITICAL",
            "#B91C1C",
            "HIGH",
            "#C2410C",
            "MODERATE",
            "#A16207",
            "#047857",
          ],
          "line-width": [
            "case",
            ["boolean", ["feature-state", "hover"], false],
            3.5,
            1.8,
          ],
        },
      });

      // Mouse interactions on incidents
      map.on("mousemove", "incidents-fill", (e) => {
        if (!e.features || e.features.length === 0) return;
        map.getCanvas().style.cursor = "pointer";

        const feature = e.features[0];
        const featureId = feature.id;

        if (hoveredIncidentIdRef.current !== null && hoveredIncidentIdRef.current !== featureId) {
          map.setFeatureState(
            { source: "incidents-source", id: hoveredIncidentIdRef.current },
            { hover: false }
          );
        }

        if (featureId !== undefined) {
          hoveredIncidentIdRef.current = featureId;
          map.setFeatureState(
            { source: "incidents-source", id: featureId },
            { hover: true }
          );
        }

        const props = feature.properties as unknown as IncidentProperties;
        popup.setLngLat(e.lngLat).setHTML(formatIncidentTooltip(props)).addTo(map);
      });

      map.on("mouseleave", "incidents-fill", () => {
        map.getCanvas().style.cursor = "";
        popup.remove();

        if (hoveredIncidentIdRef.current !== null) {
          map.setFeatureState(
            { source: "incidents-source", id: hoveredIncidentIdRef.current },
            { hover: false }
          );
          hoveredIncidentIdRef.current = null;
        }
      });

      // Click event for incident inspection
      map.on("click", "incidents-fill", (e) => {
        if (!e.features || e.features.length === 0) return;
        const props = e.features[0].properties as unknown as IncidentProperties;
        if (onIncidentSelectRef.current) {
          onIncidentSelectRef.current(props);
        }
      });

      // Hover on forest boundaries (when not hovering incidents)
      map.on("mousemove", "forest-boundaries-fill", (e) => {
        if (!e.features || e.features.length === 0) return;
        if (hoveredIncidentIdRef.current !== null) return;

        const props = e.features[0].properties as unknown as ForestBoundaryProperties;
        popup.setLngLat(e.lngLat).setHTML(formatForestTooltip(props)).addTo(map);
      });

      map.on("mouseleave", "forest-boundaries-fill", () => {
        if (hoveredIncidentIdRef.current === null) {
          popup.remove();
        }
      });

      setMapLoaded(true);
      if (onMapLoadRef.current) {
        onMapLoadRef.current(map);
      }
    });

    // ResizeObserver to ensure canvas size is always pixel-perfect and prevent black/gray gaps
    const container = mapContainerRef.current;
    let resizeObserver: ResizeObserver | null = null;
    if (container && typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        if (mapInstanceRef.current) {
          mapInstanceRef.current.resize();
        }
      });
      resizeObserver.observe(container);
    }

    return () => {
      resizeObserver?.disconnect();
      popup.remove();
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [formatIncidentTooltip, formatForestTooltip]);

  // Update incidents data in place without re-initializing map
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;
    const source = map.getSource("incidents-source") as maplibregl.GeoJSONSource | undefined;
    if (source) {
      source.setData(activeIncidents as unknown as GeoJSON.GeoJSON);
    }
  }, [activeIncidents, mapLoaded]);

  // Update forest boundaries data in place without re-initializing map
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;
    const source = map.getSource("forest-boundaries-source") as maplibregl.GeoJSONSource | undefined;
    if (source) {
      source.setData(activeForests as unknown as GeoJSON.GeoJSON);
    }
  }, [activeForests, mapLoaded]);

  // Fly to selected incident if specified
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded || !selectedIncidentId) return;

    const feature = activeIncidents.features.find(
      (f) => String(f.properties?.id) === String(selectedIncidentId)
    );
    if (feature?.geometry) {
      const bbox = computeGeometryBbox(feature.geometry);
      if (bbox) {
        map.fitBounds(
          [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]],
          ],
          {
            padding: { top: 90, bottom: 80, left: 80, right: 460 },
            maxZoom: 16.5,
            duration: 900,
          }
        );
        return;
      }
    }
    if (feature && feature.properties?.centroid_lon && feature.properties?.centroid_lat) {
      map.flyTo({
        center: [Number(feature.properties.centroid_lon), Number(feature.properties.centroid_lat)],
        zoom: 15,
        essential: true,
      });
    }
  }, [selectedIncidentId, mapLoaded, activeIncidents]);

  // Toggle incidents layer visibility
  const toggleIncidentsVisibility = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const nextState = !showIncidents;
    setShowIncidents(nextState);
    const visibility = nextState ? "visible" : "none";
    if (map.getLayer("incidents-fill")) map.setLayoutProperty("incidents-fill", "visibility", visibility);
    if (map.getLayer("incidents-line")) map.setLayoutProperty("incidents-line", "visibility", visibility);
  };

  // Toggle forest boundaries layer visibility
  const toggleForestsVisibility = () => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const nextState = !showForests;
    setShowForests(nextState);
    const visibility = nextState ? "visible" : "none";
    if (map.getLayer("forest-boundaries-fill")) {
      map.setLayoutProperty("forest-boundaries-fill", "visibility", visibility);
    }
    if (map.getLayer("forest-boundaries-line")) {
      map.setLayoutProperty("forest-boundaries-line", "visibility", visibility);
    }
  };

  // Switch base layer
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
    <div className={cn("relative w-full h-full min-h-screen overflow-hidden", className)}>
      <div
        ref={mapContainerRef}
        className="absolute inset-0 w-full h-full"
        style={{ width: "100%", height: "100%" }}
      />

      {/* Floating Map Controls & Legends (Bottom Left) */}
      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-2 max-w-xs">
        {/* Violation Tier Legend & Layer Toggles */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl shadow-2xl p-3 text-xs text-slate-200">
          <div className="flex items-center justify-between font-semibold text-white mb-2 pb-1.5 border-b border-slate-800">
            <span className="flex items-center gap-1.5 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>Legal Violation Tiers</span>
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              {activeIncidents.features.length} alerts
            </span>
          </div>

          <div className="space-y-1.5 mb-3">
            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
                <span className="text-slate-300 font-medium">Critical</span>
              </div>
              <span className="text-[10px] text-slate-400">Reserve Forest Breach</span>
            </div>

            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shadow-sm shadow-orange-500/50" />
                <span className="text-slate-300 font-medium">High</span>
              </div>
              <span className="text-[10px] text-slate-400">Riparian Corridor (50m)</span>
            </div>

            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-yellow-500 shadow-sm shadow-yellow-500/50" />
                <span className="text-slate-300 font-medium">Moderate</span>
              </div>
              <span className="text-[10px] text-slate-400">General Disturbance</span>
            </div>

            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-0.5 bg-emerald-500 border border-emerald-400" />
                <span className="text-slate-300 font-medium">Forest Bounds</span>
              </div>
              <span className="text-[10px] text-emerald-400">Gazetted Perimeter</span>
            </div>
          </div>

          {/* Layer Visibility Toggles */}
          <div className="pt-2 border-t border-slate-800 flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={toggleIncidentsVisibility}
              className={cn(
                "flex-1 flex items-center justify-center gap-1.5 py-1 px-2 rounded-lg text-[10px] font-medium transition-colors border",
                showIncidents
                  ? "bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-750"
                  : "bg-slate-900/50 border-slate-800 text-slate-500 line-through"
              )}
              title="Toggle Incidents Layer Visibility"
            >
              {showIncidents ? <Eye className="w-3 h-3 text-red-400" /> : <EyeOff className="w-3 h-3" />}
              <span>Clearings</span>
            </button>

            <button
              type="button"
              onClick={toggleForestsVisibility}
              className={cn(
                "flex-1 flex items-center justify-center gap-1.5 py-1 px-2 rounded-lg text-[10px] font-medium transition-colors border",
                showForests
                  ? "bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-750"
                  : "bg-slate-900/50 border-slate-800 text-slate-500 line-through"
              )}
              title="Toggle Forest Boundaries Visibility"
            >
              {showForests ? <TreePine className="w-3 h-3 text-emerald-400" /> : <EyeOff className="w-3 h-3" />}
              <span>Forests</span>
            </button>
          </div>
        </div>

        {/* Base Layer Switcher Button */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl shadow-2xl p-1.5 flex flex-col gap-1 transition-all">
          <button
            type="button"
            onClick={() => setIsSwitcherOpen((prev) => !prev)}
            className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-slate-200 hover:text-white rounded-lg hover:bg-slate-800/80 transition-colors"
            title="Switch Map Base Layer"
          >
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>
              Base:{" "}
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
          className="self-start bg-slate-900/90 backdrop-blur-md border border-slate-700/80 hover:border-emerald-500/50 rounded-xl shadow-xl px-3 py-2 text-xs font-medium text-slate-200 hover:text-white flex items-center gap-2 transition-all hover:bg-slate-800/80"
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
