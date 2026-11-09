"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import dynamic from "next/dynamic";
import type { Map as MapLibreMap } from "maplibre-gl";
import type { FeatureCollection, Geometry } from "geojson";
import { AlertCircle } from "lucide-react";
import { computeGeometryBbox } from "@/lib/utils";
import TopBar, { TierFilterOption } from "@/components/TopBar";
import RasterLayerToggle from "@/components/RasterLayerToggle";
import IncidentDrawer from "@/components/IncidentDrawer";
import {
  IncidentProperties,
  IncidentSummary,
  fetchIncidentsGeoJSON,
  fetchIncidentSummary,
  DEFAULT_INCIDENTS,
} from "@/lib/api";

// Dynamic import with SSR disabled to prevent WebGL/window errors on server
const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-screen min-h-screen flex flex-col items-center justify-center bg-slate-950 text-slate-400 gap-3">
      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <p className="text-sm font-medium tracking-wide">Initializing Satellite Map Canvas...</p>
    </div>
  ),
});

export default function HomePage() {
  const [selectedIncident, setSelectedIncident] = useState<IncidentProperties | null>(null);
  const [mapInstance, setMapInstance] = useState<MapLibreMap | null>(null);

  // Filter state
  const [selectedTier, setSelectedTier] = useState<TierFilterOption>("ALL");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");

  // Backend summary statistics
  const [summaryData, setSummaryData] = useState<IncidentSummary | null>(null);

  // Raw incidents dataset
  const [allIncidents, setAllIncidents] = useState<FeatureCollection<Geometry, IncidentProperties>>(
    DEFAULT_INCIDENTS
  );

  // Load incidents from API or fallback
  useEffect(() => {
    let isMounted = true;
    fetchIncidentsGeoJSON().then((data) => {
      if (isMounted && data && data.features) {
        setAllIncidents(data);
      }
    });

    fetchIncidentSummary().then((summary) => {
      if (isMounted && summary) {
        setSummaryData(summary);
      }
    });

    return () => {
      isMounted = false;
    };
  }, []);

  const hasActiveFilters =
    selectedTier !== "ALL" || startDate !== "" || endDate !== "";

  // Compute filtered GeoJSON features
  const filteredIncidents = useMemo<FeatureCollection<Geometry, IncidentProperties>>(() => {
    let features = allIncidents.features;

    // Filter by violation tier
    if (selectedTier === "CRITICAL_ONLY") {
      features = features.filter(
        (f) => (f.properties?.violation_tier || "").toUpperCase() === "CRITICAL"
      );
    } else if (selectedTier === "HIGH_PLUS_CRITICAL") {
      features = features.filter((f) =>
        ["CRITICAL", "HIGH"].includes((f.properties?.violation_tier || "").toUpperCase())
      );
    }

    // Filter chronologically by post_date
    if (startDate) {
      features = features.filter((f) => (f.properties?.post_date || "") >= startDate);
    }
    if (endDate) {
      features = features.filter((f) => (f.properties?.post_date || "") <= endDate);
    }

    return {
      type: "FeatureCollection",
      features,
    };
  }, [allIncidents, selectedTier, startDate, endDate]);

  // Compute live KPI metrics from filtered set
  const filteredAcres = useMemo(() => {
    return filteredIncidents.features.reduce(
      (sum, f) => sum + Number(f.properties?.area_acres || 0),
      0
    );
  }, [filteredIncidents]);

  const totalAlertsCount = summaryData?.total_incidents ?? allIncidents.features.length;
  const globalTotalAcres = summaryData?.total_acres ?? allIncidents.features.reduce(
    (sum, f) => sum + Number(f.properties?.area_acres || 0),
    0
  );

  const displayedTotalAcres = hasActiveFilters
    ? filteredAcres
    : summaryData?.total_acres ?? filteredAcres;

  const activeAlertsCount = filteredIncidents.features.length;

  const handleResetFilters = useCallback(() => {
    setSelectedTier("ALL");
    setStartDate("");
    setEndDate("");
  }, []);

  const handleZoomToExtent = useCallback(
    (lat: number, lon: number, incidentId?: string) => {
      if (!mapInstance) return;

      if (incidentId) {
        const feature = allIncidents.features.find(
          (f) => String(f.properties?.id) === String(incidentId)
        );
        if (feature?.geometry) {
          const bbox = computeGeometryBbox(feature.geometry);
          if (bbox) {
            mapInstance.fitBounds(
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
      }

      mapInstance.flyTo({
        center: [lon, lat],
        zoom: 15.5,
        essential: true,
      });
    },
    [mapInstance, allIncidents]
  );

  const handleMapLoad = useCallback((map: MapLibreMap) => {
    setMapInstance(map);
  }, []);

  const handleIncidentSelect = useCallback((incident: IncidentProperties) => {
    setSelectedIncident(incident);
  }, []);

  const handleDrawerClose = useCallback(() => {
    setSelectedIncident(null);
  }, []);

  return (
    <main className="relative w-full h-screen min-h-screen overflow-hidden">
      {/* Top Analytical Navigation Bar with KPI Stats & Filters */}
      <TopBar
        totalAcres={displayedTotalAcres}
        activeAlerts={activeAlertsCount}
        totalAlerts={totalAlertsCount}
        globalTotalAcres={globalTotalAcres}
        monitoredAreaKm2={1255.64}
        selectedTier={selectedTier}
        onTierChange={setSelectedTier}
        startDate={startDate}
        onStartDateChange={setStartDate}
        endDate={endDate}
        onEndDateChange={setEndDate}
        onResetFilters={handleResetFilters}
        rasterToggleSlot={<RasterLayerToggle map={mapInstance} />}
      />

      {/* Empty Filter Results Notification */}
      {filteredIncidents.features.length === 0 && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-20 bg-slate-900/95 backdrop-blur-md border border-slate-700/90 rounded-2xl px-5 py-3 shadow-2xl flex items-center gap-3.5 text-xs text-slate-200 animate-in fade-in slide-in-from-top-2 duration-200">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>No encroachment incidents match the active filters.</span>
          <button
            type="button"
            onClick={handleResetFilters}
            className="px-2.5 py-1 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 border border-emerald-500/40 font-medium transition-colors cursor-pointer"
          >
            Reset Filters
          </button>
        </div>
      )}

      {/* Full-viewport MapLibre GL Canvas */}
      <Map
        className="w-full h-full min-h-screen"
        onMapLoad={handleMapLoad}
        incidentsData={filteredIncidents}
        selectedIncidentId={selectedIncident?.id}
        onIncidentSelect={handleIncidentSelect}
      />

      {/* Coordinates / Status Pill (Bottom Center) */}
      <div className="hidden md:flex absolute bottom-6 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-full px-4 py-1.5 shadow-xl flex items-center gap-3 text-xs text-slate-300 font-mono pointer-events-auto">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span>AOI: Thirthahalli Taluk</span>
          </span>
          <span className="text-slate-700">|</span>
          <span className="text-slate-300">
            13.7000° N, 75.2500° E
          </span>
        </div>
      </div>

      {/* Slide-over Incident Inspection Drawer */}
      <IncidentDrawer
        incident={selectedIncident}
        onClose={handleDrawerClose}
        onZoomToExtent={handleZoomToExtent}
      />
    </main>
  );
}
