"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import dynamic from "next/dynamic";
import type { Map as MapLibreMap } from "maplibre-gl";
import type { FeatureCollection, Geometry } from "geojson";
import TopBar, { TierFilterOption } from "@/components/TopBar";
import RasterLayerToggle from "@/components/RasterLayerToggle";
import IncidentDrawer from "@/components/IncidentDrawer";
import {
  IncidentProperties,
  fetchIncidentsGeoJSON,
  DEFAULT_INCIDENTS,
} from "@/lib/api";

// Dynamic import with SSR disabled to prevent WebGL/window errors on server
const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-slate-400 gap-3">
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
    return () => {
      isMounted = false;
    };
  }, []);

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
  const totalClearedAcres = useMemo(() => {
    return filteredIncidents.features.reduce(
      (sum, f) => sum + Number(f.properties?.area_acres || 0),
      0
    );
  }, [filteredIncidents]);

  const activeAlertsCount = filteredIncidents.features.length;

  const handleResetFilters = useCallback(() => {
    setSelectedTier("ALL");
    setStartDate("");
    setEndDate("");
  }, []);

  const handleZoomToExtent = useCallback(
    (lat: number, lon: number) => {
      if (mapInstance) {
        mapInstance.flyTo({
          center: [lon, lat],
          zoom: 15.5,
          essential: true,
        });
      }
    },
    [mapInstance]
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
    <main className="relative w-full h-full overflow-hidden">
      {/* Top Analytical Navigation Bar with KPI Stats & Filters */}
      <TopBar
        totalAcres={totalClearedAcres}
        activeAlerts={activeAlertsCount}
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

      {/* Full-viewport MapLibre GL Canvas */}
      <Map
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
