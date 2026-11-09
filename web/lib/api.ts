/**
 * Haddu API client service for fetching geospatial vectors, incidents, and AOI boundaries.
 */

import type { FeatureCollection, Feature, Geometry } from "geojson";

export type ViolationTier = "CRITICAL" | "HIGH" | "MODERATE";

export interface IncidentProperties {
  id: string;
  pre_date: string;
  post_date: string;
  area_sqm: number;
  area_acres: number;
  centroid_lat: number;
  centroid_lon: number;
  violation_tier: ViolationTier | string;
  encroached_rf_name: string | null;
  rf_area_acres: number | null;
  riparian_breach: boolean;
  sensor: string;
  mean_delta_ndvi: number | null;
  confidence_score: number | null;
  properties?: Record<string, unknown>;
}

export type IncidentFeature = Feature<Geometry, IncidentProperties>;

export interface ForestBoundaryProperties {
  id: string;
  name: string;
  category: string;
  area_acres?: number;
  area_sqm?: number;
}

export type ForestBoundaryFeature = Feature<Geometry, ForestBoundaryProperties>;

export type GeoJSONFeatureCollection<P = Record<string, unknown>> = FeatureCollection<Geometry, P>;

export interface IncidentSummary {
  total_incidents: number;
  total_acres: number;
  critical_count: number;
  critical_acres: number;
  high_count: number;
  high_acres: number;
  moderate_count: number;
  moderate_acres: number;
  riparian_breaches: number;
}

export interface AOIItem {
  id: string;
  name: string;
  category: string;
  created_at: string;
}

export interface IncidentFilterParams {
  violation_tier?: string;
  min_acres?: number;
  max_acres?: number;
  start_date?: string;
  end_date?: string;
  sensor?: string;
  limit?: number;
  offset?: number;
}

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Fallback static reserve forest boundaries for Thirthahalli Taluk
 * (matches data/boundaries/forest_boundaries.geojson).
 */
export const DEFAULT_FOREST_BOUNDARIES: FeatureCollection<Geometry, ForestBoundaryProperties> = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "rf-agumbe",
      properties: {
        id: "rf-agumbe",
        name: "Agumbe Rainforest Reserve Forest",
        category: "RF",
        area_acres: 1420.5,
      },
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [75.06, 13.55],
              [75.14, 13.55],
              [75.14, 13.62],
              [75.06, 13.62],
              [75.06, 13.55],
            ],
          ],
        ],
      },
    },
    {
      type: "Feature",
      id: "rf-national-park",
      properties: {
        id: "rf-national-park",
        name: "Agumbe National Park Eco-Zone",
        category: "NationalPark",
        area_acres: 850.0,
      },
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [75.05, 13.55],
              [75.08, 13.55],
              [75.08, 13.60],
              [75.05, 13.60],
              [75.05, 13.55],
            ],
          ],
        ],
      },
    },
    {
      type: "Feature",
      id: "rf-kundadri",
      properties: {
        id: "rf-kundadri",
        name: "Kundadri Peak Reserve Forest",
        category: "RF",
        area_acres: 1890.2,
      },
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [75.15, 13.56],
              [75.22, 13.56],
              [75.22, 13.63],
              [75.15, 13.63],
              [75.15, 13.56],
            ],
          ],
        ],
      },
    },
    {
      type: "Feature",
      id: "rf-megaravalli",
      properties: {
        id: "rf-megaravalli",
        name: "Megaravalli State Forest",
        category: "RF",
        area_acres: 2100.8,
      },
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [75.10, 13.64],
              [75.18, 13.64],
              [75.18, 13.72],
              [75.10, 13.72],
              [75.10, 13.64],
            ],
          ],
        ],
      },
    },
    {
      type: "Feature",
      id: "rf-gomala",
      properties: {
        id: "rf-gomala",
        name: "Thirthahalli Gomala Pasture Land",
        category: "Gomala",
        area_acres: 740.0,
      },
      geometry: {
        type: "MultiPolygon",
        coordinates: [
          [
            [
              [75.22, 13.67],
              [75.28, 13.67],
              [75.28, 13.72],
              [75.22, 13.72],
              [75.22, 13.67],
            ],
          ],
        ],
      },
    },
  ],
};

/**
 * Fallback incidents for Thirthahalli Taluk if backend is offline or has no data.
 */
export const DEFAULT_INCIDENTS: FeatureCollection<Geometry, IncidentProperties> = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: "inc-001-crit",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.185, 13.590],
            [75.195, 13.590],
            [75.198, 13.598],
            [75.188, 13.602],
            [75.182, 13.595],
            [75.185, 13.590],
          ],
        ],
      },
      properties: {
        id: "c7e2b109-8432-4d1a-8289-498b31a28101",
        pre_date: "2026-01-10",
        post_date: "2026-02-15",
        area_sqm: 16997.0,
        area_acres: 4.20,
        centroid_lat: 13.595,
        centroid_lon: 75.189,
        violation_tier: "CRITICAL",
        encroached_rf_name: "Kundadri Peak Reserve Forest",
        rf_area_acres: 4.20,
        riparian_breach: false,
        sensor: "Sentinel-2 MSI",
        mean_delta_ndvi: -0.38,
        confidence_score: 0.94,
      },
    },
    {
      type: "Feature",
      id: "inc-002-high",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.242, 13.665],
            [75.250, 13.667],
            [75.252, 13.673],
            [75.244, 13.671],
            [75.242, 13.665],
          ],
        ],
      },
      properties: {
        id: "d9e3f208-9543-4e2b-939a-509c42b39202",
        pre_date: "2026-01-20",
        post_date: "2026-02-18",
        area_sqm: 8498.5,
        area_acres: 2.10,
        centroid_lat: 13.669,
        centroid_lon: 75.247,
        violation_tier: "HIGH",
        encroached_rf_name: "Tunga River Riparian Corridor",
        rf_area_acres: null,
        riparian_breach: true,
        sensor: "Sentinel-1 SAR",
        mean_delta_ndvi: -0.29,
        confidence_score: 0.88,
      },
    },
    {
      type: "Feature",
      id: "inc-003-mod",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [75.230, 13.690],
            [75.236, 13.691],
            [75.237, 13.696],
            [75.231, 13.695],
            [75.230, 13.690],
          ],
        ],
      },
      properties: {
        id: "e0f4a319-0654-5f3c-040b-610d53c40303",
        pre_date: "2026-02-01",
        post_date: "2026-02-22",
        area_sqm: 5260.8,
        area_acres: 1.30,
        centroid_lat: 13.693,
        centroid_lon: 75.233,
        violation_tier: "MODERATE",
        encroached_rf_name: "Thirthahalli Gomala Pasture Land",
        rf_area_acres: 1.30,
        riparian_breach: false,
        sensor: "Sentinel-2 MSI",
        mean_delta_ndvi: -0.22,
        confidence_score: 0.79,
      },
    },
  ],
};

/**
 * Fetch detected encroachment incidents formatted as GeoJSON.
 */
export async function fetchIncidentsGeoJSON(
  filters?: IncidentFilterParams
): Promise<FeatureCollection<Geometry, IncidentProperties>> {
  const query = new URLSearchParams();
  if (filters?.violation_tier && filters.violation_tier !== "ALL") {
    query.set("violation_tier", filters.violation_tier);
  }
  if (filters?.min_acres !== undefined) query.set("min_acres", filters.min_acres.toString());
  if (filters?.max_acres !== undefined) query.set("max_acres", filters.max_acres.toString());
  if (filters?.start_date) query.set("start_date", filters.start_date);
  if (filters?.end_date) query.set("end_date", filters.end_date);
  if (filters?.sensor) query.set("sensor", filters.sensor);
  if (filters?.limit) query.set("limit", filters.limit.toString());

  const qs = query.toString();
  const url = `${API_BASE_URL}/api/v1/incidents/geojson${qs ? `?${qs}` : ""}`;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    const data: FeatureCollection<Geometry, IncidentProperties> = await res.json();
    if (data.features && data.features.length > 0) {
      return data;
    }
    return DEFAULT_INCIDENTS;
  } catch (err) {
    console.warn("Could not fetch live incidents from API, using demo dataset:", err);
    return DEFAULT_INCIDENTS;
  }
}

/**
 * Fetch registered Areas of Interest (Taluks / Ranges).
 */
export async function fetchAOIs(): Promise<AOIItem[]> {
  const url = `${API_BASE_URL}/api/v1/aoi`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch AOIs from API:", err);
    return [{ id: "aoi-thirthahalli", name: "Thirthahalli Taluk", category: "Taluk", created_at: "2026-01-01T00:00:00Z" }];
  }
}

/**
 * Fetch registered Reserve Forest boundaries as GeoJSON.
 */
export async function fetchForestBoundariesGeoJSON(
  category?: string
): Promise<FeatureCollection<Geometry, ForestBoundaryProperties>> {
  const qs = category ? `?category=${encodeURIComponent(category)}` : "";
  const url = `${API_BASE_URL}/api/v1/aoi/forest-boundaries/geojson${qs}`;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    const data: FeatureCollection<Geometry, ForestBoundaryProperties> = await res.json();
    if (data.features && data.features.length > 0) {
      return data;
    }
    return DEFAULT_FOREST_BOUNDARIES;
  } catch (err) {
    console.warn("Could not fetch live forest boundaries from API, using default boundaries:", err);
    return DEFAULT_FOREST_BOUNDARIES;
  }
}

/**
 * Fetch aggregate violation summary metrics.
 */
export async function fetchIncidentSummary(params?: {
  start_date?: string;
  end_date?: string;
  sensor?: string;
}): Promise<IncidentSummary> {
  const query = new URLSearchParams();
  if (params?.start_date) query.set("start_date", params.start_date);
  if (params?.end_date) query.set("end_date", params.end_date);
  if (params?.sensor) query.set("sensor", params.sensor);

  const qs = query.toString();
  const url = `${API_BASE_URL}/api/v1/incidents/summary${qs ? `?${qs}` : ""}`;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch summary from API, using default metrics:", err);
    return {
      total_incidents: 3,
      total_acres: 7.60,
      critical_count: 1,
      critical_acres: 4.20,
      high_count: 1,
      high_acres: 2.10,
      moderate_count: 1,
      moderate_acres: 1.30,
      riparian_breaches: 1,
    };
  }
}

/**
 * Return download URL for formal legal PDF dossier.
 */
export function getDossierDownloadUrl(incidentId: string): string {
  return `${API_BASE_URL}/api/v1/incidents/${incidentId}/dossier`;
}
