import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Geometry } from "geojson";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Compute the bounding box [minLon, minLat, maxLon, maxLat] from a GeoJSON geometry.
 */
export function computeGeometryBbox(
  geometry: Geometry | null | undefined
): [number, number, number, number] | null {
  if (!geometry) return null;

  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;

  function traverse(coords: unknown) {
    if (!coords || !Array.isArray(coords)) return;
    if (typeof coords[0] === "number" && typeof coords[1] === "number") {
      const lon = coords[0];
      const lat = coords[1];
      if (lon < minLon) minLon = lon;
      if (lat < minLat) minLat = lat;
      if (lon > maxLon) maxLon = lon;
      if (lat > maxLat) maxLat = lat;
    } else {
      for (const item of coords) {
        traverse(item);
      }
    }
  }

  // Polygon or MultiPolygon coordinates
  if ("coordinates" in geometry) {
    traverse((geometry as { coordinates: unknown }).coordinates);
  }

  if (minLon === Infinity || minLat === Infinity) {
    return null;
  }

  return [minLon, minLat, maxLon, maxLat];
}
