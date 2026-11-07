"""Legal violation classifier and spatial intersection engine for Haddu.

Cross-references detected forest clearing polygons against gazetted Reserve Forest
boundaries and statutory 50m riparian watercourse buffers in PostGIS to categorize
violations into legal enforcement risk tiers:

  - CRITICAL : Any spatial encroachment inside a gazetted Reserve Forest (RF) or National Park.
  - HIGH     : Any breach of the statutory 50m riparian buffer along watercourses, OR
               within the 200m Eco-Sensitive Zone (ESZ) buffer of a Reserve Forest boundary.
  - MODERATE : Clearing occurring on private or revenue agricultural lands (> 200m from RF).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyproj
import shapely
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from haddu.core.boundaries import (
    _load_forest_features,
    _load_water_features,
    compute_riparian_buffer,
)
from haddu.db.models import ForestBoundary, WaterCourse

logger = logging.getLogger(__name__)

DEFAULT_ESZ_DISTANCE_M = 200.0
DEFAULT_RIPARIAN_BUFFER_M = 50.0
DEFAULT_RF_CATEGORIES = ("RF", "NationalPark")
DEFAULT_TARGET_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32643"  # UTM Zone 43N (planar meters)


class ViolationTier:
    """Legal violation tier classifications."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"


@dataclass
class LegalViolationInfo:
    """Structured legal assessment results for a single incident polygon."""

    patch_id: int
    violation_tier: str
    encroached_rf_name: str | None
    rf_area_acres: float
    riparian_breach: bool
    esz_breach: bool
    distance_to_rf_m: float
    nearest_rf_name: str | None
    violation_reasons: list[str]


def _project_geometry(geom: Any, transformer: pyproj.Transformer) -> Any:
    """Project a shapely geometry using pyproj.Transformer."""
    return shapely.transform(
        geom, lambda c: np.column_stack(transformer.transform(c[:, 0], c[:, 1]))
    )


def _load_boundary_layers(
    db_session: Any | None = None,
    boundaries_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load forest boundaries and watercourses from PostGIS session or GeoJSON files.

    Returns:
        Tuple of (forest_features, water_features) where each feature dictionary has:
        - name: str
        - category (for forests) / buffer_distance_m (for water)
        - geometry: Shapely geometry in EPSG:4326
    """
    forest_features: list[dict[str, Any]] = []
    water_features: list[dict[str, Any]] = []

    if db_session is not None:
        try:
            forest_records = db_session.scalars(select(ForestBoundary)).all()
            for r in forest_records:
                forest_features.append(
                    {
                        "name": r.name,
                        "category": r.category,
                        "geometry": to_shape(r.geom),
                    }
                )

            water_records = db_session.scalars(select(WaterCourse)).all()
            for w in water_records:
                # Use buffer_geom if present; otherwise fallback to geom
                buf_geom = to_shape(w.buffer_geom) if w.buffer_geom is not None else to_shape(w.geom)
                water_features.append(
                    {
                        "name": w.name,
                        "buffer_distance_m": getattr(w, "buffer_distance_m", DEFAULT_RIPARIAN_BUFFER_M),
                        "geometry": buf_geom,
                    }
                )

            if forest_features and water_features:
                logger.info(
                    f"Loaded {len(forest_features)} forest boundaries and {len(water_features)} "
                    "watercourses from active PostGIS session."
                )
                return forest_features, water_features
        except (SQLAlchemyError, AttributeError, TypeError, ValueError, OSError) as exc:
            logger.debug(f"Could not load boundaries from DB session ({exc}). Falling back to local files.")

    # Fallback to local GeoJSON / built-in tracts
    raw_forests = _load_forest_features(data_dir=boundaries_dir)
    raw_water = _load_water_features(data_dir=boundaries_dir)

    for item in raw_forests:
        forest_features.append(
            {
                "name": item["name"],
                "category": item["category"],
                "geometry": item["geometry"],
            }
        )

    for item in raw_water:
        geom = item["geometry"]
        buf_dist = float(item.get("buffer_distance_m", DEFAULT_RIPARIAN_BUFFER_M))
        if geom.geom_type in ("LineString", "MultiLineString"):
            buf_poly = compute_riparian_buffer(geom, buffer_distance_m=buf_dist)
        else:
            buf_poly = geom

        water_features.append(
            {
                "name": item["name"],
                "buffer_distance_m": buf_dist,
                "geometry": buf_poly,
            }
        )

    logger.info(
        f"Loaded {len(forest_features)} forest boundaries and {len(water_features)} "
        "watercourses from boundary data directory/fallbacks."
    )
    return forest_features, water_features


def classify_incident_violations(
    change_gdf: gpd.GeoDataFrame,
    db_session: Any | None = None,
    boundaries_dir: Path | None = None,
    rf_categories: Sequence[str] = DEFAULT_RF_CATEGORIES,
    esz_distance_m: float = DEFAULT_ESZ_DISTANCE_M,
    riparian_buffer_m: float = DEFAULT_RIPARIAN_BUFFER_M,
) -> gpd.GeoDataFrame:
    """Classify detected vector disturbance polygons into statutory legal violation tiers.

    Evaluates:
    1. Reserve Forest intersection: If overlap area with Reserve Forest > 0 -> CRITICAL.
    2. Riparian Buffer intersection: If overlap with 50m watercourse buffer -> HIGH.
    3. Eco-Sensitive Zone (ESZ): If Euclidean distance to nearest RF <= 200m -> HIGH.
    4. Revenue / Private Land: Remaining clearing patches -> MODERATE.

    Args:
        change_gdf: GeoDataFrame of detected canopy disturbance patches (EPSG:4326).
        db_session: Optional active SQLAlchemy session connected to PostGIS.
        boundaries_dir: Optional directory containing forest/watercourse GeoJSON files.
        rf_categories: Tuple of categories considered Reserve Forest (default: ("RF", "NationalPark")).
        esz_distance_m: Eco-Sensitive Zone boundary buffer in meters (default: 200.0).
        riparian_buffer_m: Statutory stream protection buffer distance in meters (default: 50.0).

    Returns:
        Enriched GeoDataFrame with columns:
        - violation_tier: CRITICAL | HIGH | MODERATE
        - encroached_rf_name: str | None
        - rf_area_acres: float
        - riparian_breach: bool
        - esz_breach: bool
        - distance_to_rf_m: float
        - nearest_rf_name: str | None
        - violation_reasons: list[str]
    """
    out_gdf = change_gdf.copy()

    standard_cols = [
        "violation_tier",
        "encroached_rf_name",
        "rf_area_acres",
        "riparian_breach",
        "esz_breach",
        "distance_to_rf_m",
        "nearest_rf_name",
        "violation_reasons",
    ]

    if out_gdf.empty:
        for col in standard_cols:
            out_gdf[col] = []
        return out_gdf

    # Ensure source CRS is set
    if out_gdf.crs is None:
        out_gdf.set_crs(DEFAULT_TARGET_CRS, inplace=True)
    elif str(out_gdf.crs).upper() != DEFAULT_TARGET_CRS:
        out_gdf = out_gdf.to_crs(DEFAULT_TARGET_CRS)

    # 1. Load boundary layers
    forest_items, water_items = _load_boundary_layers(
        db_session=db_session,
        boundaries_dir=boundaries_dir,
    )

    # 2. Build transformers for planar calculations in UTM Zone 43N
    t_utm = pyproj.Transformer.from_crs(DEFAULT_TARGET_CRS, PROJECTED_CRS, always_xy=True)

    # Pre-project boundaries to UTM Zone 43N
    rf_features_utm: list[dict[str, Any]] = []
    for item in forest_items:
        if item["category"] in rf_categories:
            geom_utm = _project_geometry(item["geometry"], t_utm)
            rf_features_utm.append(
                {
                    "name": item["name"],
                    "category": item["category"],
                    "geom_utm": geom_utm,
                }
            )

    water_features_utm: list[dict[str, Any]] = []
    for item in water_items:
        geom_utm = _project_geometry(item["geometry"], t_utm)
        water_features_utm.append(
            {
                "name": item["name"],
                "geom_utm": geom_utm,
            }
        )

    # 3. Classify each incident polygon
    violation_tiers: list[str] = []
    encroached_rfs: list[str | None] = []
    rf_areas: list[float] = []
    riparian_breaches: list[bool] = []
    esz_breaches: list[bool] = []
    distances_to_rf: list[float] = []
    nearest_rfs: list[str | None] = []
    violation_reasons_list: list[list[str]] = []

    for _, row in out_gdf.iterrows():
        poly_wgs = row.geometry
        poly_utm = _project_geometry(poly_wgs, t_utm)

        tier = ViolationTier.MODERATE
        encroached_rf_name: str | None = None
        rf_area_acres = 0.0
        riparian_breach = False
        encroached_water_name: str | None = None
        esz_breach = False
        min_rf_dist = float("inf")
        nearest_rf_name: str | None = None
        reasons: list[str] = []

        # A. Check Reserve Forest intersections
        max_rf_overlap_sqm = 0.0
        for rf in rf_features_utm:
            rf_geom = rf["geom_utm"]
            if poly_utm.intersects(rf_geom):
                intersection = poly_utm.intersection(rf_geom)
                inter_sqm = float(intersection.area)
                if inter_sqm > max_rf_overlap_sqm:
                    max_rf_overlap_sqm = inter_sqm
                    encroached_rf_name = rf["name"]

        if max_rf_overlap_sqm > 0.0:
            tier = ViolationTier.CRITICAL
            rf_area_acres = round(float(max_rf_overlap_sqm / 4046.8564), 4)
            min_rf_dist = 0.0
            nearest_rf_name = encroached_rf_name
            esz_breach = True
            reasons.append(
                f"Inside Reserve Forest: {encroached_rf_name} ({rf_area_acres:.2f} acres encroached)"
            )
        else:
            # Calculate distance to nearest RF boundary
            for rf in rf_features_utm:
                dist = float(poly_utm.distance(rf["geom_utm"]))
                if dist < min_rf_dist:
                    min_rf_dist = dist
                    nearest_rf_name = rf["name"]

            if min_rf_dist <= esz_distance_m:
                esz_breach = True
                tier = ViolationTier.HIGH
                reasons.append(
                    f"Within {esz_distance_m:.0f}m Eco-Sensitive Zone of {nearest_rf_name} "
                    f"({min_rf_dist:.1f}m away)"
                )

        # B. Check 50m Riparian Buffer intersections
        for water in water_features_utm:
            water_buf_geom = water["geom_utm"]
            if poly_utm.intersects(water_buf_geom):
                buf_inter = poly_utm.intersection(water_buf_geom)
                if float(buf_inter.area) > 0.0:
                    riparian_breach = True
                    encroached_water_name = water["name"]
                    break

        if riparian_breach:
            if tier != ViolationTier.CRITICAL:
                tier = ViolationTier.HIGH
            reasons.append(
                f"Riparian buffer breach: {encroached_water_name} (50m stream protection corridor)"
            )

        # C. Default reason for MODERATE
        if tier == ViolationTier.MODERATE:
            if nearest_rf_name and min_rf_dist < float("inf"):
                reasons.append(
                    f"Private/Revenue agricultural change ({min_rf_dist:.1f}m from {nearest_rf_name})"
                )
            else:
                reasons.append("Private/Revenue agricultural change outside forest protection buffers")

        violation_tiers.append(tier)
        encroached_rfs.append(encroached_rf_name)
        rf_areas.append(rf_area_acres)
        riparian_breaches.append(riparian_breach)
        esz_breaches.append(esz_breach)
        distances_to_rf.append(round(min_rf_dist, 1) if min_rf_dist < float("inf") else 0.0)
        nearest_rfs.append(nearest_rf_name)
        violation_reasons_list.append(reasons)

    out_gdf["violation_tier"] = violation_tiers
    out_gdf["encroached_rf_name"] = encroached_rfs
    out_gdf["rf_area_acres"] = rf_areas
    out_gdf["riparian_breach"] = riparian_breaches
    out_gdf["esz_breach"] = esz_breaches
    out_gdf["distance_to_rf_m"] = distances_to_rf
    out_gdf["nearest_rf_name"] = nearest_rfs
    out_gdf["violation_reasons"] = violation_reasons_list

    return out_gdf


def summarize_violations(classified_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Generate high-level aggregated audit metrics from classified violations."""
    if classified_gdf.empty:
        return {
            "total_incidents": 0,
            "total_area_acres": 0.0,
            "critical_count": 0,
            "critical_acres": 0.0,
            "high_count": 0,
            "high_acres": 0.0,
            "moderate_count": 0,
            "moderate_acres": 0.0,
            "riparian_breaches": 0,
            "esz_breaches": 0,
            "encroached_forest_tracts": [],
        }

    total_incidents = len(classified_gdf)
    total_area_acres = float(classified_gdf["area_acres"].sum())

    crit_mask = classified_gdf["violation_tier"] == ViolationTier.CRITICAL
    high_mask = classified_gdf["violation_tier"] == ViolationTier.HIGH
    mod_mask = classified_gdf["violation_tier"] == ViolationTier.MODERATE

    critical_count = int(crit_mask.sum())
    critical_acres = float(classified_gdf.loc[crit_mask, "area_acres"].sum())

    high_count = int(high_mask.sum())
    high_acres = float(classified_gdf.loc[high_mask, "area_acres"].sum())

    moderate_count = int(mod_mask.sum())
    moderate_acres = float(classified_gdf.loc[mod_mask, "area_acres"].sum())

    riparian_breaches = int(classified_gdf["riparian_breach"].sum())
    esz_breaches = int(classified_gdf["esz_breach"].sum())

    # Forest tract breakdown
    tract_breakdown: dict[str, dict[str, Any]] = {}
    for _, row in classified_gdf[crit_mask].iterrows():
        rf = row["encroached_rf_name"]
        if rf:
            if rf not in tract_breakdown:
                tract_breakdown[rf] = {"name": rf, "incidents": 0, "rf_area_acres": 0.0}
            tract_breakdown[rf]["incidents"] += 1
            tract_breakdown[rf]["rf_area_acres"] += float(row.get("rf_area_acres", 0.0))

    return {
        "total_incidents": total_incidents,
        "total_area_acres": round(total_area_acres, 2),
        "critical_count": critical_count,
        "critical_acres": round(critical_acres, 2),
        "high_count": high_count,
        "high_acres": round(high_acres, 2),
        "moderate_count": moderate_count,
        "moderate_acres": round(moderate_acres, 2),
        "riparian_breaches": riparian_breaches,
        "esz_breaches": esz_breaches,
        "encroached_forest_tracts": list(tract_breakdown.values()),
    }
