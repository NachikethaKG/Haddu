"""Boundary dataset loaders and riparian buffer geometry computations.

Loads gazetted forest boundaries and watercourses from GeoJSON files or built-in
definitions for Thirthahalli Taluk and Western Ghats eco-zones.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape

logger = logging.getLogger(__name__)

DATA_BOUNDARIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "boundaries"

# Built-in definitions as fallback if GeoJSON files are missing
FALLBACK_FOREST_TRACTS: list[dict[str, Any]] = [
    {
        "name": "Agumbe Rainforest Reserve Forest",
        "category": "RF",
        "coords": [
            [
                (75.06, 13.55),
                (75.14, 13.55),
                (75.14, 13.62),
                (75.06, 13.62),
                (75.06, 13.55),
            ]
        ],
    },
    {
        "name": "Agumbe National Park Eco-Zone",
        "category": "NationalPark",
        "coords": [
            [
                (75.05, 13.55),
                (75.08, 13.55),
                (75.08, 13.60),
                (75.05, 13.60),
                (75.05, 13.55),
            ]
        ],
    },
    {
        "name": "Kundadri Peak Reserve Forest",
        "category": "RF",
        "coords": [
            [
                (75.15, 13.56),
                (75.22, 13.56),
                (75.22, 13.63),
                (75.15, 13.63),
                (75.15, 13.56),
            ]
        ],
    },
    {
        "name": "Megaravalli State Forest",
        "category": "RF",
        "coords": [
            [
                (75.10, 13.64),
                (75.18, 13.64),
                (75.18, 13.72),
                (75.10, 13.72),
                (75.10, 13.64),
            ]
        ],
    },
    {
        "name": "Thirthahalli Gomala Pasture Land",
        "category": "Gomala",
        "coords": [
            [
                (75.22, 13.67),
                (75.28, 13.67),
                (75.28, 13.72),
                (75.22, 13.72),
                (75.22, 13.67),
            ]
        ],
    },
]

FALLBACK_WATER_COURSES: list[dict[str, Any]] = [
    {
        "name": "Tunga River Corridor",
        "buffer_distance_m": 50.0,
        "coords": [
            (75.16, 13.61),
            (75.21, 13.64),
            (75.25, 13.67),
            (75.29, 13.69),
            (75.33, 13.73),
            (75.38, 13.78),
        ],
    }
]


def compute_riparian_buffer(
    geom_4326: LineString | MultiLineString,
    buffer_distance_m: float = 50.0,
) -> MultiPolygon:
    """Compute projected 50m buffer in UTM Zone 43N and return as WGS84 MultiPolygon.

    Matches PostGIS ST_Transform(ST_Buffer(ST_Transform(geom, 32643), buffer_distance_m), 4326).

    Args:
        geom_4326: LineString or MultiLineString in EPSG:4326.
        buffer_distance_m: Buffer radius in meters (default: 50m).

    Returns:
        MultiPolygon in EPSG:4326.
    """
    t_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    t_wgs = pyproj.Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

    geom_utm = shapely.transform(
        geom_4326, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1]))
    )
    buffer_utm = geom_utm.buffer(buffer_distance_m)
    buffer_wgs = shapely.transform(
        buffer_utm, lambda c: np.column_stack(t_wgs.transform(c[:, 0], c[:, 1]))
    )

    if buffer_wgs.geom_type == "Polygon":
        return MultiPolygon([buffer_wgs])
    elif buffer_wgs.geom_type == "MultiPolygon":
        return buffer_wgs
    else:
        raise ValueError(f"Unexpected buffered geometry type: {buffer_wgs.geom_type}")


def _load_forest_features(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load forest boundaries from GeoJSON or fall back to built-in definitions."""
    base_dir = data_dir or DATA_BOUNDARIES_DIR
    geojson_path = base_dir / "forest_boundaries.geojson"

    if geojson_path.exists():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = []
            for feat in data.get("features", []):
                geom = shape(feat["geometry"])
                if geom.geom_type == "Polygon":
                    multi_geom = MultiPolygon([geom])
                elif geom.geom_type == "MultiPolygon":
                    multi_geom = geom
                else:
                    continue
                features.append(
                    {
                        "name": feat["properties"]["name"],
                        "category": feat["properties"]["category"],
                        "geometry": multi_geom,
                    }
                )
            if features:
                return features
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to read {geojson_path}: {exc}. Using built-in fallbacks.")

    # Built-in fallback
    features = []
    for item in FALLBACK_FOREST_TRACTS:
        polys = [Polygon(ring) for ring in item["coords"]]
        features.append(
            {
                "name": item["name"],
                "category": item["category"],
                "geometry": MultiPolygon(polys),
            }
        )
    return features


def _load_water_features(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load water course features from GeoJSON or fall back to built-in definitions."""
    base_dir = data_dir or DATA_BOUNDARIES_DIR
    geojson_path = base_dir / "water_courses.geojson"

    if geojson_path.exists():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = []
            for feat in data.get("features", []):
                geom = shape(feat["geometry"])
                if geom.geom_type == "LineString":
                    multi_geom = MultiLineString([geom])
                elif geom.geom_type == "MultiLineString":
                    multi_geom = geom
                else:
                    continue
                features.append(
                    {
                        "name": feat["properties"]["name"],
                        "buffer_distance_m": float(
                            feat["properties"].get("buffer_distance_m", 50.0)
                        ),
                        "geometry": multi_geom,
                    }
                )
            if features:
                return features
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to read {geojson_path}: {exc}. Using built-in fallbacks.")

    # Built-in fallback
    features = []
    for item in FALLBACK_WATER_COURSES:
        line = LineString(item["coords"])
        features.append(
            {
                "name": item["name"],
                "buffer_distance_m": float(item["buffer_distance_m"]),
                "geometry": MultiLineString([line]),
            }
        )
    return features
