"""API routes for Areas of Interest, gazetted forest boundaries, and stream buffers."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from haddu.api.schemas import (
    AOIResponse,
    ForestBoundaryResponse,
    GeoJSONFeatureCollection,
    TalukBoundsResponse,
    WaterCourseResponse,
)
from haddu.core.stac_client import DEFAULT_AOI_BBOX
from haddu.db.models import AOI, ForestBoundary, WaterCourse
from haddu.db.session import get_db_session

router = APIRouter(prefix="/api/v1/aoi", tags=["AOI & Boundaries"])


@router.get("", response_model=list[AOIResponse])
def list_aois(db: Annotated[Session, Depends(get_db_session)]) -> list[AOIResponse]:
    """List monitored administrative Areas of Interest (Taluks / Ranges)."""
    aois = db.scalars(select(AOI).order_by(AOI.name)).all()
    return list(aois)


@router.get("/bounds", response_model=TalukBoundsResponse)
def get_taluk_bounds() -> TalukBoundsResponse:
    """Return default geographic bounding coordinates for Thirthahalli Taluk."""
    return TalukBoundsResponse(
        name="Thirthahalli Taluk",
        bbox=list(DEFAULT_AOI_BBOX),
        crs="EPSG:4326",
    )


@router.get("/forest-boundaries", response_model=list[ForestBoundaryResponse])
def list_forest_boundaries(
    db: Annotated[Session, Depends(get_db_session)],
    category: Annotated[str | None, Query(description="Filter by category ('RF', 'NationalPark', 'Gomala')")] = None,
) -> list[ForestBoundaryResponse]:
    """List gazetted Reserve Forest boundaries and protected ecological tracts."""
    query = select(ForestBoundary)
    if category:
        query = query.where(func.upper(ForestBoundary.category) == category.strip().upper())
    forests = db.scalars(query.order_by(ForestBoundary.name)).all()
    return list(forests)


@router.get("/forest-boundaries/geojson", response_model=GeoJSONFeatureCollection)
def get_forest_boundaries_geojson(
    db: Annotated[Session, Depends(get_db_session)],
    category: Annotated[str | None, Query(description="Filter by category ('RF', 'NationalPark', 'Gomala')")] = None,
) -> GeoJSONFeatureCollection:
    """Export gazetted forest boundaries as GeoJSON FeatureCollection for MapLibre/QGIS."""
    query = select(ForestBoundary)
    if category:
        query = query.where(func.upper(ForestBoundary.category) == category.strip().upper())
    forests = db.scalars(query.order_by(ForestBoundary.name)).all()

    features = []
    for f in forests:
        geom_dict = mapping(to_shape(f.geom)) if f.geom is not None else {}
        props = {
            "id": str(f.id),
            "name": f.name,
            "category": f.category,
            "area_acres": f.area_acres,
            "area_sqm": f.area_sqm,
        }
        features.append({
            "type": "Feature",
            "id": str(f.id),
            "geometry": geom_dict,
            "properties": props,
        })

    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=features,
    )


@router.get("/water-courses", response_model=list[WaterCourseResponse])
def list_water_courses(db: Annotated[Session, Depends(get_db_session)]) -> list[WaterCourseResponse]:
    """List watercourses and statutory riparian buffer corridors."""
    watercourses = db.scalars(select(WaterCourse).order_by(WaterCourse.name)).all()
    return list(watercourses)


@router.get("/water-courses/geojson", response_model=GeoJSONFeatureCollection)
def get_water_courses_geojson(
    db: Annotated[Session, Depends(get_db_session)],
    include_buffers: Annotated[bool, Query(description="If True, exports 50m buffer polygon; if False, stream line")] = True,
) -> GeoJSONFeatureCollection:
    """Export watercourse corridors and riparian protection buffers as GeoJSON."""
    watercourses = db.scalars(select(WaterCourse).order_by(WaterCourse.name)).all()

    features = []
    for w in watercourses:
        target_geom = w.buffer_geom if include_buffers and w.buffer_geom is not None else w.geom
        geom_dict = mapping(to_shape(target_geom)) if target_geom is not None else {}
        props = {
            "id": str(w.id),
            "name": w.name,
            "buffer_distance_m": w.buffer_distance_m,
            "is_buffer": include_buffers,
        }
        features.append({
            "type": "Feature",
            "id": str(w.id),
            "geometry": geom_dict,
            "properties": props,
        })

    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=features,
    )
