"""Pydantic response and query models for Haddu API."""

import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class IncidentResponse(BaseModel):
    """Pydantic model representing an encroachment incident."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    aoi_id: uuid.UUID | None = None
    pre_date: date
    post_date: date
    area_sqm: float
    area_acres: float
    centroid_lat: float
    centroid_lon: float
    mean_delta_ndvi: float | None = None
    max_delta_ndvi: float | None = None
    violation_tier: str | None = None
    encroached_rf_name: str | None = None
    rf_area_acres: float | None = None
    riparian_breach: bool | None = False
    sensor: str | None = None
    confidence_score: float | None = None
    break_date: date | None = None
    properties_json: dict[str, Any] | None = None
    created_at: datetime


class IncidentListResponse(BaseModel):
    """Paginated list of encroachment incidents."""

    total: int
    limit: int
    offset: int
    incidents: list[IncidentResponse]


class IncidentSummaryResponse(BaseModel):
    """High-level legal violation metrics across all detected incidents."""

    total_incidents: int
    total_acres: float
    critical_count: int
    critical_acres: float
    high_count: int
    high_acres: float
    moderate_count: int
    moderate_acres: float
    riparian_breaches: int


class AOIResponse(BaseModel):
    """Area of Interest metadata model."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    category: str
    created_at: datetime


class ForestBoundaryResponse(BaseModel):
    """Gazetted forest boundary model."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    category: str
    area_sqm: float | None = None
    area_acres: float | None = None
    created_at: datetime


class WaterCourseResponse(BaseModel):
    """Watercourse and statutory riparian buffer corridor model."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    buffer_distance_m: float
    created_at: datetime


class TalukBoundsResponse(BaseModel):
    """Geographic bounding box for the monitored taluk."""

    name: str
    bbox: list[float]
    crs: str = "EPSG:4326"


class GeoJSONFeature(BaseModel):
    """Standard GeoJSON Feature object."""

    type: str = "Feature"
    id: str | None = None
    geometry: dict[str, Any]
    properties: dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    """Standard GeoJSON FeatureCollection object."""

    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]
