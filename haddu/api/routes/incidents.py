"""API routes for querying and exporting encroachment incidents and legal tiers."""

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from haddu.api.schemas import (
    GeoJSONFeatureCollection,
    IncidentListResponse,
    IncidentResponse,
    IncidentSummaryResponse,
)
from haddu.core.classifier import ViolationTier
from haddu.db.models import EncroachmentIncident
from haddu.db.session import get_db_session
from haddu.reporting import compile_dossier_pdf

router = APIRouter(prefix="/api/v1/incidents", tags=["Incidents"])


def _apply_incident_filters(
    query,
    violation_tier: str | None = None,
    min_acres: float | None = None,
    max_acres: float | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sensor: str | None = None,
):
    """Apply query parameter filters to SQLAlchemy query."""
    if violation_tier:
        query = query.where(func.upper(EncroachmentIncident.violation_tier) == violation_tier.strip().upper())
    if min_acres is not None:
        query = query.where(EncroachmentIncident.area_acres >= min_acres)
    if max_acres is not None:
        query = query.where(EncroachmentIncident.area_acres <= max_acres)
    if start_date is not None:
        query = query.where(EncroachmentIncident.post_date >= start_date)
    if end_date is not None:
        query = query.where(EncroachmentIncident.post_date <= end_date)
    if sensor:
        query = query.where(func.lower(EncroachmentIncident.sensor) == sensor.strip().lower())
    return query


@router.get("", response_model=IncidentListResponse)
def list_incidents(
    db: Annotated[Session, Depends(get_db_session)],
    violation_tier: Annotated[str | None, Query(description="Filter by legal tier (CRITICAL, HIGH, MODERATE)")] = None,
    min_acres: Annotated[float | None, Query(description="Minimum incident area in acres")] = None,
    max_acres: Annotated[float | None, Query(description="Maximum incident area in acres")] = None,
    start_date: Annotated[date | None, Query(description="Filter by post_date >= start_date (YYYY-MM-DD)")] = None,
    end_date: Annotated[date | None, Query(description="Filter by post_date <= end_date (YYYY-MM-DD)")] = None,
    sensor: Annotated[str | None, Query(description="Filter by sensor platform ('s1' or 's2')")] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Page limit")] = 100,
    offset: Annotated[int, Query(ge=0, description="Page offset")] = 0,
) -> IncidentListResponse:
    """Retrieve filtered list of flagged forest encroachment incidents."""
    base_query = select(EncroachmentIncident)
    filtered_query = _apply_incident_filters(
        base_query,
        violation_tier=violation_tier,
        min_acres=min_acres,
        max_acres=max_acres,
        start_date=start_date,
        end_date=end_date,
        sensor=sensor,
    )

    # Count total
    count_query = select(func.count()).select_from(filtered_query.subquery())
    total_count = db.scalar(count_query) or 0

    # Paginate
    records_query = (
        filtered_query
        .order_by(EncroachmentIncident.post_date.desc(), EncroachmentIncident.area_acres.desc())
        .offset(offset)
        .limit(limit)
    )
    incidents = db.scalars(records_query).all()

    return IncidentListResponse(
        total=total_count,
        limit=limit,
        offset=offset,
        incidents=list(incidents),
    )


@router.get("/summary", response_model=IncidentSummaryResponse)
def get_incident_summary(
    db: Annotated[Session, Depends(get_db_session)],
    start_date: Annotated[date | None, Query(description="Filter by post_date >= start_date")] = None,
    end_date: Annotated[date | None, Query(description="Filter by post_date <= end_date")] = None,
    sensor: Annotated[str | None, Query(description="Filter by sensor ('s1' or 's2')")] = None,
) -> IncidentSummaryResponse:
    """Retrieve aggregate violation statistics and risk tier metrics."""
    query = select(EncroachmentIncident)
    query = _apply_incident_filters(query, start_date=start_date, end_date=end_date, sensor=sensor)
    incidents = db.scalars(query).all()

    total_incidents = len(incidents)
    total_acres = sum(inc.area_acres for inc in incidents)

    crit = [i for i in incidents if (i.violation_tier or "").upper() == ViolationTier.CRITICAL]
    high = [i for i in incidents if (i.violation_tier or "").upper() == ViolationTier.HIGH]
    mod = [i for i in incidents if (i.violation_tier or "").upper() == ViolationTier.MODERATE]
    riparian = [i for i in incidents if i.riparian_breach]

    return IncidentSummaryResponse(
        total_incidents=total_incidents,
        total_acres=round(total_acres, 2),
        critical_count=len(crit),
        critical_acres=round(sum(i.area_acres for i in crit), 2),
        high_count=len(high),
        high_acres=round(sum(i.area_acres for i in high), 2),
        moderate_count=len(mod),
        moderate_acres=round(sum(i.area_acres for i in mod), 2),
        riparian_breaches=len(riparian),
    )


@router.get("/geojson", response_model=GeoJSONFeatureCollection)
def get_incidents_geojson(
    db: Annotated[Session, Depends(get_db_session)],
    violation_tier: Annotated[str | None, Query(description="Filter by legal tier (CRITICAL, HIGH, MODERATE)")] = None,
    min_acres: Annotated[float | None, Query(description="Minimum incident area in acres")] = None,
    max_acres: Annotated[float | None, Query(description="Maximum incident area in acres")] = None,
    start_date: Annotated[date | None, Query(description="Filter by post_date >= start_date")] = None,
    end_date: Annotated[date | None, Query(description="Filter by post_date <= end_date")] = None,
    sensor: Annotated[str | None, Query(description="Filter by sensor ('s1' or 's2')")] = None,
    limit: Annotated[int, Query(ge=1, le=5000, description="Max features to return")] = 500,
) -> GeoJSONFeatureCollection:
    """Export encroachment incident vector boundaries formatted for QGIS and MapLibre."""
    query = select(EncroachmentIncident)
    query = _apply_incident_filters(
        query,
        violation_tier=violation_tier,
        min_acres=min_acres,
        max_acres=max_acres,
        start_date=start_date,
        end_date=end_date,
        sensor=sensor,
    ).order_by(EncroachmentIncident.post_date.desc()).limit(limit)

    records = db.scalars(query).all()
    features = []

    for inc in records:
        geom_dict = mapping(to_shape(inc.geom)) if inc.geom is not None else {}
        props = {
            "id": str(inc.id),
            "pre_date": inc.pre_date.isoformat(),
            "post_date": inc.post_date.isoformat(),
            "area_sqm": inc.area_sqm,
            "area_acres": inc.area_acres,
            "centroid_lat": inc.centroid_lat,
            "centroid_lon": inc.centroid_lon,
            "violation_tier": inc.violation_tier,
            "encroached_rf_name": inc.encroached_rf_name,
            "rf_area_acres": inc.rf_area_acres,
            "riparian_breach": bool(inc.riparian_breach),
            "sensor": inc.sensor,
            "mean_delta_ndvi": inc.mean_delta_ndvi,
            "confidence_score": inc.confidence_score,
            "properties": inc.properties_json,
        }
        features.append({
            "type": "Feature",
            "id": str(inc.id),
            "geometry": geom_dict,
            "properties": props,
        })

    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=features,
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(
    incident_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db_session)],
) -> IncidentResponse:
    """Retrieve details for a specific encroachment incident by UUID."""
    incident = db.scalar(select(EncroachmentIncident).where(EncroachmentIncident.id == incident_id))
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return incident


@router.get(
    "/{incident_id}/dossier",
    summary="Generate legal evidence dossier PDF",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Returns a compiled single-page A4 legal evidence dossier PDF.",
        },
        404: {"description": "Incident not found."},
    },
)
def get_incident_dossier(
    incident_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db_session)],
) -> Response:
    """Compile and return formal legal evidence dossier (PDF) for an encroachment incident."""
    incident = db.scalar(select(EncroachmentIncident).where(EncroachmentIncident.id == incident_id))
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")

    pdf_bytes = compile_dossier_pdf(incident)
    filename = f"Haddu_Dossier_{incident_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )
