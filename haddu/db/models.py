"""Spatial and relational SQLAlchemy ORM models for Haddu.

Includes AOI (Area of Interest) and SatellitePass tables with PostGIS Geometry support.
"""

import uuid
from datetime import date, datetime
from typing import Any, Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for Haddu database models."""


class AOI(Base):
    """Area of Interest representing administrative taluks, reserve forests, or river corridors."""

    __tablename__ = "aoi"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Category: RF (Reserve Forest), Stream, Taluk",
    )
    geom: Mapped[Any] = mapped_column(
        Geometry(
            geometry_type="POLYGON",
            srid=4326,
            spatial_index=True,
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    passes: Mapped[list["SatellitePass"]] = relationship(
        "SatellitePass",
        back_populates="aoi",
        cascade="all, delete-orphan",
        order_by="desc(SatellitePass.pass_date)",
    )
    incidents: Mapped[list["EncroachmentIncident"]] = relationship(
        "EncroachmentIncident",
        back_populates="aoi",
        cascade="all, delete-orphan",
        order_by="desc(EncroachmentIncident.post_date)",
    )

    def __repr__(self) -> str:
        return f"<AOI(name='{self.name}', category='{self.category}')>"


class SatellitePass(Base):
    """Metadata record for an Earth Observation satellite pass over a registered AOI."""

    __tablename__ = "satellite_pass"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    aoi_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("aoi.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Platform: Sentinel-2A, Sentinel-2B, etc.",
    )
    pass_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    cloud_cover_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    stac_item_id: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    aoi: Mapped["AOI"] = relationship(
        "AOI",
        back_populates="passes",
    )

    def __repr__(self) -> str:
        return (
            f"<SatellitePass(platform='{self.platform}', "
            f"date='{self.pass_date}', cloud={self.cloud_cover_pct}%)>"
        )


class EncroachmentIncident(Base):
    """Detected canopy loss / forest encroachment polygon incident."""

    __tablename__ = "encroachment_incidents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    aoi_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("aoi.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pre_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    post_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )
    area_sqm: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    area_acres: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    centroid_lat: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    centroid_lon: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    mean_delta_ndvi: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    max_delta_ndvi: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    geom: Mapped[Any] = mapped_column(
        Geometry(
            geometry_type="POLYGON",
            srid=4326,
            spatial_index=True,
        ),
        nullable=False,
    )
    break_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )
    confidence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    violation_tier: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Legal violation tier: CRITICAL, HIGH, or MODERATE",
    )
    encroached_rf_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Name of Reserve Forest breached, if any",
    )
    rf_area_acres: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Area within Reserve Forest in acres",
    )
    riparian_breach: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        default=False,
        comment="Whether 50m riparian watercourse buffer was breached",
    )
    sensor: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Sensor platform: s1 (SAR) or s2 (Optical)",
    )
    properties_json: Mapped[dict | None] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    aoi: Mapped[Optional["AOI"]] = relationship(
        "AOI",
        back_populates="incidents",
    )

    def __repr__(self) -> str:
        tier_info = f", tier={self.violation_tier}" if self.violation_tier else ""
        return (
            f"<EncroachmentIncident(id={self.id}, "
            f"pre={self.pre_date}, post={self.post_date}, "
            f"area={self.area_acres:.2f} acres{tier_info})>"
        )


class ForestBoundary(Base):
    """Gazetted forest boundary (Reserve Forest, National Park, Gomala / Community Land)."""

    __tablename__ = "forest_boundaries"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Category: RF (Reserve Forest), NationalPark, Gomala",
    )
    geom: Mapped[Any] = mapped_column(
        Geometry(
            geometry_type="MULTIPOLYGON",
            srid=4326,
            spatial_index=True,
        ),
        nullable=False,
    )
    area_sqm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    area_acres: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ForestBoundary(name='{self.name}', category='{self.category}')>"


class WaterCourse(Base):
    """River, stream, or water channel corridor with statutory riparian protection buffer."""

    __tablename__ = "water_courses"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    geom: Mapped[Any | None] = mapped_column(
        Geometry(
            geometry_type="MULTILINESTRING",
            srid=4326,
            spatial_index=True,
        ),
        nullable=True,
    )
    buffer_geom: Mapped[Any] = mapped_column(
        Geometry(
            geometry_type="MULTIPOLYGON",
            srid=4326,
            spatial_index=True,
        ),
        nullable=False,
    )
    buffer_distance_m: Mapped[float] = mapped_column(
        Float,
        default=50.0,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<WaterCourse(name='{self.name}', buffer={self.buffer_distance_m}m)>"

