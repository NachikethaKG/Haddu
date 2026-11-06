"""Spatial and relational SQLAlchemy ORM models for Haddu.

Includes AOI (Area of Interest) and SatellitePass tables with PostGIS Geometry support.
"""

import uuid
from datetime import datetime, date
from typing import Optional, List, Any
from sqlalchemy import (
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    func,
    JSON,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from geoalchemy2 import Geometry


class Base(DeclarativeBase):
    """Base declarative class for Haddu database models."""
    pass


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
    passes: Mapped[List["SatellitePass"]] = relationship(
        "SatellitePass",
        back_populates="aoi",
        cascade="all, delete-orphan",
        order_by="desc(SatellitePass.pass_date)",
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
    cloud_cover_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    stac_item_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )
    metadata_json: Mapped[Optional[dict]] = mapped_column(
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
