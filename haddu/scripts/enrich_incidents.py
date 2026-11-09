"""Enrichment and spatial classification utility for EncroachmentIncident records.

Cross-references unclassified encroachment incidents against registered ForestBoundaries
and statutory riparian corridors to populate violation_tier, encroached_rf_name,
rf_area_acres, riparian_breach, sensor, and confidence_score.
"""

import logging
import sys
from typing import Any
import geopandas as gpd
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from haddu.core.classifier import classify_incident_violations
from haddu.db.models import EncroachmentIncident
from haddu.db.session import SessionLocal, get_db

logger = logging.getLogger("haddu.enrichment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def enrich_unclassified_incidents(session: Session | None = None) -> int:
    """Classify all incidents in the database that currently lack violation_tier.

    Returns:
        Number of updated incidents.
    """
    def _run(db: Session) -> int:
        unclassified = db.scalars(
            select(EncroachmentIncident).where(EncroachmentIncident.violation_tier.is_(None))
        ).all()

        if not unclassified:
            logger.info("All incidents are already classified.")
            return 0

        logger.info("Found %d unclassified incidents. Computing spatial intersections...", len(unclassified))

        records: list[dict[str, Any]] = []
        id_to_model: dict[str, EncroachmentIncident] = {}

        for idx, inc in enumerate(unclassified):
            inc_id_str = str(inc.id)
            id_to_model[inc_id_str] = inc
            records.append({
                "incident_id": inc_id_str,
                "patch_id": idx + 1,
                "area_sqm": inc.area_sqm,
                "area_acres": inc.area_acres,
                "geometry": to_shape(inc.geom),
            })

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        classified_gdf = classify_incident_violations(gdf, db_session=db)

        updated_count = 0
        for _, row in classified_gdf.iterrows():
            inc_id = str(row["incident_id"])
            inc = id_to_model.get(inc_id)
            if not inc:
                continue

            tier = str(row["violation_tier"]).upper()
            inc.violation_tier = tier
            inc.encroached_rf_name = row.get("encroached_rf_name")
            inc.rf_area_acres = float(row.get("rf_area_acres") or 0.0) if row.get("rf_area_acres") else None
            inc.riparian_breach = bool(row.get("riparian_breach", False))

            if not inc.sensor:
                inc.sensor = "Sentinel-2 MSI"

            if inc.confidence_score is None:
                delta = abs(inc.mean_delta_ndvi) if inc.mean_delta_ndvi is not None else 0.3
                inc.confidence_score = round(min(0.99, max(0.65, delta * 2.2)), 2)

            updated_count += 1

        db.commit()
        logger.info("Successfully enriched and committed %d incidents.", updated_count)
        return updated_count

    if session:
        return _run(session)
    with SessionLocal() as db:
        return _run(db)


def main() -> None:
    """CLI entrypoint for enriching incidents."""
    count = enrich_unclassified_incidents()
    print(f"Enrichment completed. Updated {count} incidents.")


if __name__ == "__main__":
    main()
