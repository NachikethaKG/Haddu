"""CLI scripts and execution entrypoints."""

from haddu.scripts.fetch_pass import fetch_and_export_pass
from haddu.scripts.run_audit import run_audit_pipeline
from haddu.scripts.seed_aoi import seed_areas_of_interest
from haddu.scripts.seed_boundaries import (
    compute_riparian_buffer,
    seed_forest_and_water_boundaries,
)

__all__ = [
    "compute_riparian_buffer",
    "fetch_and_export_pass",
    "run_audit_pipeline",
    "seed_areas_of_interest",
    "seed_forest_and_water_boundaries",
]
