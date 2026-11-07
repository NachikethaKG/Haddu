"""FastAPI application entrypoint for Haddu Satellite Encroachment Engine.

Exposes REST and GeoJSON endpoints for querying canopy disturbance incidents,
gazetted forest boundaries, and mounts TiTiler dynamic tile server for Cloud-Optimized
GeoTIFF (COG) streaming.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titiler.core.factory import TilerFactory

from haddu.api.routes import aoi, incidents

logger = logging.getLogger("haddu.api")

app = FastAPI(
    title="Haddu Satellite Encroachment API",
    description=(
        "Automated satellite change detection engine and legal enforcement backend "
        "monitoring forest canopy disturbances across Thirthahalli, Western Ghats."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable Cross-Origin Resource Sharing (CORS) for web GIS clients (MapLibre, QGIS, Leaflet)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount TiTiler dynamic tile factory router onto /api/v1/tiles
tiler = TilerFactory()
app.include_router(
    tiler.router,
    prefix="/api/v1/tiles",
    tags=["Tiles"],
)

# Include business domain route handlers
app.include_router(incidents.router)
app.include_router(aoi.router)


@app.get("/", tags=["Health"])
def root_status() -> dict[str, str]:
    """Root service information and documentation link."""
    return {
        "service": "Haddu Satellite Encroachment API",
        "version": "0.1.0",
        "status": "healthy",
        "docs_url": "/docs",
        "tiles_endpoint": "/api/v1/tiles",
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}
