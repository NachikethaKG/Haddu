# Haddu (ಹದ್ದು)

> An open-source satellite change detection engine monitoring forest encroachment in Thirthahalli Taluk, Western Ghats, Karnataka.

*Haddu* (meaning "Eagle" / "Hawk" in Kannada, symbolising keen aerial vision) leverages cloud-native Earth Observation (EO) data from Sentinel-2 and high-performance geospatial databases to detect and alert against forest loss and encroachment across ecologically sensitive Western Ghats corridors.

---

## Geographic Focus & Spatial Configuration

- **Target AOI:** Thirthahalli Taluk, Shimoga District, Karnataka
- **Bounding Box (EPSG:4326):** `[75.05, 13.55, 75.40, 13.85]` (West, South, East, North)
- **Projected CRS:** `EPSG:32643` (UTM Zone 43N)
- **Key Corridors:** Agumbe Rainforest Foothills, Tunga River Basin, Kundadri Reserve Forest

---

## Tech Stack

- **Python:** 3.11+
- **Spatial Database:** PostgreSQL 16 with PostGIS 3.4 (`postgis/postgis:16-3.4`) via Docker
- **STAC & Imagery Ingestion:** `pystac-client`, `planetary-computer`, `odc-stac` (HTTP byte-range windowed reads)
- **Multidimensional Processing:** `xarray`, `rioxarray`, `dask`, `rasterio`
- **Vector & Database ORM:** `shapely`, `geopandas`, `sqlalchemy>=2.0`, `geoalchemy2`, `psycopg2-binary`
- **Visualization:** `matplotlib`

---

## Repository Structure

```text
Haddu/
├── .env.example              # Template environment variables
├── .gitignore                # GIS, Python, credentials, and data exclusions
├── docker-compose.yml        # PostGIS 16 container definition
├── pyproject.toml            # Python package definition and dependencies
├── requirements.txt          # Direct dependency lockfile
├── README.md                 # Project documentation & quickstart
├── data/                     # Local GeoJSONs, shapefiles (gitignored)
├── haddu/
│   ├── __init__.py           # Package version and metadata
│   ├── core/                 # STAC catalog query, windowed loader, transforms
│   │   └── __init__.py
│   ├── db/                   # SQLAlchemy models, sessions, spatial migrations
│   │   ├── __init__.py
│   │   ├── models.py         # AOI and SatellitePass PostGIS models
│   │   └── session.py        # Connection engine, sessionmaker, and init_db
│   └── scripts/              # CLI scripts and entrypoints
│       ├── __init__.py
│       └── seed_aoi.py       # Migrations and AOI seeding script
└── tests/
    ├── __init__.py
    ├── test_db_models.py     # Database models, geometry, and seeding tests
    └── test_scaffolding.py   # Basic scaffolding and structure test suite
```

---

## Quickstart

### 1. Prerequisites
- [Python 3.11+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose

### 2. Clone and Setup Environment
```bash
git clone https://github.com/NachikethaKG/Haddu.git
cd Haddu

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Package and Dependencies
Install Haddu in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

### 4. Launch PostGIS Spatial Database
Spin up the containerized PostgreSQL 16 / PostGIS 3.4 database:
```bash
docker compose up -d
```

Check container status and health:
```bash
docker compose ps
```

The database is exposed on `localhost:5432`:
- **Database:** `haddu_db`
- **Username:** `haddu`
- **Password:** `haddu_secret`
- **Connection URI:** `postgresql+psycopg2://haddu:haddu_secret@localhost:5432/haddu_db`

### 5. Seed Spatial AOIs & Apply Migrations
Apply database DDL migrations and seed the Thirthahalli bounding box and reserve forest corridors:
```bash
python -m haddu.scripts.seed_aoi
```
This registers the AOIs and executes a verification query calculating the polygon area in meters via `ST_Transform(geom, 32643)`:
```text
PostGIS Verification Query Successful:
  AOI Name        : Thirthahalli Taluk
  Category        : Taluk
  Target CRS      : EPSG:32643 (UTM Zone 43N)
  Projected Area  : 1,255,640,731.29 m² (~1,255.64 km²)
```

### 6. Run Tests
Verify repository installation, models, and structure:
```bash
pytest
```

---

## Roadmap

- **Phase 1: Spatial Setup & Cloud-Native Ingestion** (Issues [#1](https://github.com/NachikethaKG/Haddu/issues/1) - [#4](https://github.com/NachikethaKG/Haddu/issues/4))
  - Repository scaffolding, Python packaging, Docker PostGIS setup
  - AOI registration and spatial tables
  - Windowed Sentinel-2 STAC reader via Microsoft Planetary Computer
  - End-to-end extraction CLI and true-color cloud-masked composite generation
- **Phase 2: Spectral Indices & Baseline Modeling**
  - NDVI, NDRE, NBR, and bi-temporal difference engines
- **Phase 3: Automated Encroachment Detection & Alerting**
  - Deforestation polygon vectorization and notification pipelines

---

## License

This project is licensed under the [MIT License](LICENSE).
