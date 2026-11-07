"""Unit tests for automated legal evidence dossier generator and WeasyPrint PDF compilation."""

import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import weasyprint
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

from haddu.db.models import EncroachmentIncident
from haddu.reporting import (
    compile_dossier_pdf,
    find_nearest_village,
    generate_dossier_visuals,
    render_dossier_html,
)


@pytest.fixture
def sample_incident_dict() -> dict:
    """Fixture providing a sample dictionary incident representing a Critical RF breach."""
    poly = Polygon([(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)])
    return {
        "id": str(uuid.uuid4()),
        "pre_date": "2026-06-28",
        "post_date": "2026-08-15",
        "area_sqm": 40468.56,
        "area_acres": 10.0,
        "centroid_lat": 13.5824,
        "centroid_lon": 75.0912,
        "mean_delta_ndvi": 0.45,
        "max_delta_ndvi": 0.65,
        "violation_tier": "CRITICAL",
        "encroached_rf_name": "Agumbe Rainforest Reserve Forest",
        "rf_area_acres": 10.0,
        "riparian_breach": False,
        "sensor": "s1",
        "confidence_score": 0.985,
        "geom": poly,
    }


@pytest.fixture
def sample_incident_model() -> EncroachmentIncident:
    """Fixture providing an EncroachmentIncident ORM instance representing a High riparian breach."""
    poly = Polygon([(75.24, 13.68), (75.26, 13.68), (75.26, 13.70), (75.24, 13.70), (75.24, 13.68)])
    geom = from_shape(poly, srid=4326)
    return EncroachmentIncident(
        id=uuid.uuid4(),
        pre_date=date(2026, 6, 28),
        post_date=date(2026, 8, 15),
        area_sqm=12140.57,
        area_acres=3.0,
        centroid_lat=13.6938,
        centroid_lon=75.2442,
        mean_delta_ndvi=0.36,
        max_delta_ndvi=0.52,
        violation_tier="HIGH",
        encroached_rf_name=None,
        rf_area_acres=0.0,
        riparian_breach=True,
        sensor="s2",
        confidence_score=0.942,
        geom=geom,
        created_at=datetime.now(UTC),
    )


def test_find_nearest_village():
    """Verify village proximity resolution for Thirthahalli Taluk coordinates."""
    # Near Agumbe
    res_agumbe = find_nearest_village(13.51, 75.09)
    assert "Agumbe" in res_agumbe

    # Near Thirthahalli Town
    res_thirthahalli = find_nearest_village(13.69, 75.24)
    assert "Thirthahalli Town" in res_thirthahalli


def test_generate_dossier_visuals(sample_incident_dict):
    """Verify side-by-side comparative imagery generation as base64 data URI."""
    uri = generate_dossier_visuals(sample_incident_dict)
    assert isinstance(uri, str)
    assert uri.startswith("data:image/png;base64,")
    # Base64 string must have meaningful size
    assert len(uri) > 2000


def test_render_dossier_html(sample_incident_dict):
    """Verify that Jinja2 HTML rendering contains all required administrative and legal items."""
    html = render_dossier_html(sample_incident_dict)

    # 1. Official Header Addressees
    assert "Office of the Range Forest Officer (RFO), Thirthahalli Range, Shivamogga Circle" in html
    assert "Deputy Conservator of Forests (DCF), Shivamogga Division" in html

    # 2. Legal Statute Citation
    assert "Karnataka Forest Act, 1963" in html
    assert "Section 64A" in html

    # 3. Spatial Impact Metrics
    assert "10.00 ac" in html
    # 40468.56 sqm / 101.17 = 400.0 guntas
    assert "400.0 guntas" in html
    assert "40,468.6 m²" in html
    assert "Agumbe Rainforest Reserve Forest" in html

    # 4. Sensor & Confidence
    assert "Sentinel-1 SAR" in html
    assert "98.5%" in html

    # 5. Image & Verification Hash
    assert '<img class="visual-img" src="data:image/png;base64,' in html
    assert "Hash:" in html


def test_compile_dossier_pdf_dict_single_page(sample_incident_dict, tmp_path: Path):
    """Verify PDF compilation from dictionary data produces a single-page document."""
    out_file = tmp_path / "test_dict_dossier.pdf"
    pdf_bytes = compile_dossier_pdf(sample_incident_dict, output_path=out_file)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert out_file.exists()
    assert out_file.stat().st_size == len(pdf_bytes)

    # Validate single page layout using WeasyPrint document
    html = render_dossier_html(sample_incident_dict)
    doc = weasyprint.HTML(string=html).render()
    assert len(doc.pages) == 1, f"Expected exactly 1 A4 page, but got {len(doc.pages)} pages."


def test_compile_dossier_pdf_orm_model_single_page(sample_incident_model):
    """Verify PDF compilation from SQLAlchemy ORM incident model produces a single-page document."""
    pdf_bytes = compile_dossier_pdf(sample_incident_model)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 10000

    html = render_dossier_html(sample_incident_model)
    assert "HIGH" in html
    assert "50m River Corridor" in html
    assert "Sentinel-2 MSI" in html

    doc = weasyprint.HTML(string=html).render()
    assert len(doc.pages) == 1, f"Expected exactly 1 A4 page, but got {len(doc.pages)} pages."
