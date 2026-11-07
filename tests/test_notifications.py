"""Unit tests for automated Telegram notification dispatcher and alert formatting."""

import asyncio
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

from haddu.db.models import EncroachmentIncident
from haddu.notifications.telegram import (
    dispatch_incident_alert,
    dispatch_incident_alert_sync,
    format_incident_alert_message,
    get_telegram_config,
)


@pytest.fixture
def critical_incident_dict() -> dict:
    """Fixture providing a sample CRITICAL incident dictionary."""
    poly = Polygon([(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)])
    return {
        "id": str(uuid.uuid4()),
        "pre_date": "2026-06-28",
        "post_date": "2026-08-15",
        "area_sqm": 40468.56,
        "area_acres": 10.0,
        "centroid_lat": 13.5824,
        "centroid_lon": 75.0912,
        "violation_tier": "CRITICAL",
        "encroached_rf_name": "Agumbe Rainforest Reserve Forest",
        "rf_area_acres": 10.0,
        "riparian_breach": False,
        "sensor": "s1",
        "geom": poly,
    }


@pytest.fixture
def high_incident_model() -> EncroachmentIncident:
    """Fixture providing a sample HIGH incident ORM model instance."""
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
        violation_tier="HIGH",
        encroached_rf_name=None,
        rf_area_acres=0.0,
        riparian_breach=True,
        sensor="s2",
        geom=geom,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def moderate_incident_dict() -> dict:
    """Fixture providing a sample MODERATE incident dictionary."""
    return {
        "id": str(uuid.uuid4()),
        "pre_date": "2026-01-10",
        "post_date": "2026-03-15",
        "area_sqm": 4046.86,
        "area_acres": 1.0,
        "centroid_lat": 13.7000,
        "centroid_lon": 75.2500,
        "violation_tier": "MODERATE",
        "encroached_rf_name": None,
        "rf_area_acres": 0.0,
        "riparian_breach": False,
        "sensor": "s2",
    }


def test_format_incident_alert_message_critical(critical_incident_dict):
    """Verify HTML message formatting for CRITICAL Reserve Forest breach incident."""
    msg = format_incident_alert_message(critical_incident_dict)

    assert "🚨 <b>HADDU ALERT: Canopy Disturbance Detected</b>" in msg
    assert "CRITICAL - Reserve Forest Breach" in msg
    assert "Agumbe Rainforest Reserve Forest" in msg
    assert "https://www.google.com/maps?q=13.58240,75.09120" in msg
    assert "10.00 acres" in msg
    assert "400.0 guntas" in msg
    assert "2026-08-15" in msg
    assert "S1" in msg


def test_format_incident_alert_message_high(high_incident_model):
    """Verify HTML message formatting for HIGH Riparian buffer incident."""
    msg = format_incident_alert_message(high_incident_model)

    assert "🚨 <b>HADDU ALERT: Canopy Disturbance Detected</b>" in msg
    assert "HIGH - Riparian Buffer Breach" in msg
    assert "Tunga River Riparian Corridor" in msg
    assert "https://www.google.com/maps?q=13.69380,75.24420" in msg
    assert "3.00 acres" in msg
    assert "120.0 guntas" in msg
    assert "2026-08-15" in msg
    assert "S2" in msg


def test_get_telegram_config(monkeypatch):
    """Verify environment variable retrieval for Telegram credentials."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-DEF")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100987654321")

    token, chat_id = get_telegram_config()
    assert token == "123456:ABC-DEF"
    assert chat_id == "-100987654321"


def test_dispatch_incident_alert_photo_success(critical_incident_dict, tmp_path: Path):
    """Verify asynchronous dispatch sending a photo alert to Telegram Bot API."""
    test_img = tmp_path / "test_thumb.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 42}}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp

    res = asyncio.run(
        dispatch_incident_alert(
            incident_data=critical_incident_dict,
            preview_image_path=test_img,
            bot_token="test_bot_token",
            chat_id="test_chat_id",
            client=mock_client,
        )
    )

    assert res == {"ok": True, "result": {"message_id": 42}}
    assert mock_client.post.called
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://api.telegram.org/bottest_bot_token/sendPhoto"
    assert call_args[1]["data"]["chat_id"] == "test_chat_id"
    assert "🚨 <b>HADDU ALERT" in call_args[1]["data"]["caption"]
    assert "photo" in call_args[1]["files"]


def test_dispatch_incident_alert_text_fallback(high_incident_model):
    """Verify asynchronous dispatch falling back to sendMessage when no preview image exists."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 43}}

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp

    with patch("haddu.notifications.telegram.generate_dossier_visuals", side_effect=ValueError("Disabled")):
        res = asyncio.run(
            dispatch_incident_alert(
                incident_data=high_incident_model,
                preview_image_path=None,
                bot_token="test_bot_token",
                chat_id="test_chat_id",
                client=mock_client,
            )
        )

    assert res == {"ok": True, "result": {"message_id": 43}}
    assert mock_client.post.called
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://api.telegram.org/bottest_bot_token/sendMessage"
    assert call_args[1]["json"]["chat_id"] == "test_chat_id"
    assert "HIGH - Riparian Buffer Breach" in call_args[1]["json"]["text"]


def test_dispatch_incident_alert_skips_moderate(moderate_incident_dict):
    """Verify non-critical incidents (MODERATE/LOW) are skipped unless forced."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    res = asyncio.run(
        dispatch_incident_alert(
            incident_data=moderate_incident_dict,
            bot_token="test_token",
            chat_id="test_chat",
            client=mock_client,
            force=False,
        )
    )

    assert res is None
    assert not mock_client.post.called

    # When forced, it should dispatch
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    mock_client.post.return_value = mock_resp

    res_forced = asyncio.run(
        dispatch_incident_alert(
            incident_data=moderate_incident_dict,
            bot_token="test_token",
            chat_id="test_chat",
            client=mock_client,
            force=True,
        )
    )
    assert res_forced == {"ok": True}
    assert mock_client.post.called


def test_dispatch_incident_alert_missing_credentials(critical_incident_dict, monkeypatch):
    """Verify dispatcher returns None gracefully without error if Telegram env vars are missing."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    res = asyncio.run(
        dispatch_incident_alert(
            incident_data=critical_incident_dict,
            client=mock_client,
        )
    )

    assert res is None
    assert not mock_client.post.called


def test_dispatch_incident_alert_api_error(critical_incident_dict):
    """Verify error dictionary returned when Telegram API returns non-200 status."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 400
    mock_resp.text = '{"ok":false,"description":"Bad Request: chat not found"}'

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp

    with patch("haddu.notifications.telegram.generate_dossier_visuals", side_effect=ValueError("Disabled")):
        res = asyncio.run(
            dispatch_incident_alert(
                incident_data=critical_incident_dict,
                bot_token="invalid_token",
                chat_id="invalid_chat",
                client=mock_client,
            )
        )

    assert res is not None
    assert res["error"] == "HTTP 400"
    assert "chat not found" in res["detail"]


def test_dispatch_incident_alert_sync(critical_incident_dict):
    """Verify synchronous wrapper dispatch_incident_alert_sync works seamlessly."""
    with patch("haddu.notifications.telegram.dispatch_incident_alert", new_callable=AsyncMock) as mock_async_dispatch:
        mock_async_dispatch.return_value = {"ok": True, "result": {"message_id": 99}}

        res = dispatch_incident_alert_sync(
            incident_data=critical_incident_dict,
            bot_token="test_token",
            chat_id="test_chat",
        )

        assert res == {"ok": True, "result": {"message_id": 99}}
        assert mock_async_dispatch.called
