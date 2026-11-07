"""Automated Telegram alert dispatcher for high-severity encroachment incidents.

Sends instantaneous incident cards with comparative satellite imagery and
direct Google Maps navigation links to Telegram channels/chats when CRITICAL
or HIGH violations occur.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

import httpx

try:
    from haddu.reporting import generate_dossier_visuals
except ImportError:
    generate_dossier_visuals = None

logger = logging.getLogger(__name__)

# Default severity levels that warrant immediate push notifications
DEFAULT_ALERTABLE_TIERS: set[str] = {"CRITICAL", "HIGH"}


def get_telegram_config() -> tuple[str | None, str | None]:
    """Retrieve Telegram credentials from environment variables."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    return bot_token, chat_id


def format_incident_alert_message(incident_data: dict[str, Any] | Any) -> str:
    """Format an incident into a structured Telegram HTML notification card.

    Includes violation severity badge, precise coordinates, Google Maps link,
    Reserve Forest / Riparian zone breach info, area in acres and guntas, and date.
    """
    if isinstance(incident_data, dict):
        lat = float(incident_data.get("centroid_lat", 13.6938))
        lon = float(incident_data.get("centroid_lon", 75.2442))
        area_sqm = float(incident_data.get("area_sqm", 4046.86))
        area_acres = float(incident_data.get("area_acres", area_sqm / 4046.86))
        tier = str(incident_data.get("violation_tier", "MODERATE")).upper()
        rf_name = incident_data.get("encroached_rf_name")
        post_date = str(incident_data.get("post_date", "N/A"))
        riparian = bool(incident_data.get("riparian_breach", False))
        sensor = str(incident_data.get("sensor", "s2"))
    else:
        lat = float(getattr(incident_data, "centroid_lat", 13.6938))
        lon = float(getattr(incident_data, "centroid_lon", 75.2442))
        area_sqm = float(getattr(incident_data, "area_sqm", 4046.86))
        area_acres = float(getattr(incident_data, "area_acres", area_sqm / 4046.86))
        tier = str(getattr(incident_data, "violation_tier", "MODERATE")).upper()
        rf_name = getattr(incident_data, "encroached_rf_name", None)
        post_date = str(getattr(incident_data, "post_date", "N/A"))
        riparian = bool(getattr(incident_data, "riparian_breach", False))
        sensor = str(getattr(incident_data, "sensor", "s2"))

    # 1 gunta = 101.17 m²
    area_guntas = area_sqm / 101.17

    if tier == "CRITICAL":
        tier_str = "CRITICAL - Reserve Forest Breach"
        tier_emoji = "🔴"
    elif tier == "HIGH":
        tier_str = "HIGH - Riparian Buffer Breach" if riparian else "HIGH - Statutory Corridor Breach"
        tier_emoji = "🟠"
    else:
        tier_str = f"{tier} - Disturbance"
        tier_emoji = "🟡"

    affected_forest = rf_name or ("Tunga River Riparian Corridor" if riparian else "State Forest Compartment")
    maps_url = f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"

    lines = [
        "🚨 <b>HADDU ALERT: Canopy Disturbance Detected</b>",
        "",
        f"<b>Violation Tier:</b> {tier_emoji} <b>{tier_str}</b>",
        f"<b>Location:</b> {lat:.5f}, {lon:.5f}",
        f"<b>Google Maps:</b> <a href=\"{maps_url}\">{maps_url}</a>",
        f"<b>Affected Forest:</b> {affected_forest}",
        f"<b>Area:</b> {area_acres:.2f} acres ({area_guntas:.1f} guntas)",
        f"<b>Detection Date:</b> {post_date}",
        f"<b>Sensor:</b> {sensor.upper()}",
    ]
    return "\n".join(lines)


async def dispatch_incident_alert(
    incident_data: dict[str, Any] | Any,
    preview_image_path: str | Path | None = None,
    bot_token: str | None = None,
    chat_id: str | None = None,
    client: httpx.AsyncClient | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Asynchronously dispatch an incident alert card to a Telegram channel/chat.

    Args:
        incident_data: Dictionary or ORM model representing the violation incident.
        preview_image_path: Optional file path to a comparative imagery snapshot.
        bot_token: Telegram Bot Token (defaults to TELEGRAM_BOT_TOKEN env var).
        chat_id: Telegram Chat/Channel ID (defaults to TELEGRAM_CHAT_ID env var).
        client: Optional existing httpx.AsyncClient instance.
        force: If True, dispatches alert even for non-CRITICAL/HIGH violation tiers.

    Returns:
        Telegram API response JSON dict on success, or None if skipped/failed.
    """
    if isinstance(incident_data, dict):
        tier = str(incident_data.get("violation_tier", "MODERATE")).upper()
        inc_id = str(incident_data.get("id", "N/A"))
    else:
        tier = str(getattr(incident_data, "violation_tier", "MODERATE")).upper()
        inc_id = str(getattr(incident_data, "id", "N/A"))

    # Only alert on high-severity violations unless forced
    if not force and tier not in DEFAULT_ALERTABLE_TIERS:
        logger.info(
            "Incident %s tier '%s' below notification threshold (%s). Skipping Telegram dispatch.",
            inc_id,
            tier,
            DEFAULT_ALERTABLE_TIERS,
        )
        return None

    # Resolve credentials
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not target_chat:
        logger.warning(
            "Telegram credentials not configured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing). "
            "Skipping alert dispatch for incident %s.",
            inc_id,
        )
        return None

    # Prepare message text
    message_text = format_incident_alert_message(incident_data)

    # Prepare visual preview image bytes if available or generate one
    image_bytes: bytes | None = None
    if preview_image_path is not None:
        p = Path(preview_image_path)
        if p.exists():
            image_bytes = p.read_bytes()
        else:
            logger.warning("Preview image path does not exist: %s", preview_image_path)

    if image_bytes is None and generate_dossier_visuals is not None:
        try:
            data_uri = generate_dossier_visuals(incident_data)
            if data_uri and data_uri.startswith("data:image/png;base64,"):
                b64_str = data_uri.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_str)
        except (ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("Automatic visual snapshot generation omitted: %s", e)

    # HTTP dispatch logic
    async def _send(http_client: httpx.AsyncClient) -> dict[str, Any] | None:
        if image_bytes is not None:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            files = {"photo": ("disturbance_preview.png", image_bytes, "image/png")}
            data = {
                "chat_id": str(target_chat),
                "caption": message_text,
                "parse_mode": "HTML",
            }
            logger.info("Sending photo alert to Telegram chat %s...", target_chat)
            resp = await http_client.post(url, data=data, files=files, timeout=15.0)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            json_body = {
                "chat_id": str(target_chat),
                "text": message_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            }
            logger.info("Sending text alert to Telegram chat %s...", target_chat)
            resp = await http_client.post(url, json=json_body, timeout=15.0)

        if resp.status_code == 200:
            logger.info("Successfully dispatched Telegram alert for incident %s", inc_id)
            return resp.json()

        logger.error(
            "Telegram API returned status %s for incident %s: %s",
            resp.status_code,
            inc_id,
            resp.text,
        )
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text}

    try:
        if client is not None:
            return await _send(client)
        async with httpx.AsyncClient() as new_client:
            return await _send(new_client)
    except Exception:
        logger.exception("Exception occurred during Telegram alert dispatch.")
        return None


def dispatch_incident_alert_sync(
    incident_data: dict[str, Any] | Any,
    preview_image_path: str | Path | None = None,
    bot_token: str | None = None,
    chat_id: str | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Synchronous convenience wrapper for dispatch_incident_alert."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Schedule in existing event loop
        task = loop.create_task(
            dispatch_incident_alert(
                incident_data=incident_data,
                preview_image_path=preview_image_path,
                bot_token=bot_token,
                chat_id=chat_id,
                force=force,
            )
        )
        return {"scheduled": True, "task": str(task)}

    return asyncio.run(
        dispatch_incident_alert(
            incident_data=incident_data,
            preview_image_path=preview_image_path,
            bot_token=bot_token,
            chat_id=chat_id,
            force=force,
        )
    )
