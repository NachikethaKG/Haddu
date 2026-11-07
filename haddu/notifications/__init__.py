"""Automated notifications and alerting modules for Haddu."""

from haddu.notifications.telegram import (
    dispatch_incident_alert,
    dispatch_incident_alert_sync,
    format_incident_alert_message,
    get_telegram_config,
)

__all__ = [
    "dispatch_incident_alert",
    "dispatch_incident_alert_sync",
    "format_incident_alert_message",
    "get_telegram_config",
]
