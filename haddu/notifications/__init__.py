"""Automated notifications and alerting modules for Haddu."""

from haddu.notifications.telegram import (
    dispatch_incident_alert,
    dispatch_incident_alert_sync,
    format_incident_alert_message,
)

__all__ = [
    "dispatch_incident_alert",
    "dispatch_incident_alert_sync",
    "format_incident_alert_message",
]
