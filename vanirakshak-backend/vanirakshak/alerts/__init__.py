"""Outbound alerting for high-risk call events."""

from .dispatcher import AlertDispatcher, build_alert_payload

__all__ = ["AlertDispatcher", "build_alert_payload"]
