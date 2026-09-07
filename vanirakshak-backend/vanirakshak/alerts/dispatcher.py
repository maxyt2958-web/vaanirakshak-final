"""Best-effort outbound alerting (webhook / SMS / email gateway).

Alerts are dispatched from the **server**, never the browser: the browser
would have to hold the gateway credentials, and the interlock decision it
reports could be forged by a tampered client.

Configure with ``VR_ALERT_WEBHOOK_URL``. When unset the dispatcher is a
no-op, so the stack still runs with zero external dependencies.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HTTP_OK = 200
_HTTP_NO_CONTENT = 299


def _env_url() -> str:
    return os.environ.get("VR_ALERT_WEBHOOK_URL", "").strip()


class AlertDispatcher:
    """POSTs risk events to a configured webhook endpoint.

    Delivery is best-effort: a failure is logged and swallowed, because an
    alerting outage must never break an in-flight call analysis.
    """

    def __init__(self, webhook_url: Optional[str] = None, timeout: float = 5.0) -> None:
        self.webhook_url = (
            webhook_url if webhook_url is not None else _env_url()
        )
        self.timeout = timeout
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def dispatch(self, payload: Dict[str, Any]) -> bool:
        """Send one alert. Returns True only on an acknowledged delivery."""
        if not self.enabled:
            logger.debug("alert suppressed: VR_ALERT_WEBHOOK_URL not set")
            return False

        try:
            import httpx  # imported lazily so the dep stays optional
        except ImportError:  # pragma: no cover
            logger.warning("alert dropped: httpx not installed")
            return False

        try:
            response = httpx.post(
                self.webhook_url, json=payload, timeout=self.timeout
            )
        except Exception as exc:  # network, DNS, TLS, timeout
            logger.warning("alert delivery failed: %s", exc)
            return False

        if _HTTP_OK <= response.status_code <= _HTTP_NO_CONTENT:
            return True

        logger.warning(
            "alert endpoint returned HTTP %s", response.status_code
        )
        return False


def build_alert_payload(
    session_id: str,
    risk: float,
    tier: str,
    interlock_active: bool,
    receipt_hash: str,
    challenge_phrase: Optional[str] = None,
) -> Dict[str, Any]:
    """Shape of the alert body sent to the gateway."""
    return {
        "source": "vanirakshak",
        "session_id": session_id,
        "timestamp_ms": int(time.time() * 1000),
        "risk": round(float(risk), 2),
        "tier": tier,
        "interlock_active": bool(interlock_active),
        "receipt_hash": receipt_hash,
        "challenge_phrase": challenge_phrase,
        "message": (
            f"VaniRakshak: risk={risk:.1f} tier={tier}"
            + (" — transaction blocked" if interlock_active else "")
        ),
    }
