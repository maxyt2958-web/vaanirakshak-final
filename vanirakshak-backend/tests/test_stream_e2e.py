"""End-to-end check of the WebSocket contract in docs/PROTOCOL.md.

Replays what the Next.js dashboard does in `frontend/app/page.tsx`:
open the socket, send the JSON text handshake, then stream int16 PCM and
read verdicts.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

# Allow direct standalone execution via `python tests/test_stream_e2e.py`
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from fastapi.testclient import TestClient

from vanirakshak.server.app import app

SAMPLE_RATE = 16000

# Keys the dashboard's normalizeResult() reads. If any is missing the UI
# silently shows zeros, which is exactly the class of bug this test catches.
REQUIRED_TOP_LEVEL = {
    "risk_score",
    "p_spoof",
    "tier",
    "metrics",
    "interlock_active",
}
REQUIRED_METRICS = {"asv_consistency", "snr_db"}


def pcm_chunk(seconds: float = 1.5, freq: float = 220.0) -> bytes:
    """A quiet sine tone as mono int16 PCM, like the AudioWorklet produces."""
    frames = int(seconds * SAMPLE_RATE)
    return b"".join(
        struct.pack(
            "<h",
            int(8000 * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))),
        )
        for i in range(frames)
    )


def test_websocket_stream_e2e() -> None:
    client = TestClient(app)
    handshake = {
        "session_id": "e2e-protocol-check",
        "sample_rate": SAMPLE_RATE,
        "caller_id": "",
        "claimed_identity": "",
        "transaction_value_inr": 0,
        "origin_country": "IN",
        "prior_fraud_score": 0,
    }

    with client.websocket_connect("/ws/stream") as ws:
        # 1. Text handshake FIRST — the server closes with 1003 otherwise.
        ws.send_text(json.dumps(handshake))

        # 2. Stream audio: 3 chunks (each 1.5s @ 16kHz = 4.5s > 4.04s window)
        chunk = pcm_chunk(1.5)
        for _ in range(3):
            ws.send_bytes(chunk)

        verdict = ws.receive_json()

    assert verdict is not None, "Server did not return a verdict after 4.5s of audio"
    missing_top = REQUIRED_TOP_LEVEL - verdict.keys()
    assert not missing_top, f"Missing top-level keys: {sorted(missing_top)}"

    metrics = verdict.get("metrics") or {}
    missing_metrics = REQUIRED_METRICS - metrics.keys()
    assert not missing_metrics, f"Missing metrics keys: {sorted(missing_metrics)}"

    assert verdict.get("tier") in {"ALLOW", "CHALLENGE", "BLOCK"}, (
        f"Invalid tier: {verdict.get('tier')!r}"
    )
    assert isinstance(verdict.get("risk_score"), (int, float))
    assert isinstance(verdict.get("p_spoof"), (int, float))
    assert isinstance(verdict.get("interlock_active"), bool)
    assert "receipt_hash" in verdict, "receipt_hash missing from verdict"
    assert "timestamp_ms" in verdict, "timestamp_ms missing from verdict"

    # Protocol enforcement check: binary frame before handshake must be closed with 1003
    with client.websocket_connect("/ws/stream") as ws_bad:
        ws_bad.send_bytes(chunk)
        close_msg = ws_bad.receive()
        assert close_msg.get("type") == "websocket.close" and close_msg.get("code") == 1003


def main() -> int:
    try:
        test_websocket_stream_e2e()
        print("PASS: test_websocket_stream_e2e succeeded")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
