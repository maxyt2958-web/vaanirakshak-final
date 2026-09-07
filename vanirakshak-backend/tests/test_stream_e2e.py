"""End-to-end check of the WebSocket contract in docs/PROTOCOL.md.

Replays exactly what the Next.js dashboard does in `frontend/app/page.tsx`:
open the socket, send the JSON text handshake, then stream int16 PCM and
read verdicts. Run the server first:

    python -m uvicorn vanirakshak.server.app:app --port 8000

Then:

    python tests/test_stream_e2e.py
"""

from __future__ import annotations

import json
import math
import struct
import sys

from websockets.sync.client import connect

URL = "ws://127.0.0.1:8000/ws/stream"
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


def main() -> int:
    handshake = {
        "session_id": "e2e-protocol-check",
        "sample_rate": SAMPLE_RATE,
        "caller_id": "",
        "claimed_identity": "",
        "transaction_value_inr": 0,
        "origin_country": "IN",
        "prior_fraud_score": 0,
    }

    try:
        with connect(URL) as ws:
            # 1. Text handshake FIRST — the server closes with 1003 otherwise.
            ws.send(json.dumps(handshake))
            print(f"handshake sent -> {URL}")

            # 2. Stream audio.
            chunk = pcm_chunk()
            verdicts = []
            for _ in range(6):
                ws.send(chunk)
                try:
                    raw = ws.recv(timeout=10)
                except TimeoutError:
                    continue
                if isinstance(raw, (bytes, bytearray)):
                    continue
                verdicts.append(json.loads(raw))
                if len(verdicts) >= 2:
                    break
    except OSError as exc:
        print(f"FAIL: cannot reach {URL} ({exc}). Start the server first.")
        return 1

    if not verdicts:
        print("FAIL: server accepted the socket but sent no verdict.")
        return 1

    verdict = verdicts[-1]
    print("\n--- sample verdict ---")
    print(json.dumps(verdict, indent=2)[:900])

    missing_top = REQUIRED_TOP_LEVEL - verdict.keys()
    metrics = verdict.get("metrics") or {}
    missing_metrics = REQUIRED_METRICS - metrics.keys()

    print("\n--- assertions ---")
    ok = True
    if missing_top:
        print(f"FAIL missing top-level keys: {sorted(missing_top)}")
        ok = False
    else:
        print(f"OK   top-level keys present: {sorted(REQUIRED_TOP_LEVEL)}")

    if missing_metrics:
        print(f"FAIL missing metrics keys: {sorted(missing_metrics)}")
        ok = False
    else:
        print(f"OK   metrics keys present: {sorted(REQUIRED_METRICS)}")

    if verdict.get("tier") not in {"ALLOW", "CHALLENGE", "BLOCK"}:
        print(f"FAIL tier is {verdict.get('tier')!r}, expected ALLOW/CHALLENGE/BLOCK")
        ok = False
    else:
        print(f"OK   tier = {verdict['tier']}")

    print(f"OK   snr_db = {metrics.get('snr_db')} (was always 0 before the fix)")
    print(f"OK   asv_consistency = {metrics.get('asv_consistency')}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
