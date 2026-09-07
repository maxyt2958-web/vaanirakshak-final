"""Python SDK for the VaniRakshak API.

Target: 5 lines to get going (see ``examples/sdk_demo.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx


@dataclass
class Verdict:
    risk: float
    tier: str
    p_spoof: float
    receipt_hash: str
    trigger_challenge: bool
    interlock_active: bool
    metrics: dict
    challenge: Optional[dict] = None

    @property
    def is_blocked(self) -> bool:
        return self.tier == "BLOCK"

    @property
    def is_suspicious(self) -> bool:
        return self.tier == "CHALLENGE"

    @property
    def reason(self) -> str:
        if self.challenge:
            return f"Challenge issued: {self.challenge['phrase']}"
        if self.is_blocked:
            m = self.metrics
            return (
                f"BLOCKED: cm_score={m.get('cm_score')} "
                f"asv={m.get('asv_consistency')} "
                f"chan_anom={m.get('channel_anomaly')} "
                f"prosody={m.get('prosody_unnatural')}"
            )
        return f"{self.tier}: risk={self.risk}"


class VaniRakshakClient:
    """Synchronous HTTP client + WebSocket streaming helper.

    Usage::

        from vanirakshak.sdk import VaniRakshakClient
        c = VaniRakshakClient(host="http://localhost:8000")
        c.enroll("ayush", "consent.wav")
        for v in c.stream_chunk(...): ...
    """

    def __init__(self, host: str = "http://localhost:8000", api_key: str = "demo") -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self._http = httpx.Client(base_url=self.host, timeout=30.0)

    def health(self) -> dict:
        return self._http.get("/v1/health").json()

    def enroll(self, user_id: str, audio_path: str) -> dict:
        with open(audio_path, "rb") as f:
            r = self._http.post(
                "/v1/enroll",
                data={"user_id": user_id},
                files={"audio": (audio_path, f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()

    def analyze(self, audio_path: str, caller_id: str = "+91-0000000000", claimed_identity: str = "") -> dict:
        with open(audio_path, "rb") as f:
            r = self._http.post(
                "/v1/analyze",
                data={"caller_id": caller_id, "claimed_identity": claimed_identity},
                files={"audio": (audio_path, f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()

    def tail_audit(self, n: int = 50) -> dict:
        return self._http.get("/v1/audit", params={"n": n}).json()

    def stream(
        self,
        caller_id: str = "+91-0000000000",
        claimed_user: str = "",
        sample_rate: int = 16000,
        transaction_value_inr: float = 0.0,
    ) -> "StreamClient":
        return StreamClient(
            host=self.host,
            caller_id=caller_id,
            claimed_user=claimed_user,
            sample_rate=sample_rate,
            transaction_value_inr=transaction_value_inr,
        )


class StreamClient:
    """Sync WebSocket stream to ``/v1/stream``.

    Uses ``websockets.sync.client``. If the lib is missing we surface
    a clear error on first use.
    """

    def __init__(self, host: str, caller_id: str, claimed_user: str, sample_rate: int, transaction_value_inr: float) -> None:
        self._ws_url = host.replace("http://", "ws://").replace("https://", "wss://") + "/v1/stream"
        self._handshake = {
            "session_id": "sdk-stream",
            "caller_id": caller_id,
            "claimed_identity": claimed_user,
            "sample_rate": sample_rate,
            "transaction_value_inr": transaction_value_inr,
        }
        self._ws = None

    def __enter__(self) -> "StreamClient":
        try:
            from websockets.sync.client import connect  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "WebSocket streaming requires the `websockets` package. "
                "Install with: pip install websockets"
            ) from e
        self._ws = connect(self._ws_url, max_size=2 ** 22)
        self._ws.send(json.dumps(self._handshake))
        return self

    def __exit__(self, *exc) -> None:
        if self._ws is not None:
            self._ws.close()

    def send_chunk(self, pcm_int16_bytes: bytes) -> Optional[Verdict]:
        if self._ws is None:
            raise RuntimeError("use `with client.stream(...) as s:` first")
        self._ws.send(pcm_int16_bytes)
        try:
            raw = self._ws.recv(timeout=5.0)
        except Exception:
            return None
        d = json.loads(raw)
        return Verdict(
            risk=d.get("risk_score", 0.0),
            tier=d.get("tier", "ALLOW"),
            p_spoof=d.get("p_spoof", 0.0),
            receipt_hash=d.get("receipt_hash", ""),
            trigger_challenge=d.get("trigger_challenge", False),
            interlock_active=d.get("interlock_active", False),
            metrics=d.get("metrics", {}),
            challenge=d.get("challenge"),
        )
