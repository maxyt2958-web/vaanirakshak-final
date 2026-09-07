"""DPDP Act 2023 compliant feature-only audit log.

We log:
  * session_id, caller_id, claimed_identity
  * timestamp (UTC ISO 8601)
  * per-window SCALAR features (no raw audio, no invertible embeddings)
  * risk score, tier, and a SHA-256 "receipt hash" that chains each
    event to the previous one for tamper evidence.

The actual log lives in an in-memory ring by default. The 73-hour
hackathon prototype keeps the last ``max_events`` events and exposes
``GET /v1/audit`` to the dashboard.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional


@dataclass
class AuditEvent:
    session_id: str
    caller_id: str
    claimed_identity: str
    timestamp: float
    risk_score: float
    tier: str
    features: Dict[str, float]
    receipt_hash: str = ""
    prev_hash: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return d


class AuditLog:
    """Append-only, hash-chained audit log.

    Each event's ``receipt_hash`` is ``SHA-256(prev_hash || canonical_json(event_without_hashes))``.
    The chain is verifiable offline: any tampering breaks the chain.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: Deque[AuditEvent] = deque(maxlen=max_events)

    @property
    def _last_hash(self) -> str:
        return self._events[-1].receipt_hash if self._events else ""

    def append(
        self,
        *,
        session_id: str,
        caller_id: str,
        claimed_identity: str,
        risk_score: float,
        tier: str,
        features: Dict[str, float],
    ) -> AuditEvent:
        prev = self._last_hash
        payload = {
            "session_id": session_id,
            "caller_id": caller_id,
            "claimed_identity": claimed_identity,
            "ts": time.time(),
            "risk": round(float(risk_score), 4),
            "tier": tier,
            "features": {k: round(float(v), 6) for k, v in features.items()},
            "prev": prev,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        receipt = hashlib.sha256(canonical).hexdigest()
        ev = AuditEvent(
            session_id=session_id,
            caller_id=caller_id,
            claimed_identity=claimed_identity,
            timestamp=payload["ts"],
            risk_score=round(float(risk_score), 4),
            tier=tier,
            features=payload["features"],
            receipt_hash=receipt,
            prev_hash=prev,
        )
        self._events.append(ev)
        return ev

    def tail(self, n: int = 50) -> List[AuditEvent]:
        return list(self._events)[-n:]

    def verify_chain(self) -> bool:
        """Re-compute the hash chain and return True iff no tampering occurred."""
        prev = ""
        for ev in self._events:
            payload = {
                "session_id": ev.session_id,
                "caller_id": ev.caller_id,
                "claimed_identity": ev.claimed_identity,
                "ts": ev.timestamp,
                "risk": ev.risk_score,
                "tier": ev.tier,
                "features": {k: round(float(v), 6) for k, v in ev.features.items()},
                "prev": prev,
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected = hashlib.sha256(canonical).hexdigest()
            if expected != ev.receipt_hash or ev.prev_hash != prev:
                return False
            prev = ev.receipt_hash
        return True
