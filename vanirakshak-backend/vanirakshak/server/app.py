"""FastAPI HTTP/WebSocket server (R4 — the missing API surface).

Endpoints (see ``docs/07-privacy-compliance-and-apis.md``):
  * ``POST /v1/enroll``           enrol a genuine voice (10-20s)
  * ``POST /v1/analyze``          one-shot WAV/FLAC → risk score
  * ``WS   /v1/stream``           real-time PCM stream → live verdicts
  * ``POST /v1/sessions/{id}/challenge``  grade a challenge response
  * ``GET  /v1/audit``            tail the DPDP-compliant audit log
  * ``GET  /v1/health``           liveness
  * ``GET  /docs``                OpenAPI UI
"""

from __future__ import annotations

import io
import time
import uuid
import wave
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..session import CallSession, SessionContext
from ..privacy.audit import AuditLog


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VaniRakshak",
    description="Real-time voice cloning impersonation detection (R4 backend).",
    version="0.1.0-sprint1",
)

# module-level singletons for the demo. In production these would be
# replaced with a proper dependency-injected pool keyed by tenant.
SESSIONS: Dict[str, CallSession] = {}
AUDIT = AuditLog(max_events=10_000)


def _read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV (PCM) byte string to (float32 in [-1,1], sample_rate)."""
    bio = io.BytesIO(data)
    with wave.open(bio, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32)
    else:
        raise HTTPException(400, f"unsupported sample width: {sampwidth}")
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr.astype(np.float32) / 32768.0, sr


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EnrollRequest(BaseModel):
    user_id: str = Field(..., description="Stable identifier for the speaker.")


class EnrollResponse(BaseModel):
    status: str
    user_id: str
    embedding_dimension: int
    audio_retained: bool
    created_at: str


class AnalyzeResponse(BaseModel):
    status: str
    file_duration_sec: float
    calibrated_risk: float
    p_spoof: float
    tier: str
    receipt_hash: str
    breakdown: dict


class SessionStartRequest(BaseModel):
    caller_id: str = "+91-0000000000"
    claimed_identity: str = ""
    transaction_value_inr: float = 0.0
    origin_country: str = "IN"
    prior_fraud_score: float = 0.0


class ChallengeGradeRequest(BaseModel):
    response_text: str
    response_latency_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/v1/health")
def health() -> dict:
    return {"ok": True, "version": app.version, "active_sessions": len(SESSIONS)}


@app.post("/v1/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), audio: UploadFile = File(...)) -> EnrollResponse:
    data = await audio.read()
    x, sr = _read_wav_bytes(data)
    if x.size < int(0.5 * sr):
        raise HTTPException(400, "enrolment audio too short (need ≥ 0.5 s)")
    # Re-use or create a session for enrolment
    sid = f"enroll-{uuid.uuid4().hex[:8]}"
    sess = CallSession(sid, SessionContext(caller_id="enrolment", claimed_identity=user_id))
    sess.enrol(user_id, x, sr)
    SESSIONS[sid] = sess
    return EnrollResponse(
        status="enrolled",
        user_id=user_id,
        embedding_dimension=192,
        audio_retained=False,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    audio: UploadFile = File(...),
    caller_id: str = Form("+91-0000000000"),
    claimed_identity: str = Form(""),
) -> AnalyzeResponse:
    data = await audio.read()
    x, sr = _read_wav_bytes(data)
    sid = f"analyze-{uuid.uuid4().hex[:8]}"
    sess = CallSession(
        sid,
        SessionContext(caller_id=caller_id, claimed_identity=claimed_identity),
    )
    # push as a single chunk
    sess.push_chunk(x)
    # if the clip was longer than the window, drain the rest
    hop = sess.buffer.hop_samples
    i = hop
    while i + hop <= x.size:
        sess.push_chunk(x[i : i + hop])
        i += hop
    last = sess.history[-1] if sess.history else None
    if last is None:
        raise HTTPException(400, "audio too short to analyse (need ≥ window size)")
    return AnalyzeResponse(
        status="success",
        file_duration_sec=round(x.size / sr, 3),
        calibrated_risk=last.risk,
        p_spoof=last.p_spoof,
        tier=last.tier,
        receipt_hash=last.receipt_hash,
        breakdown={
            "codec_guess": last.metrics.get("codec_guess"),
            "cm_score": last.metrics.get("cm_score"),
            "asv_consistency": last.metrics.get("asv_consistency"),
            "channel_anomaly": last.metrics.get("channel_anomaly"),
            "prosody_unnatural": last.metrics.get("prosody_unnatural"),
        },
    )


@app.post("/v1/sessions")
def start_session(req: SessionStartRequest) -> dict:
    sid = f"call-{uuid.uuid4().hex[:10]}"
    SESSIONS[sid] = CallSession(
        sid,
        SessionContext(
            caller_id=req.caller_id,
            claimed_identity=req.claimed_identity,
            transaction_value_inr=req.transaction_value_inr,
            origin_country=req.origin_country,
            prior_fraud_score=req.prior_fraud_score,
        ),
    )
    return {"session_id": sid}


@app.post("/v1/sessions/{sid}/challenge")
def grade_challenge(sid: str, req: ChallengeGradeRequest) -> dict:
    sess = SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return sess.grade_challenge(req.response_text, req.response_latency_ms)


@app.get("/v1/audit")
def tail_audit(n: int = 50) -> dict:
    events = AUDIT.tail(n)
    return {
        "chain_valid": AUDIT.verify_chain(),
        "events": [e.to_public_dict() for e in events],
    }


@app.websocket("/v1/stream")
async def stream_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    # first message: JSON handshake
    try:
        handshake_raw = await ws.receive_text()
        import json
        handshake = json.loads(handshake_raw)
    except Exception:
        await ws.close(code=1003)
        return

    sid = handshake.get("session_id") or f"ws-{uuid.uuid4().hex[:8]}"
    sr = int(handshake.get("sample_rate", 16000))
    ctx = SessionContext(
        caller_id=str(handshake.get("caller_id", "")),
        claimed_identity=str(handshake.get("claimed_identity", "")),
        transaction_value_inr=float(handshake.get("transaction_value_inr", 0.0)),
        origin_country=str(handshake.get("origin_country", "IN")),
        prior_fraud_score=float(handshake.get("prior_fraud_score", 0.0)),
    )
    sess = CallSession(sid, ctx)
    sess.set_sample_rate(sr)
    SESSIONS[sid] = sess

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data: Optional[bytes] = msg.get("bytes")
            if data is None:
                continue
            # interpret as int16 PCM mono
            pcm = np.frombuffer(data, dtype=np.int16)
            if pcm.size == 0:
                continue
            result = sess.push_chunk(pcm)
            if result is not None:
                await ws.send_json({
                    "timestamp_ms": result.timestamp_ms,
                    "risk_score": result.risk,
                    "p_spoof": result.p_spoof,
                    "tier": result.tier,
                    "metrics": result.metrics,
                    "trigger_challenge": result.trigger_challenge,
                    "interlock_active": result.interlock_active,
                    "receipt_hash": result.receipt_hash,
                    "challenge": result.challenge,
                })
    except WebSocketDisconnect:
        return
