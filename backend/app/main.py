"""
VaniRakshak — Enterprise Forensic Audio Intelligence Server
Target-Conditioned Recorded-Audio Deepfake Detection Platform.
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice data retention).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.config import settings
from backend.app.engine.sliding_window import StreamingWindowBuffer
from backend.app.models.ecapa_verifier import ECAPAVerifier
from backend.app.models.wavlm_detector import WavLMFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vanirakshak.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("==================================================")
    logger.info("   Starting VaniRakshak Audio Forensics Engine    ")
    logger.info(f"   Target Model: {settings.WAVLM_MODEL_ID}")
    logger.info(f"   Acoustic Vector Dim: {settings.WAVLM_HIDDEN_DIM}")
    logger.info(f"   Active Device: {settings.DEVICE}")
    logger.info(f"   DPDP Compliance: Zero raw audio retention ON")
    logger.info("==================================================")
    yield
    logger.info("Shutting down VaniRakshak Engine. Cleaning in-memory buffers.")


app = FastAPI(
    title="VaniRakshak Forensic Audio Engine",
    description=(
        "Production-grade acoustic deepfake & voice cloning forensic engine "
        "leveraging microsoft/wavlm-base-plus 768-dim intermediate vectors across "
        "2.0s sliding windows and ECAPA-TDNN speaker verification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_router)


# Real-time WebSocket Streaming Forensics Endpoint
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """Real-time forensic stream evaluation over WebSocket using 2.0-second sliding windows."""
    await websocket.accept()
    logger.info("Client connected to real-time forensic WebSocket stream.")

    buffer = StreamingWindowBuffer(
        sample_rate=settings.SAMPLE_RATE,
        window_size_sec=settings.WINDOW_SIZE_SEC,
        hop_size_sec=settings.HOP_SIZE_SEC,
    )

    wavlm = WavLMFeatureExtractor(
        model_id=settings.WAVLM_MODEL_ID,
        device=settings.DEVICE,
        use_fp16=settings.USE_FP16_ON_CUDA,
    )

    anchor_path = settings.ANCHOR_EMBEDDING_PATH if settings.ANCHOR_EMBEDDING_PATH.is_file() else None
    ecapa = ECAPAVerifier(device=settings.DEVICE, anchor_path=anchor_path)

    try:
        while True:
            # Receive binary PCM float32/int16 chunk or JSON message
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                raw_chunk = message["bytes"]
                # Decode 16-bit PCM chunk to float32
                samples = list(memoryview(raw_chunk).cast("h"))
                float_samples = [s / 32768.0 for s in samples]

                ready_windows = buffer.feed_chunk(float_samples)
                for w in ready_windows:
                    anomaly_score, vec_768, latency_ms = wavlm.compute_acoustic_anomaly_score(
                        w.samples, settings.SAMPLE_RATE
                    )
                    speaker_sim = None
                    if ecapa.has_anchor:
                        sim, _, _ = ecapa.verify(w.samples, settings.SAMPLE_RATE)
                        speaker_sim = round(sim, 4)

                    telemetry = {
                        "type": "WINDOW_ANALYSIS",
                        "window_index": w.index,
                        "start_sec": w.start_sec,
                        "end_sec": w.end_sec,
                        "acoustic_anomaly_score": round(anomaly_score, 4),
                        "speaker_similarity": speaker_sim,
                        "inference_ms": round(latency_ms, 2),
                    }
                    await websocket.send_text(json.dumps(telemetry))

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("action") == "RESET":
                    buffer.reset()
                    await websocket.send_text(json.dumps({"type": "STATUS", "message": "BUFFER_RESET"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected from forensic stream. Buffers purged.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        buffer.reset()
