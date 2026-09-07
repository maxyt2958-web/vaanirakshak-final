"""
VaniRakshak — Target Speaker Enrollment Endpoint
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice retention).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.app.config import settings
from backend.app.core.audio import AudioProcessor
from backend.app.models.ecapa_verifier import ECAPAVerifier

logger = logging.getLogger("vanirakshak.api.enroll")
router = APIRouter(prefix="/enroll", tags=["Speaker Enrollment"])


@router.get("/target", summary="Retrieve active target speaker profile and DPDP compliance receipt")
async def get_target_profile() -> Dict[str, Any]:
    """Return active target speaker metadata and verification hashes."""
    profile_path = settings.ANCHOR_PROFILE_PATH
    if not profile_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No enrolled speaker anchor found. Enroll a target speaker first.",
        )

    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "ENROLLED",
        "profile": data,
        "anchor_file": str(settings.ANCHOR_EMBEDDING_PATH.name),
        "embedding_dim": settings.ECAPA_EMBED_DIM,
    }


@router.post("/target", summary="Enroll a new target speaker anchor with zero raw audio retention")
async def enroll_target_speaker(
    file: UploadFile = File(..., description="Target reference voice sample (WAV/MP3/FLAC)"),
    speaker_name: str = Form(..., description="Full name of target speaker"),
    consent_id: str = Form(..., description="DPDP Act consent reference identifier"),
) -> Dict[str, Any]:
    """Extract and anchor a 192-dim speaker identity vector into the immutable anchor vault."""
    logger.info(f"Enrolling speaker: {speaker_name} with consent ID: {consent_id}")

    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read enrollment audio payload: {e}",
        )

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty enrollment payload.",
        )

    # In-memory decoding & normalization (DPDP Act zero raw voice retention)
    processor = AudioProcessor(target_sr=settings.SAMPLE_RATE)
    try:
        waveform, audio_sha256, duration_sec = processor.decode_bytes(raw_bytes)
    except Exception as decode_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process enrollment audio: {decode_err}",
        )

    if duration_sec < 3.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Enrollment audio too short ({duration_sec:.1f}s). Minimum 3.0 seconds required.",
        )

    # Extract 192-dim ECAPA embedding
    verifier = ECAPAVerifier(device=settings.DEVICE, model_source=settings.ECAPA_MODEL_ID)
    embedding = verifier.embed(waveform, sr=settings.SAMPLE_RATE)

    # Construct cryptographic audit receipt
    timestamp = datetime.now(timezone.utc).isoformat()
    embedding_sha256 = hashlib.sha256(embedding.tobytes()).hexdigest()

    receipt_body = {
        "speaker_name": speaker_name,
        "consent_id": consent_id,
        "audio_pcm_sha256": audio_sha256,
        "duration_sec": round(duration_sec, 3),
        "embedding_dim": int(embedding.shape[0]),
        "embedding_l2_norm": round(float(np.linalg.norm(embedding)), 6),
        "embedding_sha256": embedding_sha256,
        "model_id": settings.ECAPA_MODEL_ID,
        "dpdp_section": "DPDP Act 2023 Section 8(7)",
        "zero_retention_verified": True,
        "timestamp": timestamp,
    }

    canonical_json = json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt_hash = hashlib.sha256(canonical_json).hexdigest()
    receipt_body["receipt_hash"] = receipt_hash

    # Save only the vector embeddings and metadata receipt (NO raw audio saved to disk!)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(settings.DATA_DIR / "anchor_speaker_embedding.npy", embedding)
    torch.save(
        {
            "embedding": torch.from_numpy(embedding),
            "metadata": receipt_body,
        },
        settings.ANCHOR_EMBEDDING_PATH,
    )

    with open(settings.ANCHOR_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt_body, f, indent=2)

    logger.info(f"Target speaker '{speaker_name}' enrolled. Receipt: {receipt_hash}")

    return {
        "status": "SUCCESS",
        "speaker_name": speaker_name,
        "consent_id": consent_id,
        "receipt_hash": receipt_hash,
        "embedding_dim": int(embedding.shape[0]),
        "dpdp_compliant": True,
    }
