"""
VaniRakshak — Live & Batch Audio Forensic Analysis Endpoint
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention voice biometrics).

Processes audio strictly in-memory, standardizes audio tensors to 16 kHz mono, extracts
768-dim WavLM acoustic vectors across 2.0-second sliding windows, performs ECAPA speaker
verification against the target anchor, calculates micro-tremor, spectral flatness, and ZCR variance
to detect vocoder phase boundaries, and produces a cryptographic forensic audit report.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.app.config import settings
from backend.app.core.audio import AudioProcessor
from backend.app.core.forensic_signal import ForensicSignalProcessor
from backend.app.core.vad import EnergyVAD
from backend.app.engine.risk_fusion import ForensicRiskFusionEngine
from backend.app.engine.schemas import CallForensicReport, WindowAnalysisResult
from backend.app.engine.sliding_window import SlidingWindowEngine
from backend.app.models.ecapa_verifier import ECAPAVerifier
from backend.app.models.wavlm_detector import WavLMFeatureExtractor

logger = logging.getLogger("vanirakshak.api.analyze")
router = APIRouter(prefix="/analyze", tags=["Forensic Analysis"])

# Module-level singletons for low-overhead inference
_audio_processor: Optional[AudioProcessor] = None
_forensic_processor: Optional[ForensicSignalProcessor] = None
_vad_engine: Optional[EnergyVAD] = None
_sliding_engine: Optional[SlidingWindowEngine] = None
_wavlm_extractor: Optional[WavLMFeatureExtractor] = None
_ecapa_verifier: Optional[ECAPAVerifier] = None
_fusion_engine: Optional[ForensicRiskFusionEngine] = None


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor(target_sr=settings.SAMPLE_RATE)
    return _audio_processor


def get_forensic_processor() -> ForensicSignalProcessor:
    global _forensic_processor
    if _forensic_processor is None:
        _forensic_processor = ForensicSignalProcessor(target_sr=settings.SAMPLE_RATE)
    return _forensic_processor


def get_vad_engine() -> EnergyVAD:
    global _vad_engine
    if _vad_engine is None:
        _vad_engine = EnergyVAD(
            sample_rate=settings.SAMPLE_RATE,
            frame_duration_ms=settings.VAD_FRAME_MS,
            energy_threshold_db=settings.VAD_ENERGY_THRESHOLD_DB,
            min_speech_ratio=settings.VAD_MIN_SPEECH_RATIO,
        )
    return _vad_engine


def get_sliding_engine() -> SlidingWindowEngine:
    global _sliding_engine
    if _sliding_engine is None:
        _sliding_engine = SlidingWindowEngine(
            sample_rate=settings.SAMPLE_RATE,
            window_size_sec=settings.WINDOW_SIZE_SEC,
            hop_size_sec=settings.HOP_SIZE_SEC,
        )
    return _sliding_engine


def get_wavlm_extractor() -> WavLMFeatureExtractor:
    global _wavlm_extractor
    if _wavlm_extractor is None:
        _wavlm_extractor = WavLMFeatureExtractor(
            model_id=settings.WAVLM_MODEL_ID,
            device=settings.DEVICE,
            use_fp16=settings.USE_FP16_ON_CUDA,
        )
    return _wavlm_extractor


def get_ecapa_verifier() -> ECAPAVerifier:
    global _ecapa_verifier
    if _ecapa_verifier is None:
        anchor_path = settings.ANCHOR_EMBEDDING_PATH if settings.ANCHOR_EMBEDDING_PATH.is_file() else None
        _ecapa_verifier = ECAPAVerifier(
            device=settings.DEVICE,
            model_source=settings.ECAPA_MODEL_ID,
            anchor_path=anchor_path,
        )
    return _ecapa_verifier


def get_fusion_engine() -> ForensicRiskFusionEngine:
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = ForensicRiskFusionEngine(
            asv_threshold=settings.ASV_SIMILARITY_THRESHOLD,
        )
    return _fusion_engine


@router.post(
    "/audio",
    response_model=CallForensicReport,
    summary="Perform zero-retention forensic deepfake, signal & speaker analysis on recorded audio",
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file payload (WAV, MP3, FLAC, OGG)"),
    session_id: Optional[str] = Query(None, description="Optional custom session UUID"),
) -> CallForensicReport:
    """Analyze audio stream using 2.0s sliding windows, 768-dim WavLM features, forensic DSP, and ECAPA."""
    sid = session_id or str(uuid.uuid4())
    logger.info(f"Starting forensic analysis for session {sid}, file={file.filename}")

    # Read uploaded bytes directly in memory (DPDP Act 2023 zero raw voice retention)
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio stream: {e}",
        )

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Received empty audio payload.",
        )

    # 1. Decode & standardise audio in-memory to 16 kHz mono float32
    processor = get_audio_processor()
    try:
        _, waveform, audio_sha256, duration_sec = processor.standardize_to_16k_mono(raw_bytes)
    except Exception as decode_err:
        logger.error(f"Decoding failed for session {sid}: {decode_err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to decode audio format: {decode_err}",
        )

    # 2. Forensic Signal Processing (Full stream vocoder phase boundaries)
    forensic_dsp = get_forensic_processor()
    stream_phase_res = forensic_dsp.detect_vocoder_phase_boundaries(waveform, sr=settings.SAMPLE_RATE)

    # 3. VAD & 2.0-Second Sliding Windows
    vad = get_vad_engine()
    slicer = get_sliding_engine()
    windows = slicer.slice_waveform(waveform, vad_engine=vad)

    wavlm = get_wavlm_extractor()
    ecapa = get_ecapa_verifier()

    window_results: list[WindowAnalysisResult] = []

    # 4. Sliding Window Inference
    for w in windows:
        anomaly_score, vec_768, wavlm_latency = wavlm.compute_acoustic_anomaly_score(
            w.samples, sample_rate=settings.SAMPLE_RATE
        )

        speaker_sim: Optional[float] = None
        speaker_match: Optional[bool] = None

        if ecapa.has_anchor and w.is_voiced:
            try:
                sim, _, match = ecapa.verify(w.samples, sr=settings.SAMPLE_RATE)
                speaker_sim = round(sim, 4)
                speaker_match = match
            except Exception as asv_err:
                logger.warning(f"ECAPA verification error on window {w.index}: {asv_err}")

        # Compute window forensic signal telemetry (micro-tremor, spectral flatness, ZCR variance)
        dsp_win = forensic_dsp.analyze_window(w.samples, sr=settings.SAMPLE_RATE)

        window_results.append(
            WindowAnalysisResult(
                window_index=w.index,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                duration_sec=w.duration_sec,
                is_voiced=w.is_voiced,
                speech_ratio=round(w.speech_ratio, 3),
                acoustic_vector_dim=int(vec_768.shape[0]),
                acoustic_vector_norm=round(float(np.linalg.norm(vec_768)), 4),
                acoustic_anomaly_score=round(anomaly_score, 4),
                speaker_cosine_sim=speaker_sim,
                speaker_match=speaker_match,
                micro_tremor_ratio=dsp_win.micro_tremor.tremor_ratio,
                spectral_flatness_mean=dsp_win.spectral_flatness.mean_flatness,
                high_band_flatness=dsp_win.spectral_flatness.high_band_flatness,
                zcr_variance=dsp_win.zcr.zcr_variance,
                vocoder_boundary_score=dsp_win.phase_boundary.boundary_discontinuity_score,
                inference_ms=round(wavlm_latency + dsp_win.latency_ms, 2),
            )
        )

    # 5. Multi-Factor Risk Fusion & Cryptographic Receipt
    fusion = get_fusion_engine()
    report = fusion.fuse(
        session_id=sid,
        audio_sha256=audio_sha256,
        total_duration_sec=duration_sec,
        windows=window_results,
        device_used=settings.DEVICE,
        stream_boundary_timestamps=stream_phase_res.boundary_timestamps,
    )

    logger.info(
        f"Forensic evaluation complete: session={sid}, windows={len(windows)}, "
        f"risk_score={report.overall_risk_score}, verdict={report.verdict}"
    )

    return report
