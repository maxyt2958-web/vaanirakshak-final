"""
VaniRakshak — Forensic Schemas & Pydantic Data Contracts
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice retention on disk).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class WindowAnalysisResult(BaseModel):
    """Forensic evaluation for a single 2.0-second sliding window."""

    window_index: int = Field(description="Sequential index of the 2.0s window")
    start_sec: float = Field(description="Start offset in seconds")
    end_sec: float = Field(description="End offset in seconds")
    duration_sec: float = Field(default=2.0, description="Window duration")
    is_voiced: bool = Field(default=True, description="Voice activity detection flag")
    speech_ratio: float = Field(default=1.0, description="Ratio of active speech frames in window")

    # WavLM Deep Learning Features
    acoustic_vector_dim: int = Field(default=768, description="WavLM intermediate vector dimensionality")
    acoustic_vector_norm: float = Field(description="L2 norm of acoustic representation")
    acoustic_anomaly_score: float = Field(
        ge=0.0, le=1.0, description="Acoustic anomaly / synthetic voice score for this window"
    )

    # Biometric Speaker Identity (ECAPA-TDNN)
    speaker_cosine_sim: Optional[float] = Field(
        default=None, description="ECAPA cosine similarity against enrolled target anchor"
    )
    speaker_match: Optional[bool] = Field(
        default=None, description="Target speaker identity match flag"
    )

    # Forensic Signal Processing Features (librosa, scipy, numpy)
    micro_tremor_ratio: Optional[float] = Field(
        default=None, description="8–14 Hz physiological vocal micro-tremor ratio (Lippold band)"
    )
    spectral_flatness_mean: Optional[float] = Field(
        default=None, description="Mean spectral flatness (Wiener entropy)"
    )
    high_band_flatness: Optional[float] = Field(
        default=None, description="High-frequency (>4 kHz) vocoder spectral flatness"
    )
    zcr_variance: Optional[float] = Field(
        default=None, description="Zero-Crossing Rate (ZCR) temporal variance"
    )
    vocoder_boundary_score: Optional[float] = Field(
        default=None, description="Vocoder phase boundary discontinuity score"
    )

    inference_ms: float = Field(description="Computation latency in milliseconds")


class CallForensicReport(BaseModel):
    """Complete forensic assessment report for an audio session."""

    session_id: str = Field(description="Unique session identifier")
    audio_sha256: str = Field(description="Cryptographic SHA-256 fingerprint of processed audio")
    duration_sec: float = Field(description="Total duration in seconds")
    sample_rate: int = Field(default=16000, description="Processing sample rate")
    device_used: str = Field(description="Hardware device used for inference (cuda / cpu)")

    windows_count: int = Field(description="Total 2.0s sliding windows evaluated")
    windows: List[WindowAnalysisResult] = Field(default_factory=list, description="Detailed per-window telemetry")

    # Aggregate Deep Learning & Biometric Metrics
    mean_acoustic_anomaly: float = Field(ge=0.0, le=1.0, description="Mean WavLM acoustic anomaly score")
    max_acoustic_anomaly: float = Field(ge=0.0, le=1.0, description="Peak acoustic anomaly score")
    mean_speaker_sim: Optional[float] = Field(default=None, description="Average target speaker similarity")

    # Aggregate Forensic Signal Processing Metrics
    mean_micro_tremor: Optional[float] = Field(
        default=None, description="Average 8–14 Hz physiological vocal micro-tremor ratio"
    )
    mean_spectral_flatness: Optional[float] = Field(
        default=None, description="Average spectral flatness (Wiener entropy)"
    )
    mean_high_band_flatness: Optional[float] = Field(
        default=None, description="Average high-frequency (>4 kHz) vocoder spectral flatness"
    )
    mean_zcr_variance: Optional[float] = Field(
        default=None, description="Average Zero-Crossing Rate temporal variance"
    )
    vocoder_phase_discontinuities_detected: int = Field(
        default=0, description="Total count of detected vocoder phase boundary discontinuities"
    )
    detected_boundary_timestamps: List[float] = Field(
        default_factory=list, description="Timestamps (seconds) of detected vocoder phase discontinuities"
    )

    overall_risk_score: float = Field(
        ge=0.0, le=100.0, description="Calibrated composite threat score (0 = authentic, 100 = deepfake clone)"
    )
    verdict: Literal[
        "AUTHENTIC_TARGET",
        "IMPOSTOR_SPEAKER",
        "SYNTHETIC_VOICE_CLONE",
        "SUSPICIOUS_SPLICED_AUDIO",
        "INSUFFICIENT_SPEECH",
    ] = Field(description="Final forensic decision")

    # Compliance & Chain-of-Custody
    dpdp_compliance: Dict[str, Any] = Field(
        description="DPDP Act 2023 zero-retention compliance verification and cryptographic receipt"
    )
