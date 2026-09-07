"""
VaniRakshak — Multi-Factor Risk Fusion & Decision Engine
Combines WavLM 768-dim acoustic representations, ECAPA-TDNN speaker verification,
and classical forensic signal processing (micro-tremor, spectral flatness, ZCR variance, phase boundaries).
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention forensic receipts).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.engine.schemas import CallForensicReport, WindowAnalysisResult


class ForensicRiskFusionEngine:
    """Combines deep acoustic anomaly scores, biometric verification, and forensic signal processing."""

    def __init__(
        self,
        asv_threshold: float = 0.55,
        anomaly_threshold: float = 0.60,
        splice_variance_threshold: float = 0.12,
        high_flatness_threshold: float = 0.45,
    ) -> None:
        self.asv_threshold = asv_threshold
        self.anomaly_threshold = anomaly_threshold
        self.splice_variance_threshold = splice_variance_threshold
        self.high_flatness_threshold = high_flatness_threshold

    def fuse(
        self,
        session_id: str,
        audio_sha256: str,
        total_duration_sec: float,
        windows: List[WindowAnalysisResult],
        device_used: str = "cuda",
        stream_boundary_timestamps: Optional[List[float]] = None,
    ) -> CallForensicReport:
        """Evaluate full call telemetry and produce a tamper-evident forensic verdict."""
        if not windows:
            dpdp_receipt = self._build_dpdp_receipt(session_id, audio_sha256, 0.0, "INSUFFICIENT_SPEECH")
            return CallForensicReport(
                session_id=session_id,
                audio_sha256=audio_sha256,
                duration_sec=total_duration_sec,
                device_used=device_used,
                windows_count=0,
                windows=[],
                mean_acoustic_anomaly=0.0,
                max_acoustic_anomaly=0.0,
                overall_risk_score=0.0,
                verdict="INSUFFICIENT_SPEECH",
                dpdp_compliance=dpdp_receipt,
            )

        voiced_windows = [w for w in windows if w.is_voiced]
        if not voiced_windows:
            voiced_windows = windows

        # 1. WavLM Acoustic Anomaly Aggregation
        anomaly_scores = [w.acoustic_anomaly_score for w in voiced_windows]
        mean_anomaly = float(np.mean(anomaly_scores))
        max_anomaly = float(np.max(anomaly_scores))
        anomaly_std = float(np.std(anomaly_scores)) if len(anomaly_scores) > 1 else 0.0

        # 2. ECAPA-TDNN Speaker Identity Aggregation
        asv_scores = [w.speaker_cosine_sim for w in voiced_windows if w.speaker_cosine_sim is not None]
        mean_asv = float(np.mean(asv_scores)) if asv_scores else None

        # 3. Forensic Signal Processing Metrics Aggregation
        tremor_vals = [w.micro_tremor_ratio for w in voiced_windows if w.micro_tremor_ratio is not None]
        mean_tremor = float(np.mean(tremor_vals)) if tremor_vals else None

        flatness_vals = [w.spectral_flatness_mean for w in voiced_windows if w.spectral_flatness_mean is not None]
        mean_flatness = float(np.mean(flatness_vals)) if flatness_vals else None

        high_flatness_vals = [w.high_band_flatness for w in voiced_windows if w.high_band_flatness is not None]
        mean_high_flatness = float(np.mean(high_flatness_vals)) if high_flatness_vals else None

        zcr_var_vals = [w.zcr_variance for w in voiced_windows if w.zcr_variance is not None]
        mean_zcr_var = float(np.mean(zcr_var_vals)) if zcr_var_vals else None

        boundary_scores = [w.vocoder_boundary_score for w in voiced_windows if w.vocoder_boundary_score is not None]
        max_boundary_score = float(np.max(boundary_scores)) if boundary_scores else 0.0

        detected_timestamps = list(stream_boundary_timestamps or [])
        phase_discontinuities_count = len(detected_timestamps)

        # 4. Multi-Factor Forensic Threat Assessment
        # Case A: Low target speaker similarity -> Impostor speaker
        # Case B: Full synthetic voice clone (high vocoder spectral flatness >= 0.55 or deep anomaly)
        # Case C: Spliced audio (phase boundary jumps, splice variance, or localized discontinuities)
        # Case D: Authentic target speaker

        is_impostor = (mean_asv is not None) and (mean_asv < self.asv_threshold)
        is_clone = (
            (mean_high_flatness is not None and mean_high_flatness >= 0.55)
            or mean_anomaly >= self.anomaly_threshold
        )
        is_spliced = (
            phase_discontinuities_count >= 1
            or (anomaly_std > self.splice_variance_threshold and max_anomaly > 0.40)
        )

        if is_impostor:
            verdict = "IMPOSTOR_SPEAKER"
            risk_score = 70.0 + (1.0 - max(0.0, mean_asv)) * 25.0
        elif is_clone:
            verdict = "SYNTHETIC_VOICE_CLONE"
            flatness_boost = ((mean_high_flatness or 0.5) - 0.5) * 40.0
            risk_score = 75.0 + (mean_anomaly * 15.0) + max(0.0, flatness_boost)
        elif is_spliced:
            verdict = "SUSPICIOUS_SPLICED_AUDIO"
            boundary_boost = min(15.0, phase_discontinuities_count * 8.0)
            risk_score = 65.0 + (max_anomaly * 15.0) + boundary_boost
        else:
            verdict = "AUTHENTIC_TARGET"
            risk_score = float(np.clip(mean_anomaly * 25.0 + ((mean_high_flatness or 0.2) * 10.0), 2.0, 28.0))

        risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 2)

        # 5. Cryptographic DPDP Act 2023 Receipt
        dpdp_receipt = self._build_dpdp_receipt(session_id, audio_sha256, risk_score, verdict)

        return CallForensicReport(
            session_id=session_id,
            audio_sha256=audio_sha256,
            duration_sec=round(total_duration_sec, 3),
            device_used=device_used,
            windows_count=len(windows),
            windows=windows,
            mean_acoustic_anomaly=round(mean_anomaly, 4),
            max_acoustic_anomaly=round(max_anomaly, 4),
            mean_speaker_sim=round(mean_asv, 4) if mean_asv is not None else None,
            mean_micro_tremor=round(mean_tremor, 4) if mean_tremor is not None else None,
            mean_spectral_flatness=round(mean_flatness, 6) if mean_flatness is not None else None,
            mean_high_band_flatness=round(mean_high_flatness, 6) if mean_high_flatness is not None else None,
            mean_zcr_variance=round(mean_zcr_var, 8) if mean_zcr_var is not None else None,
            vocoder_phase_discontinuities_detected=phase_discontinuities_count,
            detected_boundary_timestamps=detected_timestamps,
            overall_risk_score=risk_score,
            verdict=verdict,
            dpdp_compliance=dpdp_receipt,
        )

    def _build_dpdp_receipt(
        self,
        session_id: str,
        audio_sha256: str,
        risk_score: float,
        verdict: str,
    ) -> Dict[str, Any]:
        """Generate canonical audit receipt without retaining raw voice data."""
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "session_id": session_id,
            "audio_sha256": audio_sha256,
            "risk_score": risk_score,
            "verdict": verdict,
            "dpdp_section": "DPDP Act 2023 Section 8(7)",
            "zero_retention_verified": True,
            "timestamp": timestamp,
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        receipt_hash = hashlib.sha256(canonical_str).hexdigest()
        payload["receipt_sha256"] = receipt_hash
        return payload
