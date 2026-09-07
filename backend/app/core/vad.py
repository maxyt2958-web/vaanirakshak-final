"""
VaniRakshak — Voice Activity Detection (VAD) Engine
Identifies speech activity to ensure acoustic models only process valid speech segments.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("vanirakshak.vad")


class EnergyVAD:
    """Robust in-memory Energy & Spectral Voice Activity Detector."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 30.0,
        energy_threshold_db: float = -45.0,
        min_speech_ratio: float = 0.25,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold_db = energy_threshold_db
        self.min_speech_ratio = min_speech_ratio

    def compute_frame_energy(self, frame: np.ndarray) -> float:
        """Compute frame RMS energy in decibels (dB)."""
        rms = np.sqrt(np.mean(frame**2) + 1e-12)
        return float(20.0 * np.log10(rms + 1e-12))

    def detect_speech(self, audio: np.ndarray) -> Tuple[bool, float, List[bool]]:
        """Evaluate if the given audio slice contains sufficient speech content.
        
        Returns:
            (has_speech: bool, speech_ratio: float, frame_decisions: List[bool])
        """
        x = np.asarray(audio, dtype=np.float32).ravel()
        if len(x) < self.frame_size:
            return False, 0.0, []

        n_frames = len(x) // self.frame_size
        frame_decisions: List[bool] = []

        for i in range(n_frames):
            frame = x[i * self.frame_size : (i + 1) * self.frame_size]
            db = self.compute_frame_energy(frame)
            frame_decisions.append(db >= self.energy_threshold_db)

        if not frame_decisions:
            return False, 0.0, []

        speech_ratio = sum(frame_decisions) / len(frame_decisions)
        has_speech = speech_ratio >= self.min_speech_ratio

        return has_speech, speech_ratio, frame_decisions
