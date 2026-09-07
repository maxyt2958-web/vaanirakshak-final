"""End-to-end per-window detection pipeline.

Glues together:
  * CM (countermeasure / AASIST-stub)
  * ASV (speaker verification / ECAPA-stub)
  * Channel anomaly
  * Prosody

Returns a :class:`WindowResult` that the risk fusion engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .cm import CMModel
from .asv import ASVModel
from .channel import detect as channel_detect
from .prosody import prosody_score


@dataclass
class WindowResult:
    cm_score: float            # LLR: positive ⇒ genuine
    asv_score: float           # consistency in [0,1] (0.5 if not enrolled)
    asv_enrolled: bool
    channel_anomaly: float     # in [0,1]
    prosody_unnatural: float   # in [0,1]
    codec_guess: str
    snr_db: float
    features: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionPipeline:
    """Stateless wrapper around the four detectors."""

    def __init__(self, cm: Optional[CMModel] = None, asv: Optional[ASVModel] = None) -> None:
        self.cm = cm or CMModel()
        self.asv = asv or ASVModel()

    def enrol(self, user_id: str, x: np.ndarray, sr: int) -> np.ndarray:
        return self.asv.enrol(user_id, x, sr)

    def is_enrolled(self, user_id: str) -> bool:
        return self.asv.is_enrolled(user_id)

    def analyse(self, x: np.ndarray, sr: int, claimed_user: Optional[str] = None) -> WindowResult:
        llr = self.cm.score(x, sr)
        # squash LLR to a [0,1] "genuine" probability (calibrated)
        cm_genuine_prob = float(1.0 / (1.0 + np.exp(-llr)))

        asv_consistency, enrolled = (0.5, False)
        if claimed_user:
            asv_consistency, enrolled = self.asv.consistency_score(claimed_user, x, sr)

        ch = channel_detect(x, sr)
        pr = prosody_score(x, sr)

        return WindowResult(
            cm_score=llr,
            asv_score=asv_consistency,
            asv_enrolled=enrolled,
            channel_anomaly=ch["anomaly"],
            prosody_unnatural=pr["unnatural"],
            codec_guess=ch["codec_guess"],
            snr_db=ch["snr_db"],
            features={
                "cm_genuine_prob": cm_genuine_prob,
                "asv_consistency": asv_consistency,
                "channel_anomaly": ch["anomaly"],
                "prosody_unnatural": pr["unnatural"],
                "snr_db": ch["snr_db"],
                "low_band_share": ch["low_band_share"],
                **pr,
            },
        )
