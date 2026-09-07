"""Detector subpackage: CM, ASV, channel, prosody, and the unified pipeline."""

from .cm import CMModel
from .asv import ASVModel, EMBED_DIM
from .channel import detect as channel_detect
from .prosody import prosody_score
from .pipeline import DetectionPipeline, WindowResult

__all__ = [
    "CMModel",
    "ASVModel",
    "EMBED_DIM",
    "channel_detect",
    "prosody_score",
    "DetectionPipeline",
    "WindowResult",
]
