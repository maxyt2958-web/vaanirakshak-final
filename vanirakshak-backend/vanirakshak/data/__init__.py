"""Audio data utilities (R1)."""

from .augment import (
    AUGMENT_REGISTRY,
    apply_augmentation,
    apply_random_pipeline,
    alaw_encode_decode,
    ulaw_encode_decode,
    amr_wb_simulate,
    gsm_efr_simulate,
    add_noise,
    add_babble_simple,
    apply_rir,
    synthetic_rir,
    random_gain,
)
from .synthetic import synthetic_speech, synthetic_spoofed, synthetic_silence

__all__ = [
    "AUGMENT_REGISTRY",
    "apply_augmentation",
    "apply_random_pipeline",
    "alaw_encode_decode",
    "ulaw_encode_decode",
    "amr_wb_simulate",
    "gsm_efr_simulate",
    "add_noise",
    "add_babble_simple",
    "apply_rir",
    "synthetic_rir",
    "random_gain",
    "synthetic_speech",
    "synthetic_spoofed",
    "synthetic_silence",
]
