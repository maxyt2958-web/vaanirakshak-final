"""
VaniRakshak — Configuration & Hyperparameters
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention voice biometrics).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and hyperparameter settings."""

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    ANCHOR_EMBEDDING_PATH: Path = DATA_DIR / "anchor_speaker_embedding.pt"
    ANCHOR_PROFILE_PATH: Path = DATA_DIR / "anchor_speaker_profile.json"

    # Hardware & Compute Layer
    FORCE_CPU: bool = False
    DEVICE: str = "cuda" if torch.cuda.is_available() and not os.getenv("FORCE_CPU") else "cpu"
    USE_FP16_ON_CUDA: bool = True
    NUM_WORKERS: int = 4

    # Deep Learning & Feature Extraction Layer (WavLM Base+)
    WAVLM_MODEL_ID: str = "microsoft/wavlm-base-plus"
    WAVLM_HIDDEN_DIM: int = 768
    WAVLM_EXTRACT_LAYER: int = -1  # Last hidden state or weighted average

    # Speaker Verification Layer (ECAPA-TDNN)
    ECAPA_MODEL_ID: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_EMBED_DIM: int = 192
    ASV_SIMILARITY_THRESHOLD: float = 0.55

    # 2.0-Second Sliding Window & Acoustic Layer
    SAMPLE_RATE: int = 16000
    WINDOW_SIZE_SEC: float = 2.0
    HOP_SIZE_SEC: float = 0.5
    WINDOW_SAMPLES: int = int(WINDOW_SIZE_SEC * SAMPLE_RATE)  # 32,000 samples
    HOP_SAMPLES: int = int(HOP_SIZE_SEC * SAMPLE_RATE)        # 8,000 samples

    # VAD & Acoustic Filtering
    VAD_FRAME_MS: float = 30.0
    VAD_ENERGY_THRESHOLD_DB: float = -45.0
    VAD_MIN_SPEECH_RATIO: float = 0.25

    # DPDP Act 2023 Compliance
    ZERO_RAW_AUDIO_RETENTION: bool = True
    ENABLE_AUDIT_LOGGING: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
