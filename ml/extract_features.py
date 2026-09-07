"""
VaniRakshak — Batch & CLI Feature Extraction Pipeline
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention voice processing).

Loads microsoft/wavlm-base-plus directly onto CUDA/CPU and extracts 768-dimensional
intermediate acoustic vectors across 2.0-second sliding windows (32,000 samples @ 16 kHz).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.audio import AudioProcessor, compute_audio_sha256
from backend.app.core.vad import EnergyVAD
from backend.app.engine.sliding_window import SlidingWindowEngine
from backend.app.models.wavlm_detector import WavLMFeatureExtractor
from backend.app.models.ecapa_verifier import ECAPAVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vanirakshak.extract_features")


def extract_file_features(
    audio_path: Path,
    wavlm: WavLMFeatureExtractor,
    ecapa: Optional[ECAPAVerifier] = None,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Extract 768-dim WavLM and 192-dim ECAPA features across 2.0s sliding windows."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read audio in-memory
    audio_bytes = audio_path.read_bytes()
    processor = AudioProcessor(target_sr=sample_rate)
    waveform, audio_sha256, duration_sec = processor.decode_bytes(audio_bytes, target_sr=sample_rate)

    # Voice Activity Detection
    vad = EnergyVAD(sample_rate=sample_rate)

    # 2.0-Second Sliding Windows
    slicer = SlidingWindowEngine(
        sample_rate=sample_rate,
        window_size_sec=window_sec,
        hop_size_sec=hop_sec,
    )
    windows = slicer.slice_waveform(waveform, vad_engine=vad)

    window_features_768: List[np.ndarray] = []
    window_metadata: List[Dict[str, Any]] = []

    t_start = time.perf_counter()

    for w in windows:
        # Extract 768-dimensional intermediate acoustic vector
        anomaly_score, vec_768, latency_ms = wavlm.compute_acoustic_anomaly_score(w.samples, sample_rate)
        window_features_768.append(vec_768)

        meta: Dict[str, Any] = {
            "window_index": w.index,
            "start_sec": w.start_sec,
            "end_sec": w.end_sec,
            "duration_sec": w.duration_sec,
            "is_voiced": w.is_voiced,
            "speech_ratio": round(w.speech_ratio, 3),
            "acoustic_vector_dim": int(vec_768.shape[0]),
            "acoustic_vector_norm": round(float(np.linalg.norm(vec_768)), 4),
            "acoustic_anomaly_score": round(anomaly_score, 4),
            "inference_ms": round(latency_ms, 2),
        }

        # If ECAPA verifier is provided and window has speech, verify speaker identity
        if ecapa is not None and ecapa.has_anchor and w.is_voiced:
            cos_sim, cal_score, match = ecapa.verify(w.samples, sample_rate)
            meta["speaker_cosine_sim"] = round(cos_sim, 4)
            meta["speaker_match"] = match

        window_metadata.append(meta)

    total_extract_ms = (time.perf_counter() - t_start) * 1000.0

    # Stack features into (N, 768) matrix
    stacked_768 = np.stack(window_features_768, axis=0) if window_features_768 else np.empty((0, 768))

    return {
        "file_name": audio_path.name,
        "audio_sha256": audio_sha256,
        "duration_sec": round(duration_sec, 3),
        "total_windows": len(windows),
        "feature_shape": list(stacked_768.shape),
        "total_extraction_ms": round(total_extract_ms, 2),
        "device": wavlm.device,
        "features": stacked_768,
        "windows": window_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VaniRakshak — 768-dim WavLM Acoustic Vector Feature Extractor"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="data/train/genuine/Narendra_Modi_voice_16k.wav",
        help="Path to audio file to extract features from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path to save extracted features (.pt or .npy)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Sliding window duration in seconds (default: 2.0s = 32,000 samples @ 16kHz)",
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        default=0.5,
        help="Sliding window hop size in seconds (default: 0.5s = 8,000 samples @ 16kHz)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="data/anchor_speaker_embedding.pt",
        help="Path to pre-cached ECAPA speaker anchor embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    # Initialize WavLM Feature Extractor
    wavlm = WavLMFeatureExtractor(device=args.device)

    # Initialize ECAPA Verifier if anchor exists
    anchor_path = Path(args.anchor)
    ecapa = None
    if anchor_path.is_file():
        logger.info(f"Loading speaker anchor from {anchor_path}...")
        ecapa = ECAPAVerifier(device=args.device, anchor_path=anchor_path)

    logger.info(f"Extracting 768-dim features across 2.0s sliding windows from {audio_path}...")
    result = extract_file_features(
        audio_path=audio_path,
        wavlm=wavlm,
        ecapa=ecapa,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
    )

    logger.info("=== Feature Extraction Summary ===")
    logger.info(f"File: {result['file_name']}")
    logger.info(f"SHA-256: {result['audio_sha256']}")
    logger.info(f"Duration: {result['duration_sec']}s")
    logger.info(f"Sliding Windows (2.0s): {result['total_windows']}")
    logger.info(f"Feature Matrix Shape: {result['feature_shape']} (768-dim intermediate vectors)")
    logger.info(f"Total Processing Time: {result['total_extraction_ms']} ms")
    logger.info(f"Inference Device: {result['device']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".pt":
            torch.save(
                {
                    "features": torch.from_numpy(result["features"]),
                    "metadata": {k: v for k, v in result.items() if k != "features"},
                },
                out_path,
            )
        else:
            np.save(out_path, result["features"])
        logger.info(f"Saved extracted features to {out_path}")

    # Output window summaries
    print("\n--- Per-Window Telemetry (First 5 Windows) ---")
    for w in result["windows"][:5]:
        asv_info = f", Target Sim: {w.get('speaker_cosine_sim', 'N/A')}" if "speaker_cosine_sim" in w else ""
        print(
            f"Window #{w['window_index']} [{w['start_sec']}s - {w['end_sec']}s]: "
            f"Anomaly Score={w['acoustic_anomaly_score']}, SpeechRatio={w['speech_ratio']}, Latency={w['inference_ms']}ms{asv_info}"
        )


if __name__ == "__main__":
    main()
