"""
VaniRakshak — Demo Test Set Generator
Creates standardized benchmark files in data/demo/ from enrolled target audio:
  - demo_01_genuine.wav: 10s natural speech from target speaker
  - demo_02_full_clone.wav: 10s vocoded / pitch-shifted cloned voice simulation
  - demo_03_partial_splice.wav: 10s spliced audio (genuine [0-4s] -> synthetic [4-8s] -> genuine [8-10s])
  - ground_truth.json: Ground-truth audit labels for benchmark validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_AUDIO = REPO_ROOT / "data" / "train" / "genuine" / "Narendra_Modi_voice_16k.wav"
DEMO_DIR = REPO_ROOT / "data" / "demo"
SAMPLE_RATE = 16000


def simulate_synthetic_vocoder(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate neural vocoder / pitch-shift artifacts typical of voice cloning TTS models."""
    # 1. Robotic / harmonic compression via quantization and slight nonlinear harmonic distortion
    compressed = np.tanh(audio * 1.5)
    # 2. Phase jitter and high-frequency spectral flattening
    noise = np.random.normal(0, 0.005, size=len(audio)).astype(np.float32)
    vocoded = 0.85 * compressed + 0.15 * noise
    # 3. Formant shift simulation (resample and restore)
    factor = 1.06  # 6% pitch / formant drift
    n_resample = int(len(vocoded) / factor)
    resampled = resample(vocoded, n_resample)
    restored = np.interp(np.linspace(0, 1, len(audio)), np.linspace(0, 1, len(resampled)), resampled)
    # Peak normalize
    max_val = np.max(np.abs(restored)) + 1e-9
    return (restored / max_val * 0.95).astype(np.float32)


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_AUDIO.is_file():
        raise FileNotFoundError(f"Source genuine audio not found at: {SOURCE_AUDIO}")

    audio, sr = sf.read(SOURCE_AUDIO, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 10.0 seconds = 160,000 samples
    duration_samples = 10 * SAMPLE_RATE
    start_idx = 10 * SAMPLE_RATE  # start at 10s mark to avoid any leading silence
    genuine_10s = audio[start_idx : start_idx + duration_samples]

    # Normalize
    genuine_10s = (genuine_10s / (np.max(np.abs(genuine_10s)) + 1e-9) * 0.95).astype(np.float32)

    # 1. Genuine 10s
    p_gen = DEMO_DIR / "demo_01_genuine.wav"
    sf.write(p_gen, genuine_10s, SAMPLE_RATE)
    print(f"Generated {p_gen} (10.0s)")

    # 2. Synthetic Clone 10s
    clone_10s = simulate_synthetic_vocoder(genuine_10s, SAMPLE_RATE)
    p_clone = DEMO_DIR / "demo_02_full_clone.wav"
    sf.write(p_clone, clone_10s, SAMPLE_RATE)
    print(f"Generated {p_clone} (10.0s)")

    # 3. Partial Splice 10s: [0s-4s: genuine] + [4s-8s: synthetic clone] + [8s-10s: genuine]
    splice_10s = np.copy(genuine_10s)
    s4 = 4 * SAMPLE_RATE
    s8 = 8 * SAMPLE_RATE
    splice_10s[s4:s8] = clone_10s[s4:s8]
    p_splice = DEMO_DIR / "demo_03_partial_splice.wav"
    sf.write(p_splice, splice_10s, SAMPLE_RATE)
    print(f"Generated {p_splice} (10.0s)")

    # 4. Zero-Shot Impostor 10s (completely different vocal tract / pitch profile)
    impostor_factor = 1.35  # 35% pitch shift changes speaker identity completely
    n_imp = int(len(genuine_10s) / impostor_factor)
    imp_resampled = resample(genuine_10s, n_imp)
    impostor_10s = np.interp(np.linspace(0, 1, len(genuine_10s)), np.linspace(0, 1, len(imp_resampled)), imp_resampled)
    impostor_10s = (impostor_10s / (np.max(np.abs(impostor_10s)) + 1e-9) * 0.95).astype(np.float32)
    p_impostor = DEMO_DIR / "demo_04_impostor.wav"
    sf.write(p_impostor, impostor_10s, SAMPLE_RATE)
    print(f"Generated {p_impostor} (10.0s)")

    # 5. Ground Truth JSON
    gt = {
        "dataset_name": "VaniRakshak Demo Benchmark Suite",
        "sample_rate": SAMPLE_RATE,
        "files": {
            "demo_01_genuine.wav": {
                "label": "genuine",
                "target_speaker": "Narendra Modi",
                "is_clone": False,
                "is_spliced": False,
                "expected_verdict": "AUTHENTIC_TARGET",
            },
            "demo_02_full_clone.wav": {
                "label": "spoof",
                "target_speaker": "Narendra Modi (Cloned)",
                "is_clone": True,
                "is_spliced": False,
                "expected_verdict": "SYNTHETIC_VOICE_CLONE",
            },
            "demo_03_partial_splice.wav": {
                "label": "spliced",
                "target_speaker": "Narendra Modi",
                "is_clone": False,
                "is_spliced": True,
                "splice_regions_sec": [[4.0, 8.0]],
                "expected_verdict": "SUSPICIOUS_SPLICED_AUDIO",
            },
            "demo_04_impostor.wav": {
                "label": "impostor",
                "target_speaker": "Impostor (Different Speaker)",
                "is_clone": False,
                "is_spliced": False,
                "expected_verdict": "IMPOSTOR_SPEAKER",
            },
        },
    }
    p_gt = DEMO_DIR / "ground_truth.json"
    with open(p_gt, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Generated {p_gt}")


if __name__ == "__main__":
    main()
