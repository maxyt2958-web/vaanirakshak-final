"""
VaniRakshak — Forensic Signal Processing Layer
Powered directly by open-source libraries: librosa, soundfile, scipy, numpy, and torch.
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice retention on disk).

Capabilities:
1. Standardizes audio tensors to 16 kHz mono float32 using soundfile, librosa & torchaudio.
2. Extracts 8–14 Hz vocal micro-tremor using scipy.signal (Hilbert envelope & Welch PSD).
3. Computes spectral flatness (Wiener entropy) using librosa.feature.spectral_flatness.
4. Computes Zero-Crossing Rate (ZCR) and temporal variance using librosa.feature.zero_crossing_rate & numpy.
5. Pinpoints vocoder phase boundaries and splice discontinuities using librosa STFT phase & peak picking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch

from backend.app.core.audio import AudioProcessor, compute_audio_sha256

logger = logging.getLogger("vanirakshak.forensic_signal")
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class MicroTremorResult:
    tremor_ratio: float
    modulation_index: float
    peak_frequency_hz: float
    is_physiological: bool
    anomaly_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralFlatnessResult:
    mean_flatness: float
    std_flatness: float
    high_band_flatness: float
    flatness_variance: float
    anomaly_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZCRResult:
    zcr_mean: float
    zcr_variance: float
    zcr_std: float
    zcr_max_jump: float
    anomaly_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseBoundaryResult:
    boundary_timestamps: List[float]
    boundary_count: int
    mean_phase_acceleration: float
    max_phase_acceleration: float
    boundary_discontinuity_score: float
    is_vocoder_or_spliced: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicWindowTelemetry:
    micro_tremor: MicroTremorResult
    spectral_flatness: SpectralFlatnessResult
    zcr: ZCRResult
    phase_boundary: PhaseBoundaryResult
    composite_vocoder_score: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "micro_tremor": self.micro_tremor.to_dict(),
            "spectral_flatness": self.spectral_flatness.to_dict(),
            "zcr": self.zcr.to_dict(),
            "phase_boundary": self.phase_boundary.to_dict(),
            "composite_vocoder_score": self.composite_vocoder_score,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ForensicStreamReport:
    audio_sha256: str
    duration_sec: float
    sample_rate: int
    mean_micro_tremor_ratio: float
    mean_spectral_flatness: float
    mean_high_band_flatness: float
    mean_zcr_variance: float
    vocoder_phase_discontinuities_detected: int
    detected_boundary_timestamps: List[float]
    composite_vocoder_risk_score: float
    verdict_indicator: str
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ForensicSignalProcessor:
    """Forensic signal processor built on open-source audio toolkits (librosa, soundfile, scipy, numpy)."""

    def __init__(self, target_sr: int = DEFAULT_SAMPLE_RATE) -> None:
        self.target_sr = target_sr
        self.audio_processor = AudioProcessor(target_sr=target_sr)

    def standardize_audio(
        self,
        audio_data: Union[bytes, np.ndarray, torch.Tensor, Path, str],
        orig_sr: Optional[int] = None,
        peak: float = 0.95,
        remove_dc: bool = True,
    ) -> Tuple[torch.Tensor, np.ndarray, str, float]:
        """Standardize arbitrary audio to 16 kHz mono float32 tensor and numpy array."""
        return self.audio_processor.standardize_to_16k_mono(
            audio=audio_data, orig_sr=orig_sr, peak=peak, remove_dc=remove_dc
        )

    def compute_micro_tremor(self, waveform: np.ndarray, sr: Optional[int] = None) -> MicroTremorResult:
        """Extract physiological 8–14 Hz vocal micro-tremor using scipy.signal."""
        sample_rate = sr or self.target_sr
        y = np.asarray(waveform, dtype=np.float32).ravel()

        if len(y) < int(0.2 * sample_rate):
            return MicroTremorResult(0.0, 0.0, 0.0, False, 0.5)

        # 1. Analytic amplitude envelope via scipy.signal.hilbert
        env = np.abs(signal.hilbert(y))
        ds_factor = max(1, sample_rate // 200)
        env_ds = signal.decimate(env, ds_factor, ftype="iir")
        fs_env = sample_rate / float(ds_factor)

        # 2. Welch power spectral density via scipy.signal.welch
        freqs, psd = signal.welch(signal.detrend(env_ds), fs=fs_env, nperseg=min(len(env_ds), 256))
        tremor_mask = (freqs >= 8.0) & (freqs <= 14.0)
        base_mask = (freqs >= 2.0) & (freqs <= 30.0)

        tremor_p = float(np.sum(psd[tremor_mask]))
        base_p = float(np.sum(psd[base_mask])) + 1e-9
        tremor_ratio = float(tremor_p / base_p)

        peak_freq = float(freqs[tremor_mask][np.argmax(psd[tremor_mask])]) if np.any(tremor_mask) else 0.0
        mod_index = float(np.std(env_ds) / (np.mean(env) + 1e-9))

        is_physiological = (0.04 <= tremor_ratio <= 0.28) and (8.0 <= peak_freq <= 14.0)
        anomaly_score = 0.05 if is_physiological else float(np.clip(1.0 - (tremor_ratio / 0.04), 0.3, 0.95))

        return MicroTremorResult(
            tremor_ratio=round(tremor_ratio, 4),
            modulation_index=round(mod_index, 4),
            peak_frequency_hz=round(peak_freq, 2),
            is_physiological=is_physiological,
            anomaly_score=round(anomaly_score, 4),
        )

    def compute_spectral_flatness(
        self, waveform: np.ndarray, sr: Optional[int] = None, n_fft: int = 1024, hop_length: int = 256
    ) -> SpectralFlatnessResult:
        """Compute spectral flatness (Wiener entropy) using librosa.feature.spectral_flatness."""
        sample_rate = sr or self.target_sr
        y = np.asarray(waveform, dtype=np.float32).ravel()

        if len(y) < n_fft:
            return SpectralFlatnessResult(0.0, 0.0, 0.0, 0.0, 0.0)

        # 1. Global Wiener entropy via librosa
        sfm = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        mean_sfm = float(np.mean(sfm))
        std_sfm = float(np.std(sfm))
        var_sfm = float(np.var(sfm))

        # 2. High-band spectral flatness (> 4 kHz vocoder artifact zone) via librosa STFT
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        high_S = S[freqs >= 4000, :]

        high_sfm = np.exp(np.mean(np.log(high_S + 1e-9), axis=0)) / (np.mean(high_S, axis=0) + 1e-9)
        mean_high_sfm = float(np.mean(high_sfm))

        high_sfm_anomaly = float(np.clip((mean_high_sfm - 0.25) / 0.35, 0.0, 1.0))
        global_sfm_anomaly = float(np.clip((mean_sfm - 0.015) / 0.045, 0.0, 1.0))
        anomaly_score = float(np.clip(0.70 * high_sfm_anomaly + 0.30 * global_sfm_anomaly, 0.0, 1.0))

        return SpectralFlatnessResult(
            mean_flatness=round(mean_sfm, 6),
            std_flatness=round(std_sfm, 6),
            high_band_flatness=round(mean_high_sfm, 6),
            flatness_variance=round(var_sfm, 8),
            anomaly_score=round(anomaly_score, 4),
        )

    def compute_zcr_variance(
        self, waveform: np.ndarray, sr: Optional[int] = None, frame_length: int = 1024, hop_length: int = 256
    ) -> ZCRResult:
        """Compute Zero-Crossing Rate & temporal variance using librosa.feature.zero_crossing_rate."""
        y = np.asarray(waveform, dtype=np.float32).ravel()

        if len(y) < frame_length:
            return ZCRResult(0.0, 0.0, 0.0, 0.0, 0.0)

        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_var = float(np.var(zcr))
        zcr_std = float(np.std(zcr))
        zcr_diffs = np.abs(np.diff(zcr))
        zcr_max_jump = float(np.max(zcr_diffs)) if len(zcr_diffs) > 0 else 0.0

        anomaly_score = float(np.clip(zcr_max_jump * 3.5, 0.0, 1.0))

        return ZCRResult(
            zcr_mean=round(zcr_mean, 6),
            zcr_variance=round(zcr_var, 8),
            zcr_std=round(zcr_std, 6),
            zcr_max_jump=round(zcr_max_jump, 6),
            anomaly_score=round(anomaly_score, 4),
        )

    def detect_vocoder_phase_boundaries(
        self, waveform: np.ndarray, sr: Optional[int] = None, hop_length: int = 256, n_fft: int = 1024
    ) -> PhaseBoundaryResult:
        """Detect vocoder phase boundaries using librosa STFT phase differences & scipy.signal.find_peaks."""
        sample_rate = sr or self.target_sr
        y = np.asarray(waveform, dtype=np.float32).ravel()

        if len(y) < 2 * n_fft:
            return PhaseBoundaryResult([], 0, 0.0, 0.0, 0.0, False)

        # 1. Instantaneous phase acceleration via librosa.stft
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        phase = np.unwrap(np.angle(D), axis=1)
        phase_acc = np.mean(np.abs(np.diff(phase, n=2, axis=1)), axis=0)
        mean_acc = float(np.mean(phase_acc))
        max_acc = float(np.max(phase_acc)) if len(phase_acc) > 0 else 0.0

        # 2. Corroborating gradients: librosa spectral flatness & zero-crossing rate
        sfm = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        speech_mask = rms > (np.max(rms) * 0.05) if np.max(rms) > 1e-6 else np.ones_like(rms, dtype=bool)

        grad_sfm = np.pad(np.abs(np.diff(sfm)), (1, 0))
        grad_zcr = np.pad(np.abs(np.diff(zcr)), (1, 0))
        phase_acc_pad = np.pad(phase_acc, (1, 1))

        # Joint boundary curve during active voiced speech
        boundary_curve = (grad_sfm * 5.0) + (grad_zcr * 2.0) + (phase_acc_pad / 3.0)
        boundary_curve_speech = boundary_curve * speech_mask

        # 3. Peak detection via scipy.signal.find_peaks (at least 160ms apart)
        thresh = float(np.mean(boundary_curve_speech) + 3.0 * np.std(boundary_curve_speech))
        peaks, _ = signal.find_peaks(boundary_curve_speech, height=thresh, distance=10)
        timestamps = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=hop_length)
        boundary_timestamps = [round(float(t), 3) for t in timestamps]

        boundary_count = len(boundary_timestamps)
        discontinuity_score = float(np.clip(boundary_count / 4.0, 0.0, 1.0))
        is_vocoder_or_spliced = boundary_count > 0 or max_acc > 1.85

        return PhaseBoundaryResult(
            boundary_timestamps=boundary_timestamps,
            boundary_count=boundary_count,
            mean_phase_acceleration=round(mean_acc, 4),
            max_phase_acceleration=round(max_acc, 4),
            boundary_discontinuity_score=round(discontinuity_score, 4),
            is_vocoder_or_spliced=is_vocoder_or_spliced,
        )

    def analyze_window(self, samples: np.ndarray, sr: Optional[int] = None) -> ForensicWindowTelemetry:
        """Analyze a single 2.0s sliding window (< 10 ms execution time)."""
        t0 = time.perf_counter()
        sample_rate = sr or self.target_sr
        y = np.asarray(samples, dtype=np.float32).ravel()

        tremor = self.compute_micro_tremor(y, sr=sample_rate)
        flatness = self.compute_spectral_flatness(y, sr=sample_rate)
        zcr_res = self.compute_zcr_variance(y, sr=sample_rate)
        phase_res = self.detect_vocoder_phase_boundaries(y, sr=sample_rate)

        composite = float(
            np.clip(
                0.50 * flatness.anomaly_score
                + 0.25 * phase_res.boundary_discontinuity_score
                + 0.15 * tremor.anomaly_score
                + 0.10 * zcr_res.anomaly_score,
                0.0,
                1.0,
            )
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ForensicWindowTelemetry(
            micro_tremor=tremor,
            spectral_flatness=flatness,
            zcr=zcr_res,
            phase_boundary=phase_res,
            composite_vocoder_score=round(composite, 4),
            latency_ms=round(latency_ms, 2),
        )

    def analyze_stream(
        self, audio_input: Union[bytes, np.ndarray, torch.Tensor, Path, str], orig_sr: Optional[int] = None
    ) -> ForensicStreamReport:
        """Process full audio stream using open-source DSP toolchain."""
        t0 = time.perf_counter()
        _, np_mono, sha256, duration_sec = self.standardize_audio(audio_input, orig_sr=orig_sr)

        tremor = self.compute_micro_tremor(np_mono, sr=self.target_sr)
        flatness = self.compute_spectral_flatness(np_mono, sr=self.target_sr)
        zcr_res = self.compute_zcr_variance(np_mono, sr=self.target_sr)
        phase_res = self.detect_vocoder_phase_boundaries(np_mono, sr=self.target_sr)

        composite_risk = float(
            np.clip(
                0.50 * flatness.anomaly_score
                + 0.30 * phase_res.boundary_discontinuity_score
                + 0.15 * tremor.anomaly_score
                + 0.05 * zcr_res.anomaly_score,
                0.0,
                1.0,
            )
        )

        if composite_risk >= 0.60:
            verdict = "SYNTHETIC_VOCODER_DETECTED"
        elif phase_res.boundary_count >= 1 and (composite_risk >= 0.25 or flatness.high_band_flatness > 0.35):
            verdict = "SUSPICIOUS_PHASE_DISCONTINUITY"
        elif composite_risk >= 0.40:
            verdict = "SUSPICIOUS_HIGH_FREQUENCY_FLATNESS"
        else:
            verdict = "NATURAL_ACOUSTIC_PROFILE"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ForensicStreamReport(
            audio_sha256=sha256,
            duration_sec=round(duration_sec, 3),
            sample_rate=self.target_sr,
            mean_micro_tremor_ratio=tremor.tremor_ratio,
            mean_spectral_flatness=flatness.mean_flatness,
            mean_high_band_flatness=flatness.high_band_flatness,
            mean_zcr_variance=zcr_res.zcr_variance,
            vocoder_phase_discontinuities_detected=phase_res.boundary_count,
            detected_boundary_timestamps=phase_res.boundary_timestamps,
            composite_vocoder_risk_score=round(composite_risk * 100.0, 2),
            verdict_indicator=verdict,
            total_latency_ms=round(latency_ms, 2),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VaniRakshak — Forensic Signal Processing CLI (librosa, soundfile, scipy, numpy)"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file to evaluate")
    parser.add_argument("--json", action="store_true", help="Output JSON telemetry")
    parser.add_argument("--output", type=str, default=None, help="Save report or standardized WAV")

    args = parser.parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    processor = ForensicSignalProcessor()
    report = processor.analyze_stream(audio_path)

    if args.output and args.output.endswith(".json"):
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
    elif args.output and args.output.endswith(".wav"):
        _, np_mono, _, _ = processor.standardize_audio(audio_path)
        sf.write(args.output, np_mono, DEFAULT_SAMPLE_RATE)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print("\n" + "=" * 66)
    print("  🛡️  VANIRAKSHAK FORENSIC SIGNAL PROCESSING TELEMETRY")
    print("=" * 66)
    print(f" Target File:         {audio_path.name}")
    print(f" SHA-256 Digest:      {report.audio_sha256[:16]}...{report.audio_sha256[-8:]}")
    print(f" Duration:            {report.duration_sec:.2f}s @ {report.sample_rate} Hz mono")
    print(f" Processing Latency:  {report.total_latency_ms:.2f} ms")
    print("-" * 66)
    print(f" 8–14 Hz Micro-Tremor:   {report.mean_micro_tremor_ratio:.4f} (Lippold physiological band)")
    print(f" Global Spectral Flat:   {report.mean_spectral_flatness:.6f} (Wiener entropy)")
    print(f" High-Band Flat (>4kHz): {report.mean_high_band_flatness:.6f} (Vocoder artifact band)")
    print(f" ZCR Temporal Variance:  {report.mean_zcr_variance:.8f}")
    print(f" Phase Discontinuities:  {report.vocoder_phase_discontinuities_detected} detected")
    if report.detected_boundary_timestamps:
        print(f" Boundary Timestamps:    {report.detected_boundary_timestamps} sec")
    print("-" * 66)
    print(f" Vocoder Risk Score:     {report.composite_vocoder_risk_score:.1f}%")
    print(f" Verdict Indicator:      {report.verdict_indicator}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
