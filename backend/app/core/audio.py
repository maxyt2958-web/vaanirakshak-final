"""
VaniRakshak — In-Memory Forensic Audio Processor & Tensor Standardizer
Compliant with DPDP Act 2023 Section 8(7) (Zero raw voice retention on disk).

Handles in-memory PCM decoding, multi-channel downmixing, standardization to 16 kHz mono,
peak normalization, DC offset removal, and cryptographic SHA-256 fingerprinting for
chain-of-custody audit tracking.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger("vanirakshak.audio")

DEFAULT_SAMPLE_RATE = 16000


def compute_audio_sha256(audio_data: Union[bytes, np.ndarray, torch.Tensor]) -> str:
    """Compute SHA-256 hash over in-memory audio data for immutable audit logging."""
    if isinstance(audio_data, bytes):
        raw_bytes = audio_data
    elif isinstance(audio_data, np.ndarray):
        raw_bytes = audio_data.astype(np.float32).tobytes()
    elif isinstance(audio_data, torch.Tensor):
        raw_bytes = audio_data.detach().cpu().to(torch.float32).numpy().tobytes()
    else:
        raise TypeError(f"Unsupported audio data type for hashing: {type(audio_data)}")

    return hashlib.sha256(raw_bytes).hexdigest()


class AudioProcessor:
    """Zero-retention forensic audio processor and standardizer."""

    def __init__(self, target_sr: int = DEFAULT_SAMPLE_RATE) -> None:
        self.target_sr = target_sr
        self._resamplers: dict[int, T.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
        return self._resamplers[orig_sr]

    def decode_bytes(
        self,
        audio_bytes: bytes,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, str, float]:
        """Decode raw in-memory audio bytes to a 1D float32 numpy array.
        
        Zero disk I/O occurs during this process.
        
        Returns:
            (waveform_1d, sha256_hash, duration_seconds)
        """
        if not audio_bytes:
            raise ValueError("Audio bytes buffer is empty.")

        sr = target_sr or self.target_sr
        sha256 = hashlib.sha256(audio_bytes).hexdigest()

        # Attempt decoding with soundfile first (handles WAV, FLAC, OGG, etc.)
        try:
            buffer = io.BytesIO(audio_bytes)
            data, orig_sr = sf.read(buffer, dtype="float32", always_2d=True)
            # Mixdown multi-channel to mono
            if data.shape[1] > 1:
                mono = np.mean(data, axis=1)
            else:
                mono = data[:, 0]
        except Exception as sf_err:
            # Fallback to torchaudio buffer decoding (handles MP3, AAC, etc.)
            try:
                buffer = io.BytesIO(audio_bytes)
                tensor_data, orig_sr = torchaudio.load(buffer)
                if tensor_data.shape[0] > 1:
                    tensor_mono = torch.mean(tensor_data, dim=0)
                else:
                    tensor_mono = tensor_data.squeeze(0)
                mono = tensor_mono.numpy().astype(np.float32)
            except Exception as ta_err:
                raise ValueError(
                    f"Failed to decode in-memory audio format: sf_err='{sf_err}', ta_err='{ta_err}'"
                ) from ta_err

        # Resample in-memory if needed
        if orig_sr != sr:
            tensor_mono = torch.from_numpy(mono).unsqueeze(0)
            resampler = self._get_resampler(orig_sr)
            tensor_mono = resampler(tensor_mono).squeeze(0)
            mono = tensor_mono.numpy().astype(np.float32)

        # Normalize audio signal
        mono = self.normalize(mono)
        duration = len(mono) / float(sr)

        return mono, sha256, duration

    def standardize_to_16k_mono(
        self,
        audio: Union[bytes, np.ndarray, torch.Tensor, Path, str],
        orig_sr: Optional[int] = None,
        peak: float = 0.95,
        remove_dc: bool = True,
    ) -> Tuple[torch.Tensor, np.ndarray, str, float]:
        """Standardize arbitrary audio input to 16 kHz mono float32 tensor and numpy array.

        Strictly compliant with DPDP Act 2023 Section 8(7) (Zero raw disk retention).

        Args:
            audio: Audio input (bytes buffer, numpy array, torch Tensor, or file Path/str).
            orig_sr: Sampling rate of input if audio is already a Tensor or ndarray.
                     Defaults to 16000 if unspecified.
            peak: Target peak amplitude for normalization (default: 0.95).
            remove_dc: Whether to remove DC bias / baseline offset.

        Returns:
            (tensor_1d, numpy_1d, sha256_hash, duration_sec):
                - tensor_1d: torch.Tensor of shape (1, N) on CPU with dtype torch.float32
                - numpy_1d: np.ndarray of shape (N,) with dtype np.float32
                - sha256_hash: Cryptographic audit fingerprint of the standardized waveform
                - duration_sec: Length of standardized audio in seconds
        """
        # Case 1: Path or filename
        if isinstance(audio, (Path, str)):
            p = Path(audio)
            if not p.is_file():
                raise FileNotFoundError(f"Audio file not found: {p}")
            audio_bytes = p.read_bytes()
            np_mono, sha256, duration = self.decode_bytes(audio_bytes, target_sr=self.target_sr)
            tensor_mono = torch.from_numpy(np_mono).unsqueeze(0)
            return tensor_mono, np_mono, sha256, duration

        # Case 2: In-memory raw bytes buffer
        if isinstance(audio, bytes):
            np_mono, sha256, duration = self.decode_bytes(audio, target_sr=self.target_sr)
            tensor_mono = torch.from_numpy(np_mono).unsqueeze(0)
            return tensor_mono, np_mono, sha256, duration

        # Case 3: torch.Tensor or numpy.ndarray
        source_sr = orig_sr or self.target_sr

        if isinstance(audio, torch.Tensor):
            t = audio.detach().cpu().to(torch.float32)
            # Remove batch dimension if shape is (1, C, T) or (1, T)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            # Multi-channel downmix to mono
            if t.ndim == 2:
                # If shape is (C, T) with C <= 8, average across C
                if t.shape[0] <= 8 and t.shape[1] > t.shape[0]:
                    t = torch.mean(t, dim=0)
                else:
                    # Shape is (T, C)
                    t = torch.mean(t, dim=1)
            t = t.ravel()
            np_arr = t.numpy()
        elif isinstance(audio, np.ndarray):
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.ndim == 2:
                if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                    arr = np.mean(arr, axis=0)
                else:
                    arr = np.mean(arr, axis=1)
            np_arr = arr.ravel()
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Resample if sample rate does not match target 16 kHz
        if source_sr != self.target_sr:
            tensor_in = torch.from_numpy(np_arr).unsqueeze(0)
            resampler = self._get_resampler(source_sr)
            tensor_out = resampler(tensor_in).squeeze(0)
            np_arr = tensor_out.numpy().astype(np.float32)

        # Remove DC offset if requested
        if remove_dc and len(np_arr) > 0:
            np_arr = np_arr - np.mean(np_arr)

        # Peak normalization
        np_arr = self.normalize(np_arr, peak=peak)

        # Create standardized torch.Tensor (1, N)
        tensor_standard = torch.from_numpy(np_arr).unsqueeze(0)

        # Cryptographic fingerprint
        sha256 = compute_audio_sha256(np_arr)
        duration = len(np_arr) / float(self.target_sr)

        return tensor_standard, np_arr, sha256, duration

    @staticmethod
    def normalize(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """Peak-normalize audio waveform to prevent digital clipping."""
        x = np.asarray(waveform, dtype=np.float32).ravel()
        max_val = np.max(np.abs(x))
        if max_val > 1e-6:
            x = (x / max_val) * peak
        return x.astype(np.float32)
