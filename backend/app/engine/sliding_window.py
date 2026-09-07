"""
VaniRakshak — 2.0-Second Sliding Window Engine
Slices streaming or recorded audio into 2.0-second acoustic observation windows (32,000 samples @ 16 kHz).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Union

import numpy as np
import torch


@dataclass
class AudioWindow:
    """Represents a single 2.0-second acoustic analysis window."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    samples: np.ndarray  # float32 array of shape (window_samples,)
    is_voiced: bool = True
    speech_ratio: float = 1.0


class SlidingWindowEngine:
    """Generates 2.0-second sliding windows with configurable hop sizes."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_sec: float = 2.0,
        hop_size_sec: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_size_sec = window_size_sec
        self.hop_size_sec = hop_size_sec

        self.window_samples = int(window_size_sec * sample_rate)  # 32,000 samples @ 16kHz
        self.hop_samples = int(hop_size_sec * sample_rate)        # 8,000 samples @ 16kHz

    def slice_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        vad_engine: Optional[Any] = None,
    ) -> List[AudioWindow]:
        """Slice a complete waveform into overlapping 2.0s AudioWindows."""
        if isinstance(waveform, torch.Tensor):
            x = waveform.detach().cpu().to(torch.float32).numpy().ravel()
        else:
            x = np.asarray(waveform, dtype=np.float32).ravel()

        total_samples = len(x)
        windows: List[AudioWindow] = []

        if total_samples < self.window_samples:
            # Zero-pad if total audio is shorter than 2.0s
            padded = np.zeros(self.window_samples, dtype=np.float32)
            padded[:total_samples] = x
            is_voiced = True
            speech_ratio = 1.0
            if vad_engine is not None:
                is_voiced, speech_ratio, _ = vad_engine.detect_speech(padded)
            return [
                AudioWindow(
                    index=0,
                    start_sec=0.0,
                    end_sec=float(total_samples) / self.sample_rate,
                    duration_sec=self.window_size_sec,
                    samples=padded,
                    is_voiced=is_voiced,
                    speech_ratio=speech_ratio,
                )
            ]

        idx = 0
        start = 0
        while start + self.window_samples <= total_samples:
            end = start + self.window_samples
            chunk = x[start:end]
            start_sec = round(start / self.sample_rate, 4)
            end_sec = round(end / self.sample_rate, 4)

            is_voiced = True
            speech_ratio = 1.0
            if vad_engine is not None:
                is_voiced, speech_ratio, _ = vad_engine.detect_speech(chunk)

            windows.append(
                AudioWindow(
                    index=idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=self.window_size_sec,
                    samples=chunk,
                    is_voiced=is_voiced,
                    speech_ratio=speech_ratio,
                )
            )
            idx += 1
            start += self.hop_samples

        return windows


class StreamingWindowBuffer:
    """In-memory sliding window buffer for live streaming audio chunks."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_sec: float = 2.0,
        hop_size_sec: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_size_sec * sample_rate)  # 32,000
        self.hop_samples = int(hop_size_sec * sample_rate)        # 8,000
        self._buffer: List[float] = []
        self._window_idx: int = 0
        self._samples_ingested: int = 0

    def feed_chunk(self, chunk: Union[np.ndarray, List[float]]) -> List[AudioWindow]:
        """Append incoming PCM samples and return any ready 2.0s windows."""
        if isinstance(chunk, np.ndarray):
            arr = chunk.astype(np.float32).ravel().tolist()
        else:
            arr = [float(s) for s in chunk]

        self._buffer.extend(arr)
        self._samples_ingested += len(arr)

        ready_windows: List[AudioWindow] = []

        while len(self._buffer) >= self.window_samples:
            window_data = np.array(self._buffer[: self.window_samples], dtype=np.float32)
            end_sample = self._samples_ingested - (len(self._buffer) - self.window_samples)
            start_sample = end_sample - self.window_samples

            ready_windows.append(
                AudioWindow(
                    index=self._window_idx,
                    start_sec=round(start_sample / self.sample_rate, 4),
                    end_sec=round(end_sample / self.sample_rate, 4),
                    duration_sec=round(self.window_samples / self.sample_rate, 4),
                    samples=window_data,
                )
            )
            self._window_idx += 1
            # Advance buffer by hop_samples
            self._buffer = self._buffer[self.hop_samples :]

        return ready_windows

    def reset(self) -> None:
        """Clear buffer state for zero-retention between calls."""
        self._buffer.clear()
        self._window_idx = 0
        self._samples_ingested = 0
