"""DPDP Act 2023 compliant in-memory audio buffer.

Raw audio frames live ONLY in volatile RAM. The buffer exposes a
``read_window()`` that yields the current 4.04 s window (the AASIST
receptive field) without ever serialising it to disk.

This is the **single point of audio truth** in the system; the rest
of the pipeline only ever sees numpy arrays that fall out of scope.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from ..config import AudioConfig, settings


class SlidingAudioBuffer:
    """FIFO ring of PCM samples, indexed by hop.

    PCM is treated as 16-bit signed mono. The buffer holds **at most
    ``window_samples + hop_samples`` samples** — i.e. one full analysis
    window plus the next chunk that may trigger a hop. Older samples
    are overwritten in place (DPDP §8(7) compliant: no raw audio ever
    sits in persistent storage).
    """

    def __init__(self, cfg: AudioConfig | None = None) -> None:
        self.cfg = cfg or settings.audio
        self.window_samples = int(round(self.cfg.window_seconds * self.cfg.sample_rate))
        self.hop_samples = int(round(self.cfg.hop_seconds * self.cfg.sample_rate))
        capacity = self.window_samples + self.hop_samples
        self._buf: np.ndarray = np.zeros(capacity, dtype=np.int16)
        self._write_pos = 0
        self._filled = 0  # number of valid samples in the ring
        self._samples_since_hop = 0

    @property
    def total_samples(self) -> int:
        return self._filled

    def push(self, pcm: np.ndarray) -> bool:
        """Append a chunk of int16/float32 mono PCM.

        Returns ``True`` if a new window of size ``window_samples``
        is now available to read.
        """
        if pcm.dtype == np.float32 or pcm.dtype == np.float64:
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
        else:
            pcm = pcm.astype(np.int16, copy=False)

        n = len(pcm)
        cap = self._buf.size

        # wrap-aware copy
        end = self._write_pos + n
        if end <= cap:
            self._buf[self._write_pos : end] = pcm
        else:
            first = cap - self._write_pos
            self._buf[self._write_pos : cap] = pcm[:first]
            self._buf[: n - first] = pcm[first:]
        self._write_pos = (self._write_pos + n) % cap
        self._filled = min(self._filled + n, cap)
        self._samples_since_hop += n

        if self._samples_since_hop >= self.hop_samples and self._filled >= self.window_samples:
            self._samples_since_hop = 0
            return True
        return False

    def read_window(self) -> np.ndarray | None:
        """Return the current window as a fresh numpy copy, or ``None``
        if we have not accumulated at least ``window_samples`` yet.
        """
        if self._filled < self.window_samples:
            return None
        cap = self._buf.size
        start = (self._write_pos - self.window_samples) % cap
        if start + self.window_samples <= cap:
            return self._buf[start : start + self.window_samples].copy()
        first = cap - start
        return np.concatenate([self._buf[start:cap], self._buf[: self.window_samples - first]]).copy()

    def wipe(self) -> None:
        """Defensive wipe — for tests, and for the "drop the call" path."""
        self._buf.fill(0)
        self._write_pos = 0
        self._filled = 0
        self._samples_since_hop = 0

    def __repr__(self) -> str:
        return (
            f"SlidingAudioBuffer(window={self.window_samples} samples, "
            f"hop={self.hop_samples} samples, buffered={self._filled}/{self._buf.size})"
        )
