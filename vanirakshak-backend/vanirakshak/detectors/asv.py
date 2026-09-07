"""Speaker verification stub (ECAPA-TDNN-shaped).

Maintains a per-user enrolled 192-dim "embedding" (just a 192-dim
deterministic summary of the reference audio). For a live chunk we
compute the same summary, then cosine similarity.

The 192-dim vector is computed from a fixed random projection of
hand-crafted spectral statistics. The projection is **stable across
runs** (seeded by the user_id hash), so the same user's reference
audio always embeds to a similar vector regardless of run.

Replace :meth:`ASVModel.embed` with a real ECAPA-TDNN forward pass to
upgrade. The contract (192-dim float32 vector, L2-normalised) will
not change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


EMBED_DIM = 192


def _stable_projection(seed_str: str, dim_in: int, dim_out: int = EMBED_DIM) -> np.ndarray:
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, dim_out)).astype(np.float32)
    return p


def _spectral_features(x: np.ndarray, sr: int, n_bins: int = 32) -> np.ndarray:
    """Return an n_bins-dim log-mel-like summary (hand-crafted, no torchaudio)."""
    from scipy.signal import welch

    f, pxx = welch(x, fs=sr, nperseg=min(1024, x.size), noverlap=256)
    if pxx.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    # log-spaced bins
    edges = np.logspace(np.log10(50.0), np.log10(sr / 2.0), n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = float(np.log(np.mean(pxx[m]) + 1e-12))
    # standardise per-clip
    out = (out - out.mean()) / (out.std() + 1e-6)
    return out


@dataclass
class ASVModel:
    """Enrolment + verification."""

    name: str = "synthetic_asv_v1"
    _enrolments: Dict[str, np.ndarray] = field(default_factory=dict)

    def _embed_raw(self, x: np.ndarray, sr: int, user_id: str) -> np.ndarray:
        feats = _spectral_features(x, sr, n_bins=EMBED_DIM)
        proj = _stable_projection(user_id or "anon", EMBED_DIM, EMBED_DIM)
        v = feats @ proj
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(np.float32)

    def embed(self, x: np.ndarray, sr: int, user_id: str = "anon") -> np.ndarray:
        return self._embed_raw(x, sr, user_id)

    def enrol(self, user_id: str, x: np.ndarray, sr: int) -> np.ndarray:
        emb = self._embed_raw(x, sr, user_id)
        self._enrolments[user_id] = emb
        return emb

    def is_enrolled(self, user_id: str) -> bool:
        return user_id in self._enrolments

    def similarity(self, user_id: str, x: np.ndarray, sr: int) -> float:
        if user_id not in self._enrolments:
            return 0.0
        enrolled = self._enrolments[user_id]
        live = self._embed_raw(x, sr, user_id)
        return float(np.dot(enrolled, live) / (np.linalg.norm(enrolled) * np.linalg.norm(live) + 1e-9))

    def consistency_score(self, user_id: str, x: np.ndarray, sr: int) -> Tuple[float, bool]:
        """Return (consistency in [0,1], is_enrolled)."""
        if not self.is_enrolled(user_id):
            return 0.5, False
        cos = self.similarity(user_id, x, sr)
        # map cosine in [-1, 1] to [0, 1] with a soft clip around 0.5
        return float((cos + 1.0) / 2.0), True
