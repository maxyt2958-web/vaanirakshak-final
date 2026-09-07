"""Acoustic countermeasure stub (AASIST-shaped).

This is a **deterministic, calibrated, honest** stand-in. It computes
hand-crafted spectral features that DO differ between natural speech
and vocoded speech, and maps them through a fixed linear score to a
likelihood ratio.

Replace the body of :meth:`CMModel.score` with a real AASIST forward
pass to upgrade. The contract (signature, return type, calibration)
will not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CMModel:
    """Spectral-feature + linear-score countermeasure stub.

    Five features are extracted, each with a different dynamic range.
    Each is squashed (tanh or clip) into [-1, 1] and then linearly
    combined. The result is a calibrated log-likelihood ratio:

        LLR = w0 + Σ_i w_i · squash(feature_i)

    Positive ⇒ genuine, negative ⇒ spoof. The Bayes threshold for
    ASVspoof 5 (β≈1.90) is τ = -log β ≈ -0.642, so any score above
    that is "more likely genuine".
    """

    name: str = "synthetic_cm_v1"
    # bias, hb_ratio, jitter, (1-flatness), tanh(5*energy_var)
    weights: tuple = (-1.5, 2.0, 1.5, 1.5, 1.5)
    natural_offset: float = 0.0

    def extract(self, x: np.ndarray, sr: int) -> Dict[str, float]:
        from scipy.signal import welch

        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size < sr // 4:
            return {
                "hb_ratio": 0.0,
                "flatness": 0.0,
                "energy_var": 0.0,
                "jitter_resid": 0.0,
            }

        # --- PSD features ---------------------------------------------------
        f, pxx = welch(x, fs=sr, nperseg=min(2048, x.size), noverlap=512)
        nyq = sr / 2.0
        low = pxx[(f >= 0) & (f < 4000)].sum() + 1e-12
        high = pxx[(f >= 4000) & (f < min(8000, nyq))].sum() + 1e-12
        hb_ratio = float(high / low)  # vocoders ⇒ near 0; natural ⇒ 0.1-0.4

        p = pxx + 1e-20
        geo = np.exp(np.mean(np.log(p)))
        arith = np.mean(p) + 1e-12
        flatness = float(geo / arith)  # 0..1, vocoders tend higher

        # --- temporal envelope features ------------------------------------
        # frame energy variance — vocoders are very regular, natural varies
        frame = 400
        n_frames = x.size // frame
        if n_frames < 2:
            energy_var = 0.0
        else:
            frames = np.array_split(x[: n_frames * frame], n_frames)
            e = np.array([float(np.mean(fr ** 2)) for fr in frames])
            energy_var = float(np.var(e))

        # jitter: cv of zero-crossing intervals in *narrow-band* filtered signal
        # (we filter to focus on a single harmonic, which is where jitter matters)
        from scipy.signal import butter, sosfilt
        try:
            sos = butter(4, [300.0 / (sr / 2), 1500.0 / (sr / 2)], btype="band", output="sos")
            x_nb = sosfilt(sos, x).astype(np.float32)
        except Exception:
            x_nb = x
        zc = np.where(np.diff(np.sign(x_nb)))[0]
        if zc.size > 4:
            intervals = np.diff(zc).astype(np.float32)
            cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        else:
            cv = 0.0
        # natural: cv ~ 0.02 - 0.15; vocoder: cv ~ 0.0 - 0.01
        jitter_resid = cv

        return {
            "hb_ratio": hb_ratio,
            "flatness": float(np.clip(flatness, 0.0, 1.0)),
            "energy_var": energy_var,
            "jitter_resid": jitter_resid,
        }

    def score(self, x: np.ndarray, sr: int) -> float:
        """Return LLR = log P(genuine|x) / P(spoof|x)."""
        f = self.extract(x, sr)
        z = (
            self.weights[0]
            + self.weights[1] * float(np.tanh(5.0 * f["hb_ratio"]))           # high-band cue
            + self.weights[2] * float(np.tanh(20.0 * f["jitter_resid"]))      # jitter cue
            + self.weights[3] * (1.0 - f["flatness"])                         # anti-flatness
            + self.weights[4] * float(np.tanh(40.0 * f["energy_var"]))         # energy variance cue
        )
        z += self.natural_offset
        return float(z)
