"""Prosody & micro-tremor consistency (Layer 3 of the multi-layer defence).

Bandpasses the signal to 300-1500 Hz (the pitch region for adult
speech), then extracts zero-crossing intervals. Vocoders regularise
the pitch ⇒ very low CV. Natural speech has CV ~ 0.05-0.15.

``unnatural`` ∈ [0, 1] — higher means the prosody looks more
vocoded (low jitter, low variance).
"""

from __future__ import annotations

import numpy as np


def prosody_score(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size < sr:
        return {"unnatural": 0.5, "pitch_var": 0.0, "jitter": 0.0}

    # bandpass to the pitch region
    from scipy.signal import butter, sosfilt
    lo = max(300.0 / (sr / 2), 0.001)
    hi = min(1500.0 / (sr / 2), 0.999)
    if lo >= hi:
        return {"unnatural": 0.5, "pitch_var": 0.0, "jitter": 0.0}
    try:
        sos = butter(4, [lo, hi], btype="band", output="sos")
        xb = sosfilt(sos, x).astype(np.float32)
    except Exception:
        xb = x

    zc = np.where(np.diff(np.sign(xb)))[0]
    if zc.size < 6:
        return {"unnatural": 0.5, "pitch_var": 0.0, "jitter": 0.0}

    intervals = np.diff(zc).astype(np.float32)
    pitch = sr / intervals
    # only keep plausible pitch
    pitch = pitch[(pitch > 50) & (pitch < 500)]
    if pitch.size < 4:
        return {"unnatural": 0.5, "pitch_var": 0.0, "jitter": 0.0}

    pitch_var = float(np.var(pitch) / (np.mean(pitch) + 1e-9))
    jitter = float(np.std(intervals) / (np.mean(intervals) + 1e-9))

    # vocoders ⇒ low var + low jitter; natural ⇒ both > 0
    # we map (pitch_var, jitter) to [0, 1] with tanh so even moderate
    # values map to ~0.5, and very low (vocoder) maps to 1.0
    cv_combined = 3.0 * pitch_var + 2.0 * jitter
    unnatural = 1.0 - float(np.tanh(cv_combined))
    return {"unnatural": max(0.0, min(1.0, unnatural)), "pitch_var": pitch_var, "jitter": jitter}
