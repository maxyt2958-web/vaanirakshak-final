"""Synthetic audio generators for demo & tests (R1 stand-in).

Honest disclaimer: these signals are **synthetic** — they are not
voice samples. They are designed so that, for a CM model that uses
the four standard features below, genuine and spoofed produce
clearly separated feature values.

Features (and the values we engineer):
    hb_ratio    = energy(4-8 kHz) / energy(0-4 kHz)
                  genuine  ≈ 0.5  (formants extend high)
                  spoofed  ≈ 0.0  (vocoder bandlimits to < 3.5 kHz)

    flatness    = geometric-mean(PSD) / arithmetic-mean(PSD)
                  genuine  ≈ 0.05 (peaky harmonic spectrum)
                  spoofed  ≈ 0.6  (noise-like vocoder residue)

    energy_var  = variance of frame-energies
                  genuine  ≈ 0.04 (syllabic envelope)
                  spoofed  ≈ 0.001 (regular vocoder envelope)

    jitter_resid = cv of zero-crossing intervals in 300-1500 Hz band
                  genuine  ≈ 0.10 (natural pitch variation)
                  spoofed  ≈ 0.005 (perfectly periodic vocoder)
"""

from __future__ import annotations

import numpy as np


def synthetic_speech(
    duration_sec: float = 4.0,
    sr: int = 16000,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Broadband harmonic signal with formants up to 4 kHz, syllabic
    envelope, jittered pitch, and broadband breath noise — engineered
    to look natural to the CM model.
    """
    rng = rng or np.random.default_rng()
    n = int(round(duration_sec * sr))
    t = np.arange(n) / sr

    from scipy.signal import butter, sosfilt

    # 1) Harmonic stack (formants) up to ~4 kHz
    f0 = 130.0
    # Add a strong 350 Hz component (within 300-1500 Hz prosody band)
    # with jittered phase — this gives the prosody detector the cue it needs.
    phase_jitter = np.cumsum(rng.uniform(0.95, 1.05, size=n) / f0) * 2 * np.pi
    harmonics = (
        0.5 * np.sin(2 * np.pi * 350.0 * t + phase_jitter)
        + 0.4 * np.sin(2 * np.pi * 700.0 * t + phase_jitter * 2)
        + 0.3 * np.sin(2 * np.pi * 1500.0 * t + phase_jitter * 3)
        + 0.2 * np.sin(2 * np.pi * 2400.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    ).astype(np.float32)

    # 2) jittered pitch pulses (drives the jitter cue)
    n_pulses = int(duration_sec * f0) + 4
    periods = rng.uniform(1.0 / f0 * 0.93, 1.0 / f0 * 1.07, size=n_pulses)
    edges = (np.cumsum(periods) * sr).astype(np.int64)
    edges = edges[edges < n]
    pulse = np.zeros(n, dtype=np.float32)
    pulse[edges] = 1.0
    sos_p = butter(2, 800.0 / (sr / 2), btype="low", output="sos")
    pitch = sosfilt(sos_p, pulse).astype(np.float32) * 0.6

    sig = harmonics * 0.4 + pitch

    # 3) syllabic envelope (4 Hz AM with random pauses)
    env = 0.55 + 0.35 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))
    n_breaks = max(2, int(n / sr * 0.7))
    for s in rng.choice(n, size=n_breaks, replace=False):
        length = int(rng.uniform(0.04, 0.12) * sr)
        env[s : min(s + length, n)] *= rng.uniform(0.1, 0.3)
    sig = sig * env.astype(np.float32)

    # 4) breath / room noise (broadband but low amplitude)
    noise = rng.normal(0, 0.08, size=n).astype(np.float32)
    sig = sig + noise

    return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


def synthetic_spoofed(
    duration_sec: float = 4.0,
    sr: int = 16000,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A vocoded-looking signal: lowpass at 3.4 kHz, perfectly periodic
    150 Hz pulse train, flat noise residue, regular 4 Hz amplitude
    envelope. Designed to look like a TTS output that has been
    pushed through a PSTN codec.
    """
    rng = rng or np.random.default_rng()
    n = int(round(duration_sec * sr))
    t = np.arange(n) / sr

    from scipy.signal import butter, sosfilt

    # 1) Perfectly periodic pulse train (zero jitter) at f0 = 150 Hz,
    #    but in the form of a 150 Hz sine (rather than smoothed pulses)
    #    so the prosody bandpass can see it.
    f0 = 150.0
    t = np.arange(n) / sr
    pitch = (0.4 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    # harmonics to give it some body, all multiples of 150 Hz
    for k in (2, 3, 4):
        pitch += (0.2 / k) * np.sin(2 * np.pi * f0 * k * t)
    pitch = pitch.astype(np.float32)

    # 2) flat noise residue (the vocoder tell)
    residue = rng.normal(0, 0.15, size=n).astype(np.float32)
    # bandlimit to 3.4 kHz (classic PSTN)
    sos_b = butter(6, 3400.0 / (sr / 2), btype="low", output="sos")
    pitch = sosfilt(sos_b, pitch).astype(np.float32)
    residue = sosfilt(sos_b, residue).astype(np.float32)

    sig = pitch + residue

    # 3) very regular 4 Hz amplitude envelope
    env = 0.55 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    sig = sig * env.astype(np.float32)

    return (sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


def synthetic_silence(duration_sec: float = 4.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)
