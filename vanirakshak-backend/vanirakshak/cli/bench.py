"""Latency micro-benchmark: chunk → verdict p50 / p95.

Run::

    python -m vanirakshak bench
"""

from __future__ import annotations

import time

import numpy as np

from ..config import settings
from ..data.synthetic import synthetic_speech
from ..session import CallSession, SessionContext


def run_bench(n_windows: int = 200, sr: int = 16000) -> int:
    sess = CallSession("bench", SessionContext(caller_id="+91-bench", claimed_identity="bench_user"))
    sess.enrol("bench_user", synthetic_speech(4.0, sr=sr), sr)

    # warmup
    for _ in range(5):
        sess.push_chunk(np.zeros(int(sr * 0.5), dtype=np.int16))

    hop = int(sr * settings.audio.hop_seconds)
    latencies_ms: list[float] = []
    for i in range(n_windows):
        chunk = np.random.randint(-2000, 2000, size=hop, dtype=np.int16)
        t0 = time.perf_counter()
        sess.push_chunk(chunk)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(latencies_ms)
    print("=" * 50)
    print(" VaniRakshak pipeline latency (single-thread CPU)")
    print("=" * 50)
    print(f"  windows : {n_windows}")
    print(f"  p50     : {np.percentile(a, 50):.2f} ms")
    print(f"  p95     : {np.percentile(a, 95):.2f} ms")
    print(f"  p99     : {np.percentile(a, 99):.2f} ms")
    print(f"  mean    : {a.mean():.2f} ms")
    print("=" * 50)
    print("  Target SLA: p95 < 200 ms (see docs/02)")
    return 0
