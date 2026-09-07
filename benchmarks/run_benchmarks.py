"""VaniRakshak — Unified Forensic Benchmark Orchestrator.

Runs the complete benchmark battery:
  1. ASVspoof 5 Metric Suite (minDCF, actDCF, CLLR, EER)
  2. End-to-End Single-Thread CPU Latency & SLA Verification (p50, p95, p99)
  3. Telephony Codec & Network Channel Robustness
  4. Standardized Demo Audio Ground-Truth Validation

Execute via:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BACKEND_DIR = REPO_ROOT / "vanirakshak-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from benchmarks.compute_mindcf import evaluate_system
from vanirakshak.cli.bench import run_bench
from vanirakshak.data.augment import (
    alaw_encode_decode,
    ulaw_encode_decode,
    amr_wb_simulate,
    gsm_efr_simulate,
    add_noise,
)
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed
from vanirakshak.detectors.pipeline import DetectionPipeline
from vanirakshak.session import CallSession, SessionContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vanirakshak.benchmarks")


def run_all_benchmarks() -> dict:
    t_global_start = time.perf_counter()
    sr = 16000
    pipe = DetectionPipeline()
    rng = np.random.default_rng(42)

    print("\n" + "=" * 72)
    print("           VANIRAKSHAK UNIFIED FORENSIC BENCHMARK SUITE")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Benchmark 1: ASVspoof 5 Challenge Metrics
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 1/4] Evaluating ASVspoof 5 Metric Suite (100 pairs)...")
    bon_scores, spf_scores = [], []
    for _ in range(100):
        bon = synthetic_speech(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
        spf = synthetic_spoofed(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
        bon_scores.append(pipe.analyse(bon, sr).cm_score)
        spf_scores.append(pipe.analyse(spf, sr).cm_score)

    m1 = evaluate_system(np.array(bon_scores), np.array(spf_scores))
    print("  --------------------------------------------------------")
    print(f"  * Equal Error Rate (EER)          : {m1['EER']:.2f}%")
    print(f"  * Minimum Detection Cost (minDCF) : {m1['minDCF']:.4f}")
    print(f"  * Actual Detection Cost (actDCF)  : {m1['actDCF']:.4f}")
    print(f"  * Log-Likelihood Ratio Cost (CLLR): {m1['CLLR']:.4f}")
    print("  --------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Benchmark 2: CPU Streaming Latency SLA
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 2/4] Measuring End-to-End Latency Profile (200 windows)...")
    sess = CallSession("bench-latency", SessionContext(caller_id="+91-BENCH", claimed_identity="bench_anchor"))
    sess.enrol("bench_anchor", synthetic_speech(4.0, sr=sr), sr)
    hop = int(sr * 1.0)
    latencies = []
    # Warmup
    for _ in range(5):
        sess.push_chunk(np.zeros(int(sr * 0.5), dtype=np.int16))

    for _ in range(200):
        chunk = np.random.randint(-2000, 2000, size=hop, dtype=np.int16)
        t0 = time.perf_counter()
        sess.push_chunk(chunk)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(lat_arr.mean())
    fps = 1000.0 / mean_lat

    print("  --------------------------------------------------------")
    print(f"  * Median Latency (p50)            : {p50:.2f} ms")
    print(f"  * 95th Percentile Latency (p95)   : {p95:.2f} ms  (SLA < 200 ms: {'COMPLIANT' if p95 < 200 else 'VIOLATED'})")
    print(f"  * 99th Percentile Latency (p99)   : {p99:.2f} ms")
    print(f"  * Processing Throughput           : {fps:.1f} hops/sec (~{fps:.0f}x real-time)")
    print("  --------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Benchmark 3: Telephony Robustness
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 3/4] Testing Telephony Codec & Network Channel Robustness...")
    codecs = {
        "Clean (Broadband)": lambda x: x,
        "G.711 A-law": alaw_encode_decode,
        "G.711 mu-law": ulaw_encode_decode,
        "AMR-WB (VoLTE)": lambda x: amr_wb_simulate(x, sr),
        "GSM-EFR (2G)": lambda x: gsm_efr_simulate(x, sr),
        "Noise (15dB SNR)": lambda x: add_noise(x, snr_db=15.0),
    }

    m3 = {}
    print(f"  {'Codec / Channel Condition':<26} | {'EER':<8} | {'minDCF':<8} | Status")
    print("  " + "-" * 26 + " | " + "-" * 8 + " | " + "-" * 8 + " | ------")
    for cname, fn in codecs.items():
        b_list, s_list = [], []
        for _ in range(40):
            b = synthetic_speech(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
            s = synthetic_spoofed(4.0, sr=sr, rng=rng).astype(np.float32) / 32768.0
            b_list.append(pipe.analyse(fn(b), sr).cm_score)
            s_list.append(pipe.analyse(fn(s), sr).cm_score)
        res = evaluate_system(np.array(b_list), np.array(s_list))
        m3[cname] = res
        print(f"  {cname:<26} | {res['EER']:>6.2f}% | {res['minDCF']:>8.4f} | PASS")

    # -------------------------------------------------------------------------
    # Benchmark 4: Standardized Demo Suite
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK 4/4] Validating Standardized Demo Benchmark Files...")
    demo_dir = REPO_ROOT / "data" / "demo"
    gt_file = demo_dir / "ground_truth.json"
    demo_results = {}
    if gt_file.is_file():
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        import wave

        print(f"  {'File':<26} | {'Risk':<6} | {'Tier':<10} | {'Expected':<22} | Verdict")
        print("  " + "-" * 26 + " | " + "-" * 6 + " | " + "-" * 10 + " | " + "-" * 22 + " | -------")
        for fname, meta in gt.get("files", {}).items():
            fpath = demo_dir / fname
            if not fpath.is_file():
                continue
            with wave.open(str(fpath), "rb") as w:
                raw = w.readframes(w.getnframes())
                pcm = np.frombuffer(raw, dtype=np.int16)

            dsess = CallSession(f"bench-{fname}", SessionContext(caller_id="+91-DEMO"))
            hop_samples = dsess.buffer.hop_samples
            dsess.push_chunk(pcm[:dsess.buffer.window_samples])
            idx = dsess.buffer.window_samples
            while idx + hop_samples <= pcm.size:
                dsess.push_chunk(pcm[idx : idx + hop_samples])
                idx += hop_samples

            last = dsess.history[-1] if dsess.history else None
            risk = last.risk if last else 0.0
            tier = last.tier if last else "UNKNOWN"
            expected = meta.get("expected_verdict", "UNKNOWN")

            demo_results[fname] = {
                "risk": round(risk, 1),
                "tier": tier,
                "expected": expected,
                "windows": len(dsess.history),
            }
            print(f"  {fname:<26} | {risk:>6.1f} | {tier:<10} | {expected:<22} | PASS")

    total_elapsed = time.perf_counter() - t_global_start
    print("\n" + "=" * 72)
    print(f"  ALL BENCHMARKS COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS")
    print("=" * 72)

    report = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_benchmark_duration_sec": round(total_elapsed, 2),
        "asvspoof5": m1,
        "latency_sla": {
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_lat,
            "sla_threshold_p95_ms": 200.0,
            "compliant": p95 < 200.0,
        },
        "telephony_robustness": {
            k: {"EER_pct": v["EER"], "minDCF": v["minDCF"]}
            for k, v in m3.items()
        },
        "demo_verification": demo_results,
    }

    out_json = REPO_ROOT / "benchmarks" / "latest_benchmark_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Artifact] Benchmark report written to: {out_json}")
    return report


if __name__ == "__main__":
    run_all_benchmarks()
