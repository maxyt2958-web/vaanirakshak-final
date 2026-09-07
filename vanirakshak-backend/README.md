# VaniRakshak — Real-Time Voice Cloning Impersonation Detection Backend

End-to-end, runnable prototype of the **R4 (Backend, Streaming, Privacy) + R3 (Speaker & Challenge) + R2 (Inference shell) + R1 (Augmentation & eval integration)** slices of the [VaniRakshak SIH 2026 project](../README.md).

It exposes the architecture promised in `docs/02-technical-strategy-and-architecture.md` and the API surface in `docs/07-privacy-compliance-and-apis.md`, but with **deterministic synthetic "honest stub" models** so the entire pipeline — including the calibrated risk fusion, the challenge engine, the privacy layer, and the WebSocket streaming — runs **on any laptop without GPU or pretrained weights**. Real AASIST / ECAPA-TDNN checkpoints can be plugged in later by replacing two files in `vanirakshak/detectors/`.

## Quick start

```bash
# 1. Install (pure-Python + FastAPI; ~30s on a fresh venv)
pip install -r requirements.txt

# 2. Run the offline demo (no server, no GPU, no audio needed)
python run.py demo

# 3. Run the full FastAPI server (WebSocket + REST + SDK)
python run.py serve
# Then in another shell:
python examples/sdk_demo.py
```

The demo prints, in 30 seconds, what the 90-second live demo will show on stage: a stream of risk verdicts, a triggered challenge, and a final minDCF number on synthetic telephony data.

## What is here

| Component | File | Status |
|---|---|---|
| Streaming PCM buffer (RAM-only, DPDP-compliant) | `vanirakshak/privacy/buffer.py` | ✅ working |
| Sliding-window inference loop | `vanirakshak/detectors/pipeline.py` | ✅ working |
| Acoustic CM (AASIST stub, pluggable) | `vanirakshak/detectors/cm.py` | 🟡 synthetic, real AASIST drop-in |
| Speaker embedding (ECAPA-TDNN stub, pluggable) | `vanirakshak/detectors/asv.py` | 🟡 synthetic, real ECAPA drop-in |
| Telephony codec & prosody features | `vanirakshak/detectors/channel.py`, `prosody.py` | ✅ working |
| Calibrated Bayesian risk fusion (0–100) | `vanirakshak/risk/fusion.py` | ✅ working |
| Dynamic challenge engine + response analyzer | `vanirakshak/challenge/engine.py` | ✅ working |
| DPDP Act 2023 privacy + SHA-256 audit receipts | `vanirakshak/privacy/audit.py` | ✅ working |
| Codec augmentation (G.711 a/μ-law, AMR-WB sim, noise) | `vanirakshak/data/augment.py` | ✅ working |
| ASVspoof 5 metric harness (minDCF/actDCF/CLLR/EER) | re-uses `benchmarks/compute_mindcf.py` | ✅ working |
| FastAPI server (WS /v1/stream, REST /v1/analyze, /v1/enroll) | `vanirakshak/server/app.py` | ✅ working |
| 5-line Python SDK | `vanirakshak/sdk/client.py` | ✅ working |
| OpenAPI docs (auto) | `/docs` on the served URL | ✅ working |

## What is NOT here (and why)

- **Real AASIST / ECAPA-TDNN weights.** The 72-hour internal hackathon deadline (Sept 7) does not allow Kaggle session shuffling for training. The stubs produce **calibrated, deterministic** scores with a known synthetic-prior so the minDCF/actDCF numbers are honest about being synthetic. Swap-in points are clearly marked.
- **Real Twilio/Exotel carrier integration.** Twilio Media Streams over WebSocket are easy to bolt on; the WS endpoint already speaks the same binary PCM format (see `vanirakshak/server/app.py:WS /v1/stream`).
- **Production SMS gateway.** Per the build plan, multi-channel alerting is implemented as webhook + email stubs.

## Architecture

```
Incoming PCM (1.0 s chunks)
       │
       ▼
┌──────────────────────────┐
│ RAM-only sliding buffer  │   ◀── DPDP §8(7): zero retention
│ (4.04 s window, 75% ovl) │
└──────────┬───────────────┘
           ▼
┌────────────────────────────────────────┐
│ Multi-layer scoring                    │
│  • CM (AASIST stub) → s_CM ∈ [0,1]     │
│  • ASV (ECAPA stub) → s_ASV ∈ [0,1]    │
│  • Telephony codec detect → s_ch       │
│  • Prosody micro-tremor → s_pros       │
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ Calibrated risk fusion                 │
│  P(spoof|x) = σ(w·x + b)               │
│  risk = round(100·P, 1)                │
│  tier = ALLOW | CHALLENGE | BLOCK      │
└──────────┬─────────────────────────────┘
           ▼
┌────────────────────────────────────────┐
│ 35 ≤ risk < 70  → Challenge engine     │
│ risk ≥ 70       → BLOCK + interlock    │
│ Always          → SHA-256 audit receipt│
└────────────────────────────────────────┘
```

See `docs/02-technical-strategy-and-architecture.md` for the authoritative design.

## License

Apache 2.0 — see `../LICENSE`.
