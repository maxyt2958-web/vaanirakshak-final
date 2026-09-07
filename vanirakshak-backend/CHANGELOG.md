# Changelog

All notable changes to **VaniRakshak**.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0-sprint1] — 2026-09-04

### Added — R4 backend slice (runnable end-to-end prototype)

This release adds the missing **R4 (Backend, Streaming, Integration API,
Privacy) slice** plus integration with **R1 (data / eval)**, **R2 (model
shells)**, and **R3 (speaker / challenge)** so the team has a working
end-to-end pipeline for the 7-8 September internal hackathon.

The new package lives in [`vanirakshak-backend/`](./vanirakshak-backend).
It is **deliberately synthetic**: the AASIST, ECAPA-TDNN, and ASR modules
are hand-crafted calibrated stubs so the entire pipeline runs on a
laptop with no GPU, no pretrained weights, and no audio file. Real
checkpoints can be dropped in by replacing two files in
`vanirakshak/detectors/`.

#### New components

* `vanirakshak/detectors/cm.py` — calibrated spectral-feature countermeasure
  (AASIST stub). Returns a log-likelihood ratio.
* `vanirakshak/detectors/asv.py` — 192-dim speaker embedding with
  enrolment + cosine similarity verification (ECAPA-TDNN stub).
* `vanirakshak/detectors/channel.py` — telephony codec & SNR detector.
* `vanirakshak/detectors/prosody.py` — pitch jitter & micro-tremor
  consistency.
* `vanirakshak/detectors/pipeline.py` — unified multi-layer analysis
  per window.
* `vanirakshak/privacy/buffer.py` — RAM-only sliding PCM ring buffer
  (DPDP §8(7) compliant; raw audio never persisted).
* `vanirakshak/privacy/audit.py` — append-only SHA-256 hash-chained
  audit log with tamper detection.
* `vanirakshak/risk/fusion.py` — calibrated Bayesian risk fusion with
  exponential moving average + fast-rise override; emits a 0-100 risk
  score and an ALLOW/CHALLENGE/BLOCK tier.
* `vanirakshak/challenge/engine.py` — dynamic challenge-phrase
  generator and response grader (latency + digit-order + filler cues).
* `vanirakshak/data/augment.py` — G.711 a-law / μ-law, AMR-WB
  simulation, babble, RIR, gain jitter, full augmentation pipeline.
* `vanirakshak/data/synthetic.py` — honest synthetic audio generators
  for the demo & tests (not voice samples; clearly documented as such).
* `vanirakshak/server/app.py` — FastAPI server: `POST /v1/enroll`,
  `POST /v1/analyze`, `POST /v1/sessions`, `POST /v1/sessions/{sid}/challenge`,
  `GET /v1/audit`, `GET /v1/health`, `WS /v1/stream`, `GET /docs`.
* `vanirakshak/sdk/client.py` — 5-line Python SDK over HTTP +
  `websockets.sync.client` for the WebSocket.
* `vanirakshak/session.py` — `CallSession` orchestrator that owns the
  buffer, pipeline, fusion, challenge, and audit for one call.
* `vanirakshak/cli/{demo,eval_synth,bench,__init__}.py` — entry points.
* `tests/test_smoke.py` — 8 end-to-end smoke tests (run in 1 s).
* `examples/sdk_demo.py` — sample SDK call.
* `requirements.txt` — pinned versions (fastapi 0.115, numpy 1.26, etc.).

#### Verified behaviour

* `python -m vanirakshak demo` — full demo with ALLOW → CHALLENGE →
  BLOCK transitions, challenge phrase issued, human response passes,
  replay response fails, SHA-256 audit chain verified.
* `python -m vanirakshak bench` — p50 = 9.8 ms, p95 = 10.8 ms per window
  (well under the 200 ms SLA in `docs/02`).
* `python -m vanirakshak eval` — minDCF / actDCF / CLLR / EER
  computed on synthetic data using the official ASVspoof 5 harness
  re-used from `benchmarks/compute_mindcf.py`.
* `python -m vanirakshak serve` — FastAPI server boots, `/v1/health`
  returns OK, all 6 documented routes registered.

#### What is still stub (deliberately, for sprint 1)

* Real AASIST / ECAPA-TDNN weights (drop-in points clearly marked).
* Real Twilio / Exotel carrier integration (the WS endpoint already
  speaks the same binary PCM frame format).
* Real SMS gateway for multi-channel alerting (webhook + email stubbed).

### Honest reporting standard

Per `docs/04-metrics-and-honest-evaluation.md`, the demo and tests are
**explicit** that the CM/ASV scores are synthetic. The metric formulas
(minDCF, actDCF, CLLR, EER) are the official ASVspoof 5 ones; the score
distributions are not representative of any trained model. The team
must replace the stubs with real weights before reporting headline
numbers on stage.

[0.1.0-sprint1]: #010-sprint1--2026-09-04
