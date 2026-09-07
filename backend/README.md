# VaniRakshak Enterprise Deep Learning Backend

Enterprise FastAPI backend leveraging deep neural acoustic architectures:
- Intermediate 768-dimensional representations from `microsoft/wavlm-base-plus` across 2.0s sliding windows.
- Speaker verification using 192-dimensional embeddings from `speechbrain/spkrec-ecapa-voxceleb`.
- High-frequency Wiener entropy (spectral flatness) and vocoder phase boundary localization.
- Energy-based Voice Activity Detection (VAD).

## Architecture

```
backend/
├── Dockerfile           # Production container definition
├── requirements.txt     # Deep learning dependencies (torch, torchaudio, transformers, speechbrain)
├── app/
│   ├── main.py          # FastAPI application entrypoint & WebSocket /ws/stream handler
│   ├── config.py        # Hardware auto-detection, thresholds, model identifiers
│   ├── api/             # REST API routers (/api/v1/health, /api/v1/analyze/audio, /api/v1/enroll)
│   ├── core/            # In-memory audio tensor standardization and forensic signal DSP
│   ├── engine/          # Sliding window buffer and multi-factor risk fusion engine
│   └── models/          # WavLM feature extractor & ECAPA speaker verifier
└── tests/               # Forensic DSP and signal processing test suite
```

## Running the Backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Lightweight Alternative

For resource-constrained laptop environments without CUDA GPUs or deep learning wheels, see `vanirakshak-backend/`, which provides a pure NumPy / SciPy implementation capable of sub-25ms latency on a single CPU core.
