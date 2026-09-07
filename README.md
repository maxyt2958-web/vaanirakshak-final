# VaniRakshak

**Target-Conditioned Recorded-Audio Forensic System** — real-time detection of voice-cloning and deepfake-audio fraud on live calls, with an autonomous transaction interlock.

VaniRakshak listens to a live audio stream, scores it for synthetic/manipulated speech, and drives a risk decision (`ALLOW` → `CHALLENGE` → `BLOCK`) that can lock a transaction mid-call and fire alerts.

## How it works

The dashboard streams microphone audio to a forensic backend over WebSocket and renders the verdict in real time.

```
Microphone  →  16-bit PCM @ 16 kHz  →  WebSocket /ws/stream  →  Forensic backend
                                                                      │
                                                                      ▼
                                                     risk_score, spoof_score,
                                                     speaker_similarity, snr_db
                                                                      │
                                                                      ▼
                                                    ALLOW / CHALLENGE / BLOCK
```

### ML-side architecture

VaniRakshak analyzes the raw waveform directly. It does **not** use speech-to-text,
NER, BERT, emotion classification, or LLM-RAG. The current AASIST and ECAPA-TDNN
components are deterministic synthetic stubs with documented replacement points for
real model weights.

```mermaid
flowchart LR
    A["Raw 16 kHz PCM<br/>sliding audio window"] --> B["CM anti-spoofing<br/>AASIST stub"]
    A --> C["ASV speaker verification<br/>ECAPA-TDNN stub"]
    A --> D["Channel and codec detector"]
    A --> E["Prosody micro-tremor detector"]
    B --> B1["CM score and genuine probability"]
    C --> C1["ASV consistency to claimed identity"]
    D --> D1["Channel anomaly, codec and SNR"]
    E --> E1["Prosody unnaturalness"]
    B1 --> F["Calibrated Bayesian fusion<br/>sigmoid plus EMA"]
    C1 --> F
    D1 --> F
    E1 --> F
    G["Context risk<br/>country, value and prior fraud"] --> F
    F --> H["Risk score from 0 to 100<br/>fast-rise override"]
    H --> I{"ALLOW, CHALLENGE or BLOCK"}
```

| Risk score | Label | Action |
|---|---|---|
| `< 35` | LOW RISK | `ALLOW` — transaction proceeds |
| `35 – 69` | SUSPICIOUS | `CHALLENGE` — caller must repeat a dynamic phrase |
| `>= 70` | CRITICAL | `BLOCK` — transaction interlock engages, alerts fire |

`BLOCK` or `risk >= 70` hard-locks the transaction control (`transactionLocked`), disabling the proceed button until the stream clears.

### The four detection stages

The landing experience is a scroll-driven canvas animation (220 pre-rendered frames) walking through the pipeline:

1. **Ingestion** — edge-first capture, 16-bit PCM, sub-12 ms latency
2. **Multi-tier analysis** — STFT and bi-spectral inspection for vocoder phase discontinuities and neural speech footprints (AASIST / LFCC)
3. **Biometric defense** — acoustic spoof detection and speaker-verification similarity scoring
4. **Autonomous action** — real-time interlock and quarantine of the transaction

## Project structure

```
.
├── package.json          # workspace wrapper — delegates to frontend/
├── docs/PROTOCOL.md      # frontend <-> backend WebSocket contract (read this first)
├── frontend/             # Next.js 16 (App Router) dashboard
│   ├── public/worklets/pcm-capture.js   # AudioWorklet PCM capture
│   └── app/
│       ├── page.tsx                     # console: live stream, risk meter, event log
│       ├── layout.tsx
│       ├── globals.css
│       └── components/
│           ├── ScrollyVideoCanvas.tsx   # scroll-driven animated pipeline explainer
│           └── ...                      # placeholder stubs (not yet implemented)
└── vanirakshak-backend/  # Python FastAPI forensic service
    └── vanirakshak/
        ├── server/app.py       # REST + WebSocket endpoints
        ├── session.py          # per-call state, interlock, challenges
        ├── risk/fusion.py      # calibrated Bayesian risk fusion
        ├── detectors/          # asv, channel, cm (spoof), prosody, pipeline
        ├── challenge/engine.py # dynamic phrase generation + grading
        └── privacy/            # DPDP-style audit log
```

The wire format between the two halves is documented in **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

## Getting started

Run the backend and the dashboard in two terminals.

**Backend** (port 8000):

```bash
cd vanirakshak-backend
pip install -r requirements.txt
python run.py
```

**Dashboard** (port 3001):

```bash
npm install --prefix frontend
npm run dev
```

Open **http://localhost:3001** and click the status badge to start streaming.

With no backend reachable, the UI falls back to **Demo mode**, which cycles three
canned verdicts (risk 18 / 43 / 87) every 4 seconds. Demo results are labelled
`DEMO · SIMULATED` — they are not real detections.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_BACKEND_WS` | `ws://localhost:8000/ws/stream` | Dashboard → backend socket |
| `VR_PORT` | `8000` | Backend port |
| `VR_HOST` | `127.0.0.1` | Backend bind host |
| `VR_CORS_ORIGINS` | `http://localhost:3001,http://127.0.0.1:3001` | Allowed browser origins |

### Other scripts

```bash
npm run build   # production build
npm run start   # serve production build
npm run lint    # eslint
```

## Tech stack

- **Next.js 16.3.4** (App Router) · **React 19.2.8**
- **Tailwind CSS v4**
- **lucide-react** iconography
- **TypeScript 5**
- Web Audio API (`ScriptProcessorNode`) for PCM capture

## Notes

**The detectors are synthetic stubs.** Per the backend README, `vanirakshak-backend`
ships *deterministic synthetic "honest stub" models* so the full pipeline — fusion,
challenge engine, privacy layer, WebSocket streaming — runs on any laptop with no GPU
and no pretrained weights. Real AASIST / ECAPA-TDNN checkpoints can be dropped in later
by replacing two files in `vanirakshak/detectors/`. Scores are therefore **not** yet
real detections.

Other known gaps:

- `Makefile` and `docker-compose.yml` are empty placeholders.
- `frontend/app/components/` contains four empty stub files (`AudioUploadZone`,
  `DemoQuickSelector`, `ForensicVerdictCard`, `WaveformEvidenceTimeline`) reserved for
  planned features; they are not imported anywhere.
- The interlock is `interlock_active || action === "BLOCK" || risk >= 70`. The server
  owns the decision; the client-side thresholds exist only for demo mode.
- `alertPayload` renders the alert text on screen but no webhook / SMS / email is
  actually dispatched.
- There is no WebSocket reconnect logic — a dropped socket logs an event and stops.

## License

Apache-2.0
