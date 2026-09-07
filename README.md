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
└── frontend/             # Next.js 16 (App Router) dashboard
    └── app/
        ├── page.tsx                        # main console: live stream, risk meter, event log
        ├── layout.tsx
        ├── globals.css
        └── components/
            ├── ScrollyVideoCanvas.tsx      # scroll-driven animated pipeline explainer
            └── ...                         # placeholder stubs (not yet implemented)
```

## Getting started

```bash
npm install --prefix frontend
npm run dev
```

The dev server runs on **http://localhost:3001**.

The frontend expects the forensic backend at:

```
ws://localhost:8000/ws/stream?session_id=<id>
```

The backend is **not** included in this repository. Without it the UI falls back to **Demo mode**, which cycles through three canned verdicts (risk 18 / 43 / 87) every 4 seconds — enough to exercise the full UI and the interlock.

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

- `Makefile` and `docker-compose.yml` are currently empty placeholders.
- `frontend/app/components/` contains four empty stub files (`AudioUploadZone`, `DemoQuickSelector`, `ForensicVerdictCard`, `WaveformEvidenceTimeline`) reserved for planned features; they are not imported anywhere.

## License

Apache-2.0
