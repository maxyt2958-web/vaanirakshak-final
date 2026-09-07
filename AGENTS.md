<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

# VaniRakshak Agent Rules & Architecture Directives

## 1. Core Mandate
VaniRakshak is a target-conditioned recorded-audio forensic system that streams 16-bit mono PCM audio at 16 kHz over WebSockets to a FastAPI forensic backend. The backend scores risk, detects synthetic audio/cloning, and enforces autonomous transaction interlocks.

## 2. Separation of Concerns
- **Backend owns the verdict & interlocks**: `vanirakshak-backend` computes the Bayesian fused risk score and determines whether an interlock is `ALLOW`, `CHALLENGE`, or `BLOCK`. The Next.js frontend only displays verdicts and locks controls accordingly.
- **WebSocket Protocol contract**: Any changes to message fields or formats MUST match `docs/PROTOCOL.md`. Both `frontend/app/page.tsx` and `vanirakshak-backend/vanirakshak/server/app.py` must stay in sync with this protocol.
- **Server-Side Alerting**: Outbound security webhooks (`vanirakshak/alerts/`) must execute server-side. Never dispatch alert webhooks or store webhook secrets in client code.

## 3. Git & Hygiene
- Keep `.opencode/`, `.workbuddy-ai/`, `.venv/`, and audio datasets out of git.
- Adhere to `CONTRIBUTING.md` for branch naming, PR syncs, and verification steps.
