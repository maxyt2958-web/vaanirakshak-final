# Privacy Compliance & API Reference

## 1. Regulatory Framework: India DPDP Act 2023 Compliance

The **Digital Personal Data Protection (DPDP) Act, 2023** establishes strict requirements for the processing, retention, and protection of digital personal data within India. Voice recordings constitute **biometric personal data** under Indian privacy law.

### DPDP Act §8(7) Compliance: Zero Raw-Audio Retention

Section 8(7) of the DPDP Act mandates:
> *"A Data Fiduciary shall erase personal data, upon the Data Principal withdrawing her consent or as soon as it is reasonable to assume that the specified purpose for which personal data was collected is no longer being served, whichever is earlier."*

VaniRakshak achieves compliance through **privacy-by-architecture**:

```
                                 ┌─────────────────────────────────┐
                                 │   Continuous Audio Ingestion    │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ VOLATILE RAM SLIDING BUFFER (SlidingAudioBuffer)                                                │
│ • Fixed ring size: 80,640 samples (approx. 161 KB in memory)                                    │
│ • Window size: 4.04 s | Hop size: 1.0 s                                                         │
│ • In-place circular overwrite: Audio frames auto-expire in < 5.04 seconds                       │
│ • ZERO disk writes: No temporary .wav files, no swap-disk dumps, no persistent object storage   │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DEFENSIVE MEMORY WIPING                                                                         │
│ • Call Hangup / Session Completion: buffer.wipe() zero-fills buffer memory (_buf.fill(0))       │
│ • Interlock Triggered / Anomaly Detected: Immediate volatile buffer purge                       │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FEATURE-ONLY CRYPTOGRAPHIC AUDIT LOG (AuditLog)                                                 │
│ • Non-invertible scalar metrics only (cm_score, asv_consistency, snr_db)                        │
│ • NO raw audio waveforms, NO high-dimensional invertible embeddings                             │
│ • SHA-256 hash-chained receipts for tamper-evident compliance audit trail                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Memory Buffering Architecture (`SlidingAudioBuffer`)

The in-memory audio ring buffer is implemented in [`vanirakshak-backend/vanirakshak/privacy/buffer.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/privacy/buffer.py):

### Key Operational Characteristics
1. **Bounded Ring Memory Allocation:**
   $$\text{Capacity} = \text{window\_samples} + \text{hop\_samples} = \operatorname{round}(4.04 \times 16000) + \operatorname{round}(1.0 \times 16000) = 80,640\text{ samples}$$
   For 16-bit signed integers, this consumes exactly **$161,280\text{ bytes}$ ($\approx 161.3\text{ KB}$)** of volatile RAM per active call session.
2. **Circular FIFO Overwrite:**
   Incoming PCM frames advance `_write_pos`. When `_write_pos` reaches the buffer boundary, it wraps around to index `0`. Older audio frames are overwritten in-place. At no point can audio older than $5.04\text{ seconds}$ exist in memory.
3. **Defensive Wipe (`wipe()`):**
   When a call terminates, or upon transaction interlock execution, `buffer.wipe()` executes:
   ```python
   self._buf.fill(0)
   self._write_pos = 0
   self._filled = 0
   self._samples_since_hop = 0
   ```
   This ensures zero residual audio data remains in unmanaged RAM.

---

## 3. SHA-256 Append-Only Hash-Chained Audit Trail

To satisfy banking compliance auditing (RBI Master Directions on Fraud Risk Management and CERT-In logging directives) while adhering to DPDP Act privacy rules, VaniRakshak implements a **feature-only cryptographic audit log** in [`vanirakshak-backend/vanirakshak/privacy/audit.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/privacy/audit.py).

### Non-Invertibility Guarantee
The audit log stores strictly non-invertible scalar metrics:
- Session metadata: `session_id`, `caller_id`, `claimed_identity`, UTC timestamp
- Forensic outputs: `risk_score`, `tier` (`ALLOW`, `CHALLENGE`, `BLOCK`)
- Scalar features: `cm_score`, `cm_genuine_prob`, `asv_consistency`, `channel_anomaly`, `prosody_unnatural`, `snr_db`
- **Zero raw audio and zero raw neural embeddings are logged.** It is mathematically impossible to reconstruct the speaker's voice from these scalar telemetry records.

### Cryptographic Chaining
Every event is cryptographically bound to its predecessor via SHA-256:

$$\text{receipt\_hash}_k = \operatorname{SHA-256}\Big(\operatorname{canonical\_json}(\text{payload}_k) \Big)$$

where:
$$\text{payload}_k = \Big\{ \text{session\_id}, \; \text{caller\_id}, \; \text{claimed\_identity}, \; \text{ts}, \; \text{risk}, \; \text{tier}, \; \text{features}, \; \text{prev} = \text{receipt\_hash}_{k-1} \Big\}$$

The canonical JSON formatting uses strict key sorting and minimal whitespace (`separators=(",", ":")`), guaranteeing deterministic hash generation across heterogeneous systems.

### Verification (`verify_chain()`)
The audit log includes an automated integrity verification method:
```python
is_valid = audit_log.verify_chain()
```
If an insider or attacker alters, re-orders, or deletes any audit log entry, the hash chain breaks from that point forward, rendering tampering immediately evident.

---

## 4. API Reference

VaniRakshak exposes both real-time streaming WebSocket endpoints and standard HTTP REST endpoints.

### 4.1. WebSocket API: `/ws/stream` (alias `/v1/stream`)

The primary streaming interface for live call monitoring and real-time frontend dashboard interaction.

#### Connection Lifecycle

```
Client (Browser / CTI Gateway)                         Server (FastAPI)
             │                                                │
             │──── 1. HTTP WebSocket Upgrade ───────────────▶│ (Accepts on ws://host:8000/ws/stream)
             │◀─── 2. 101 Switching Protocols ────────────────│
             │                                                │
             │──── 3. Frame 1: JSON Text Handshake ──────────▶│ (Parses metadata, initializes session)
             │                                                │
             │──── 4. Frame 2..N: Binary PCM Chunks ─────────▶│ (Buffers 16 kHz 16-bit PCM in RAM)
             │                                                │
             │◀─── 5. JSON Text Verdict (every 1.0s hop) ─────│ (Emits risk score, tier, receipt hash)
             │                                                │
```

> [!IMPORTANT]
> **Strict Handshake Protocol Requirement:**
> The first frame sent across the WebSocket **must be a JSON text handshake**. If binary audio is sent before the handshake, the backend closes the connection immediately with WebSocket error code `1003`.

#### 1. Handshake Schema (Client $\to$ Server)
```json
{
  "session_id": "call-ind-9821-4412",
  "sample_rate": 16000,
  "caller_id": "+91-9876543210",
  "claimed_identity": "customer_acct_44901",
  "transaction_value_inr": 150000.0,
  "origin_country": "IN",
  "prior_fraud_score": 0.05
}
```

#### 2. Binary Audio Frames (Client $\to$ Server)
- **Format:** Raw 16-bit signed integer mono PCM (`Int16Array`).
- **Sample Rate:** 16,000 Hz.
- **Chunk Size:** Arbitrary chunk sizes supported (typically 50 ms to 1000 ms blocks).

#### 3. Verdict Response Schema (Server $\to$ Client)
Emitted every 1.0 second once the sliding buffer reaches 4.04 seconds:
```json
{
  "timestamp_ms": 1773070080123,
  "risk_score": 84.5,
  "p_spoof": 0.9412,
  "tier": "BLOCK",
  "metrics": {
    "cm_score": -3.42,
    "cm_genuine_prob": 0.0317,
    "asv_consistency": 0.22,
    "channel_anomaly": 0.85,
    "prosody_unnatural": 0.91,
    "snr_db": 14.2,
    "codec_guess": "wideband_opus"
  },
  "trigger_challenge": false,
  "interlock_active": true,
  "receipt_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "challenge": null
}
```

#### WebSocket Status Codes
| Code | Reason |
|---|---|
| `1000` | Normal socket closure. |
| `1003` | Handshake format violation (binary audio sent before JSON text handshake). |

---

### 4.2. REST API Reference

#### `GET /api/v1/health` (and `GET /v1/health`)
Checks backend operational status, liveness, and hardware execution context.

**Response (`200 OK`):**
```json
{
  "ok": true,
  "version": "0.1.0-sprint1",
  "active_sessions": 2
}
```

---

#### `POST /v1/enroll`
Registers a 10–20 second genuine voice sample for target-conditioned speaker verification.

- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `user_id` (string, required): Unique identifier for the speaker/account.
  - `audio` (file, required): WAV/FLAC audio file (minimum 0.5 seconds).

**Response (`200 OK`):**
```json
{
  "status": "enrolled",
  "user_id": "customer_acct_44901",
  "embedding_dimension": 192,
  "audio_retained": false,
  "created_at": "2026-09-07T15:18:00Z"
}
```

---

#### `POST /v1/analyze`
One-shot offline forensic analysis of a recorded WAV or FLAC file.

- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `audio` (file, required): Recorded audio file.
  - `caller_id` (string, optional): Caller phone number.
  - `claimed_identity` (string, optional): Enrolled speaker identity.

**Response (`200 OK`):**
```json
{
  "status": "success",
  "file_duration_sec": 4.5,
  "calibrated_risk": 78.4,
  "p_spoof": 0.8921,
  "tier": "BLOCK",
  "receipt_hash": "8f4a1c3...",
  "breakdown": {
    "codec_guess": "g711_or_pstn",
    "cm_score": -2.85,
    "asv_consistency": 0.31,
    "channel_anomaly": 0.65,
    "prosody_unnatural": 0.82
  }
}
```

---

#### `POST /v1/sessions`
Initializes a stateful session context prior to an incoming call or transaction.

**Request Body (`application/json`):**
```json
{
  "caller_id": "+91-9876543210",
  "claimed_identity": "customer_acct_44901",
  "transaction_value_inr": 250000.0,
  "origin_country": "IN",
  "prior_fraud_score": 0.0
}
```

**Response (`200 OK`):**
```json
{
  "session_id": "call-7a8f1b2c3d"
}
```

---

#### `POST /v1/sessions/{session_id}/challenge`
Submits a user's verbal challenge response for verification.

**Request Body (`application/json`):**
```json
{
  "response_text": "Mango 8 nadi 4 blue",
  "response_latency_ms": 1420
}
```

**Response (`200 OK`):**
```json
{
  "ok": true,
  "reason": "pass",
  "confidence": 0.95,
  "blocked": false
}
```

---

#### `GET /v1/audit`
Retrieves recent audit events and verifies cryptographic hash-chain integrity.

- **Query Parameters:** `n` (integer, default: 50): Number of tail events to retrieve.

**Response (`200 OK`):**
```json
{
  "chain_valid": true,
  "events": [
    {
      "session_id": "call-7a8f1b2c3d",
      "caller_id": "+91-9876543210",
      "claimed_identity": "customer_acct_44901",
      "timestamp": 1773070080.123,
      "timestamp_iso": "2026-09-07T15:18:00Z",
      "risk_score": 84.5,
      "tier": "BLOCK",
      "features": {
        "cm_genuine_prob": 0.0317,
        "asv_consistency": 0.22,
        "channel_anomaly": 0.85,
        "prosody_unnatural": 0.91,
        "snr_db": 14.2
      },
      "receipt_hash": "e3b0c442...",
      "prev_hash": "a1b2c3d4..."
    }
  ]
}
```

---

## 5. Configuration & Environment Variables

All parameters can be configured via environment variables or a `.env` file:

| Environment Variable | Default Value | Type | Scope | Description |
|---|---|---|---|---|
| `VR_PORT` | `8000` | Integer | Server | Backend HTTP & WebSocket listen port. |
| `VR_HOST` | `127.0.0.1` | String | Server | Backend network interface bind address. |
| `VR_CORS_ORIGINS` | `http://localhost:3001,http://127.0.0.1:3001` | String (csv) | Security | Permitted CORS origins for Next.js browser clients. |
| `VR_ALERT_WEBHOOK_URL` | *(unset)* | String (URL) | Alerting | Destination URL for outbound POST alerts on `CHALLENGE` or `BLOCK`. Unset disables outbound webhook dispatch. |
| `NEXT_PUBLIC_BACKEND_WS` | `ws://localhost:8000/ws/stream` | String (WS URI)| Frontend | Target WebSocket URI used by the Next.js UI. |
| `VR_SAMPLE_RATE` | `16000` | Integer | DSP | Processing sample rate in Hz. |
| `VR_WINDOW_SECONDS` | `4.04` | Float | DSP | Receptive field duration for countermeasure analysis. |
| `VR_HOP_SECONDS` | `1.0` | Float | DSP | Sliding window step interval in seconds. |
| `VR_ALLOW_THRESHOLD` | `35.0` | Float | Fusion | Upper risk score bound for the `ALLOW` tier. |
| `VR_CHALLENGE_THRESHOLD`| `70.0` | Float | Fusion | Upper risk score bound for the `CHALLENGE` tier ($\ge 70.0$ triggers `BLOCK`). |
| `VR_EMA_ALPHA` | `0.65` | Float | Fusion | Exponential moving average smoothing weight ($0 < \alpha \le 1$). |
| `VR_FAST_RISE_THRESHOLD`| `0.85` | Float | Fusion | Posterior spoof probability threshold that bypasses EMA smoothing. |
| `VR_FUSION_BIAS` | `-1.6` | Float | Fusion | Bayesian logistic regression intercept ($\beta_0$). |
| `VR_FUSION_W_CM` | `2.4` | Float | Fusion | Weight for countermeasure spoof evidence ($w_{\text{cm}}$). |
| `VR_FUSION_W_ASV` | `1.6` | Float | Fusion | Weight for speaker verification mismatch ($w_{\text{asv}}$). |
| `VR_FUSION_W_PROSODY` | `0.7` | Float | Fusion | Weight for prosody unnaturalness ($w_{\text{prosody}}$). |
| `VR_FUSION_W_CHANNEL` | `0.9` | Float | Fusion | Weight for channel codec anomaly ($w_{\text{channel}}$). |
| `VR_FUSION_W_CONTEXT` | `0.5` | Float | Fusion | Weight for contextual metadata risk ($w_{\text{context}}$). |
