"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ShieldCheck,
  Activity,
  Mic,
  Play,
  Square,
  Lock,
  AlertOctagon,
} from "lucide-react";
import ScrollyVideoCanvas from "./components/ScrollyVideoCanvas";

type RiskAction = "ALLOW" | "CHALLENGE" | "BLOCK";

/**
 * Wire format emitted by the Python backend
 * (`vanirakshak-backend/vanirakshak/server/app.py` → `WS /v1/stream`).
 * Normalised into `AnalysisResult` by `normalizeResult()`.
 * See docs/PROTOCOL.md for the full contract.
 */
type BackendResult = {
  timestamp_ms?: number;
  risk_score?: number;
  p_spoof?: number;
  tier?: string;
  metrics?: Record<string, unknown>;
  trigger_challenge?: boolean;
  interlock_active?: boolean;
  receipt_hash?: string;
  challenge?: { phrase?: string } | null;
};

type AnalysisResult = {
  session_id?: string;
  window_index?: number;
  timestamp?: number;
  risk_score: number;
  action: RiskAction;
  spoof_score: number;
  speaker_similarity: number;
  snr_db: number;
  challenge_phrase?: string | null;
  /** Server-authoritative lock. The client must not override this. */
  interlock_active: boolean;
};

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Map the backend payload onto the shape the dashboard renders. */
function normalizeResult(raw: BackendResult): AnalysisResult {
  const metrics = raw.metrics ?? {};
  const tier = raw?.tier;

  return {
    risk_score: asNumber(raw.risk_score),
    action:
      tier === "ALLOW" || tier === "CHALLENGE" || tier === "BLOCK"
        ? tier
        : "ALLOW",
    spoof_score: asNumber(raw.p_spoof),
    speaker_similarity: asNumber(metrics.asv_consistency, 1),
    snr_db: asNumber(metrics.snr_db),
    challenge_phrase: raw.challenge?.phrase ?? null,
    interlock_active: raw.interlock_active === true,
  };
}

type EventLog = {
  id: number;
  time: string;
  message: string;
  severity: "info" | "warning" | "critical";
};

const BACKEND_WS =
  process.env.NEXT_PUBLIC_BACKEND_WS ?? "ws://localhost:8000/ws/stream";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;
const MAX_RECONNECT_ATTEMPTS = 5;

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [micLive, setMicLive] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [risk, setRisk] = useState(0);
  const [alertMessage, setAlertMessage] = useState("No active security alert");
  const [alertPayload, setAlertPayload] = useState("");
  const [action, setAction] = useState<RiskAction>("ALLOW");
  const [spoofScore, setSpoofScore] = useState(0);
  const [speakerSimilarity, setSpeakerSimilarity] = useState(1);
  const [snr, setSnr] = useState(0);
  const [challenge, setChallenge] = useState<string | null>(null);
  const [events, setEvents] = useState<EventLog[]>([]);
  const [showAllEvents, setShowAllEvents] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [interlockActive, setInterlockActive] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mutedGainRef = useRef<GainNode | null>(null);

  const demoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  /** Set while the user is intentionally disconnected, to suppress retries. */
  const manualCloseRef = useRef(false);

  // The backend owns the interlock decision. Local thresholds are only a
  // fallback for demo mode, where no server is involved.
  const transactionLocked =
    interlockActive || action === "BLOCK" || risk >= 70;

  const riskLabel = useMemo(() => {
    if (risk >= 70) return "CRITICAL";
    if (risk >= 35) return "SUSPICIOUS";
    return "LOW RISK";
  }, [risk]);

  function addEvent(
    message: string,
    severity: EventLog["severity"] = "info"
  ) {
    setEvents((current) => [
      {
        id: Date.now() + Math.random(),
        time: new Date().toLocaleTimeString(),
        message,
        severity,
      },
      ...current,
    ].slice(0, 12));
  }

  function applyResult(result: AnalysisResult) {
    setRisk(result.risk_score);
    setAction(result.action);
    setSpoofScore(result.spoof_score);
    setSpeakerSimilarity(result.speaker_similarity);
    setSnr(result.snr_db);
    setChallenge(result.challenge_phrase ?? null);
    setInterlockActive(result.interlock_active);

    if (result.action === "BLOCK") {
      setAlertMessage("CRITICAL ALERT • Transaction blocked");
      setAlertPayload(
        `WEBHOOK/SMS/EMAIL ALERT → risk=${result.risk_score}, action=BLOCK`
      );
    } else if (result.action === "CHALLENGE") {
      setAlertMessage("SECURITY ALERT • Caller verification required");
      setAlertPayload(
        `WEBHOOK/SMS/EMAIL ALERT → risk=${result.risk_score}, action=CHALLENGE`
      );
    } else {
      setAlertMessage("No active security alert");
      setAlertPayload("");
    }

    if (result.action === "BLOCK") {
      addEvent(
        `Critical risk detected: transaction interlock activated`,
        "critical"
      );
    } else if (result.action === "CHALLENGE") {
      addEvent("Suspicious call: dynamic challenge requested", "warning");
    } else {
      addEvent("Call analysis normal: transaction allowed", "info");
    }
  }

  function connectToBackend() {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    // An explicit connect cancels any pending retry and re-enables
    // auto-reconnect for this session.
    manualCloseRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const sessionId = `r5-demo-${Date.now()}`;
    const ws = new WebSocket(
      `${BACKEND_WS}?session_id=${encodeURIComponent(sessionId)}`
    );

    socketRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setDemoMode(false);
      reconnectAttemptsRef.current = 0;
      addEvent("Connected to VaniRakshak backend", "info");

      // The backend expects a JSON text handshake before any audio frames.
      // Without it the server closes the socket with 1003 and no audio is
      // ever analysed. See docs/PROTOCOL.md.
      ws.send(
        JSON.stringify({
          session_id: sessionId,
          sample_rate: 16000,
          caller_id: "",
          claimed_identity: "",
          transaction_value_inr: 0,
          origin_country: "IN",
          prior_fraud_score: 0,
        })
      );

      startMicrophone(ws);
    };

    ws.onmessage = (message) => {
      try {
        const raw: BackendResult = JSON.parse(message.data);
        applyResult(normalizeResult(raw));
      } catch {
        addEvent("Received an invalid backend message", "warning");
      }
    };

    ws.onerror = () => {
      addEvent(
        "Backend connection failed — demo mode can be used",
        "warning"
      );
    };

    ws.onclose = () => {
      setConnected(false);
      setMicLive(false);
      socketRef.current = null;

      if (manualCloseRef.current) return;

      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        addEvent(
          "Backend unreachable — retry limit reached, use demo mode",
          "warning"
        );
        return;
      }

      const attempt = ++reconnectAttemptsRef.current;
      // Exponential backoff with a random jitter so retried clients
      // don't all hammer the server in lockstep.
      const backoff = Math.min(
        RECONNECT_BASE_MS * 2 ** (attempt - 1),
        RECONNECT_MAX_MS
      );
      const delay = backoff + Math.random() * 300;

      addEvent(
        `Connection lost — retrying in ${Math.round(delay / 1000)}s (attempt ${attempt}/${MAX_RECONNECT_ATTEMPTS})`,
        "warning"
      );

      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        if (!manualCloseRef.current) connectToBackend();
      }, delay);
    };

    async function startMicrophone(ws: WebSocket) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
          },
        });

        mediaStreamRef.current = stream;

        const AudioContextClass =
          window.AudioContext ||
          (window as typeof window & {
            webkitAudioContext?: typeof AudioContext;
          }).webkitAudioContext;

        const audioContext = new AudioContextClass({
          sampleRate: 16000,
        });

        audioContextRef.current = audioContext;

        // AudioWorklet replaces the deprecated ScriptProcessorNode.
        await audioContext.audioWorklet.addModule("/worklets/pcm-capture.js");

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, "pcm-capture");

        // Kept in the graph so the worklet is pulled, but muted so the
        // microphone is never echoed back through the speakers.
        const mutedGain = audioContext.createGain();
        mutedGain.gain.value = 0;

        sourceRef.current = source;
        workletNodeRef.current = workletNode;
        mutedGainRef.current = mutedGain;

        workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
          if (ws.readyState !== WebSocket.OPEN) return;

          const input = event.data;

          let sum = 0;

          for (let i = 0; i < input.length; i++) {
            sum += input[i] * input[i];
          }

          const rms = Math.sqrt(sum / input.length);
          setAudioLevel(Math.min(100, Math.round(rms * 300)));

          const pcm = new Int16Array(input.length);

          for (let i = 0; i < input.length; i++) {
            const sample = Math.max(-1, Math.min(1, input[i]));
            pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
          }

          ws.send(pcm.buffer);
        };

        source.connect(workletNode);
        workletNode.connect(mutedGain);
        mutedGain.connect(audioContext.destination);

        addEvent("Microphone streaming started", "info");
        setMicLive(true);
      } catch {
        addEvent("Microphone access failed", "warning");
      }
    }
  }

  function disconnect() {
    manualCloseRef.current = true;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectAttemptsRef.current = 0;

    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    sourceRef.current?.disconnect();
    sourceRef.current = null;

    mutedGainRef.current?.disconnect();
    mutedGainRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;

    audioContextRef.current?.close();
    audioContextRef.current = null;

    socketRef.current?.close();
    socketRef.current = null;

    setConnected(false);
    setMicLive(false);
    addEvent("Microphone streaming stopped", "info");
  }

  function startDemo() {
    disconnect();

    setDemoMode(true);
    addEvent("Demo mode started", "info");

    let step = 0;

    const demoResults: AnalysisResult[] = [
      {
        risk_score: 18,
        action: "ALLOW",
        spoof_score: 0.08,
        speaker_similarity: 0.91,
        snr_db: 29,
        challenge_phrase: null,
        interlock_active: false,
      },
      {
        risk_score: 43,
        action: "CHALLENGE",
        spoof_score: 0.55,
        speaker_similarity: 0.68,
        snr_db: 18,
        challenge_phrase: "Say: Mango 8 nadi 4 blue",
        interlock_active: false,
      },
      {
        risk_score: 87,
        action: "BLOCK",
        spoof_score: 0.91,
        speaker_similarity: 0.38,
        snr_db: 11,
        challenge_phrase: null,
        interlock_active: true,
      },
    ];

    applyResult(demoResults[0]);

    demoTimerRef.current = setInterval(() => {
      step++;

      if (step < demoResults.length) {
        applyResult(demoResults[step]);
      } else {
        step = 0;
      }
    }, 4000);
  }

  useEffect(() => {
    return () => {
      manualCloseRef.current = true;

      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      socketRef.current?.close();

      if (demoTimerRef.current) {
        clearInterval(demoTimerRef.current);
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-[#121110] text-stone-100 selection:bg-amber-500/30 selection:text-white">
      {/* Replicated VaniRakshak Studio Taskbar (Matching User Design) */}
      <header className="fixed top-0 inset-x-0 z-50 border-b border-[#D8D2C6] bg-[#F0EDE2] shadow-xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 sm:px-10 py-3.5">
          {/* Logo & Brand Typography */}
          <div
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-3.5 cursor-pointer select-none group"
          >
            {/* Custom Shield Microphone Logo */}
            <svg
              className="h-11 w-9.5 text-[#403328] transition-transform duration-300 group-hover:scale-105"
              viewBox="0 0 38 44"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M19 2.5 C28 6.5 35 7.5 35 18 C35 29.5 27 38 19 42 C11 38 3 29.5 3 18 C3 7.5 10 6.5 19 2.5 Z"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinejoin="round"
              />
              {/* Mic Capsule */}
              <rect
                x="15"
                y="11"
                width="8"
                height="13"
                rx="4"
                stroke="currentColor"
                strokeWidth="1.9"
                fill="currentColor"
                fillOpacity="0.12"
              />
              {/* Cradle Arc */}
              <path
                d="M12 18 C12 23.5 26 23.5 26 18"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              {/* Stand */}
              <line
                x1="19"
                y1="23.5"
                x2="19"
                y2="28"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              <line
                x1="14"
                y1="28"
                x2="24"
                y2="28"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              {/* Audio waves left */}
              <line
                x1="8.5"
                y1="16.5"
                x2="8.5"
                y2="22.5"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              <line
                x1="5.5"
                y1="18.5"
                x2="5.5"
                y2="20.5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              {/* Audio waves right */}
              <line
                x1="29.5"
                y1="16.5"
                x2="29.5"
                y2="22.5"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              <line
                x1="32.5"
                y1="18.5"
                x2="32.5"
                y2="20.5"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>

            {/* Stacked Brand Name and Tagline */}
            <div className="flex flex-col justify-center">
              <div className="font-extrabold tracking-[0.14em] text-[#1E1A17] text-[15px] leading-tight uppercase font-sans">
                VAANI
              </div>
              <div className="font-extrabold tracking-[0.14em] text-[#1E1A17] text-[15px] leading-tight uppercase font-sans">
                RAKSHAK
              </div>
              <div className="text-[7.5px] font-semibold tracking-[0.24em] text-[#73685F] uppercase mt-0.5 leading-none font-sans">
                SAVING VOICES • SECURING TOMORROW
              </div>
            </div>
          </div>

          {/* Navigation Links from the Image */}
          <nav className="flex items-center gap-6 sm:gap-9 md:gap-11">
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
              className="text-xs sm:text-[13px] font-semibold tracking-[0.16em] uppercase text-[#4A433B] hover:text-[#1E1A17] transition-colors cursor-pointer"
            >
              HOME
            </button>

            <button
              onClick={() => {
                // Scroll to 3D pipeline breakdown
                window.scrollTo({ top: window.innerHeight * 0.9, behavior: "smooth" });
              }}
              className="text-xs sm:text-[13px] font-semibold tracking-[0.16em] uppercase text-[#4A433B] hover:text-[#1E1A17] transition-colors cursor-pointer"
            >
              FEATURES
            </button>

            <button
              onClick={() => {
                document.getElementById("console-dashboard")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-xs sm:text-[13px] font-semibold tracking-[0.16em] uppercase text-[#4A433B] hover:text-[#1E1A17] transition-colors cursor-pointer"
            >
              OUR AI
            </button>

            <button
              onClick={() => {
                document.getElementById("console-dashboard")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-xs sm:text-[13px] font-semibold tracking-[0.16em] uppercase text-[#4A433B] hover:text-[#1E1A17] transition-colors cursor-pointer"
            >
              IMPACT
            </button>

            <button
              onClick={() => {
                // Download telemetry audit report
                const report = {
                  project: "VaniRakshak",
                  timestamp: new Date().toISOString(),
                  systemState: connected ? "CONNECTED" : demoMode ? "DEMO_ACTIVE" : "STANDBY",
                  threatScore: risk,
                  decision: action,
                  spoofConfidence: spoofScore,
                  speakerSimilarity,
                  channelSNR: snr,
                  logEntries: events,
                };
                const blob = new Blob([JSON.stringify(report, null, 2)], {
                  type: "application/json",
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `vanirakshak-telemetry-${Date.now()}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="text-xs sm:text-[13px] font-semibold tracking-[0.16em] uppercase text-[#4A433B] hover:text-[#1E1A17] transition-colors cursor-pointer"
            >
              DOWNLOAD
            </button>

            {/* Subtle Live Telemetry Badge & Quick Controller */}
            <div className="hidden lg:flex items-center gap-2 pl-2 border-l border-[#D9D3C8]">
              <div
                onClick={connected || demoMode ? disconnect : startDemo}
                title={connected ? "Connected to Backend. Click to disconnect." : demoMode ? "Demo mode running. Click to stop." : "Click to run simulated demo"}
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-bold tracking-wider uppercase transition cursor-pointer border ${
                  connected
                    ? "border-emerald-600/40 bg-emerald-600/10 text-emerald-800 hover:bg-emerald-600/20"
                    : demoMode
                      ? "border-amber-600/40 bg-amber-600/10 text-amber-800 hover:bg-amber-600/20"
                      : "border-[#D9D3C8] bg-[#EDE8E0] text-[#6E6358] hover:bg-[#E2DDD3]"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    connected
                      ? "bg-emerald-600 animate-pulse"
                      : demoMode
                        ? "bg-amber-600 animate-pulse"
                        : "bg-[#8C7F72]"
                  }`}
                />
                <span>
                  {connected
                    ? "MIC LIVE"
                    : demoMode
                      ? "DEMO · SIMULATED"
                      : "STANDBY"}
                </span>
              </div>
            </div>
          </nav>
        </div>
      </header>

      {/* Immersive 3D Scrollytelling Visualizer */}
      <ScrollyVideoCanvas />

      {/* Real-time Security Console Anchor */}
      <div id="console-dashboard" className="relative z-10 scroll-mt-14 pt-8">
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-stone-800/80 pb-6">
            <div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-500" />
                <span className="font-mono text-xs uppercase tracking-widest text-amber-400 font-bold">
                  Active Monitoring Deck
                </span>
              </div>
              <h2 className="mt-1 text-2xl sm:text-3xl font-black tracking-tight text-stone-100">
                Live Acoustic Security Console
              </h2>
              <p className="mt-1 text-xs sm:text-sm text-stone-400">
                Real-time SASV speaker verification, neural vocoder spoof screening, and autonomous interlock
              </p>
            </div>

            {/* Quick action bar */}
            <div className="flex items-center gap-3">
              <button
                onClick={connectToBackend}
                disabled={connected}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition shadow-lg ${
                  connected
                    ? "border border-stone-800 bg-stone-900 text-stone-500 cursor-not-allowed"
                    : "bg-amber-500 hover:bg-amber-400 text-stone-950 shadow-amber-500/20 active:scale-95 cursor-pointer"
                }`}
              >
                <Mic className="h-3.5 w-3.5" />
                <span>{connected ? "Mic Stream Active" : "Connect Mic Stream"}</span>
              </button>

              <button
                onClick={startDemo}
                className="flex items-center gap-2 rounded-lg border border-stone-700 bg-stone-900 px-4 py-2 text-xs font-semibold text-stone-200 transition hover:bg-stone-800 active:scale-95 cursor-pointer"
              >
                <Play className="h-3.5 w-3.5 text-amber-400" />
                <span>Run Demo</span>
              </button>

              {(connected || demoMode) && (
                <button
                  onClick={disconnect}
                  className="flex items-center gap-2 rounded-lg border border-stone-700 bg-stone-900/80 px-4 py-2 text-xs font-semibold text-stone-300 transition hover:bg-rose-950/40 hover:border-rose-800/80 hover:text-rose-300 active:scale-95 cursor-pointer"
                >
                  <Square className="h-3.5 w-3.5 text-rose-400" />
                  <span>Disconnect</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        {/* Main dashboard grid */}
        <section className="grid gap-6 lg:grid-cols-3">
          {/* Risk meter */}
          <div className="rounded-2xl border border-stone-800/90 bg-[#181614] p-6 shadow-xl shadow-black/30 lg:col-span-1">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-amber-400" />
                <h3 className="font-bold text-stone-200 text-sm tracking-wide">Overall Threat Risk</h3>
              </div>
              <span className="font-mono text-xs text-stone-500">0–100 SCALE</span>
            </div>

            <div className="flex flex-col items-center py-6">
              <div className="relative flex h-52 w-52 items-center justify-center rounded-full border-[18px] border-stone-800/70">
                <div
                  className={`absolute inset-[-18px] rounded-full border-[18px] border-transparent transition-all duration-700 ${
                    risk >= 70
                      ? "border-t-rose-500"
                      : risk >= 35
                        ? "border-t-amber-500"
                        : "border-t-emerald-400"
                  }`}
                  style={{
                    transform: `rotate(${risk * 3.6}deg)`,
                  }}
                />

                <div className="text-center">
                  <div className="text-6xl font-black text-stone-100 tracking-tight">{risk}</div>
                  <div className="mt-1 font-mono text-[10px] uppercase tracking-wider text-stone-400">
                    RISK SCORE
                  </div>
                </div>
              </div>

              <div className="mt-6 text-center">
                <div
                  className={`text-lg font-black tracking-wide ${
                    risk >= 70
                      ? "text-rose-400"
                      : risk >= 35
                        ? "text-amber-400"
                        : "text-emerald-400"
                  }`}
                >
                  {riskLabel}
                </div>
                <div className="mt-1 font-mono text-xs text-stone-400">
                  DECISION: <span className="font-bold text-stone-200">{action}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Alert Center */}
          <div className="rounded-2xl border border-stone-800/90 bg-[#181614] p-6 shadow-xl shadow-black/30">
            <div className="flex items-center gap-2 mb-4">
              <AlertOctagon className="h-4 w-4 text-amber-400" />
              <h3 className="font-bold text-stone-200 text-sm tracking-wide">Alert Center & Dispatch</h3>
            </div>

            <div className="rounded-xl border border-stone-800/80 bg-[#100f0e] p-4">
              <div className="text-sm font-semibold text-stone-200">
                {alertMessage}
              </div>

              <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-lg border border-stone-800/80 bg-[#141210] p-2">
                  <div className="font-mono text-[10px] uppercase text-stone-500">UI Console</div>
                  <div className="mt-1 font-bold text-emerald-400">ACTIVE</div>
                </div>

                <div className="rounded-lg border border-stone-800/80 bg-[#141210] p-2">
                  <div className="font-mono text-[10px] uppercase text-stone-500">Webhook</div>
                  <div className="mt-1 font-bold text-amber-400">SIMULATED</div>
                </div>

                <div className="rounded-lg border border-stone-800/80 bg-[#141210] p-2">
                  <div className="font-mono text-[10px] uppercase text-stone-500">SMS / Email</div>
                  <div className="mt-1 font-bold text-amber-400">SIMULATED</div>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-stone-800/80 bg-[#141210] p-3">
                <div className="text-[10px] uppercase font-mono tracking-wider text-stone-500">
                  Simulated Dispatch Payload
                </div>

                <div className="mt-1.5 font-mono text-xs text-stone-300">
                  {alertPayload || "No active security alert payload"}
                </div>
              </div>
            </div>
          </div>

          {/* Explainability Evidence */}
          <div className="rounded-2xl border border-stone-800/90 bg-[#181614] p-6 shadow-xl shadow-black/30 lg:col-span-2">
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-amber-400" />
                <h3 className="font-bold text-stone-200 text-sm tracking-wide">
                  Explainability Evidence & Signal Breakdown
                </h3>
              </div>
              <span className="font-mono text-xs text-stone-500">4 CORE VECTORS</span>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <EvidenceCard
                title="Deepfake / Spoof"
                value={`${Math.round(spoofScore * 100)}%`}
                description="Synthetic vocoder likelihood"
              />

              <EvidenceCard
                title="Speaker Similarity"
                value={`${Math.round(speakerSimilarity * 100)}%`}
                description="ECAPA-TDNN reference match"
              />

              <EvidenceCard
                title="Channel SNR"
                value={`${snr.toFixed(1)} dB`}
                description="Acoustic background clarity"
              />

              <EvidenceCard
                title="Live Audio Level"
                value={`${Math.round(audioLevel)}%`}
                description={micLive ? "16kHz PCM stream" : "Microphone idle"}
              />
            </div>

            <div className="mt-5 rounded-xl border border-stone-800/80 bg-[#100f0e] p-4">
              <div className="mb-1 text-[10px] font-mono font-bold uppercase tracking-wider text-amber-400">
                Decision Matrix Logic
              </div>

              <p className="text-xs sm:text-sm leading-relaxed text-stone-300">
                VaniRakshak correlates spectral phase artifacts, speaker acoustic embeddings, and channel noise profiles in real time before releasing or isolating the call stream.
              </p>
            </div>
          </div>
        </section>

        {/* Interlock + Dynamic Challenge */}
        <section className="grid gap-6 lg:grid-cols-2">
          {/* Transaction interlock */}
          <div
            className={`rounded-2xl border p-6 shadow-xl shadow-black/30 transition-all duration-300 ${
              transactionLocked
                ? "border-rose-900/60 bg-rose-950/20"
                : "border-stone-800/90 bg-[#181614]"
            }`}
          >
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Lock className={`h-4 w-4 ${transactionLocked ? "text-rose-400" : "text-emerald-400"}`} />
                <div>
                  <h3 className="font-bold text-stone-100 text-sm">Pre-Transaction Interlock</h3>
                  <p className="text-xs text-stone-400">Autonomous risk enforcement</p>
                </div>
              </div>

              <span
                className={`rounded-full px-3 py-1 font-mono text-xs font-bold tracking-wider ${
                  transactionLocked
                    ? "bg-rose-500/15 border border-rose-500/30 text-rose-400"
                    : "bg-emerald-500/15 border border-emerald-500/30 text-emerald-400"
                }`}
              >
                {transactionLocked ? "LOCKED" : "ARMED / READY"}
              </span>
            </div>

            <div className="rounded-xl border border-stone-800/80 bg-[#100f0e] p-5">
              <div className="mb-4 flex justify-between text-sm">
                <span className="text-stone-400">Protected Transaction:</span>
                <span className="font-mono font-bold text-stone-100">Wire Transfer ₹50,000</span>
              </div>

              <button
                disabled={transactionLocked}
                className={`w-full rounded-lg px-4 py-3 font-bold text-sm transition shadow-lg ${
                  transactionLocked
                    ? "cursor-not-allowed bg-rose-950/80 border border-rose-800/50 text-rose-300"
                    : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/20 cursor-pointer active:scale-95"
                }`}
              >
                {transactionLocked
                  ? "🔒 Security Interlock Active — Transfer Frozen"
                  : "✓ Authorize Transaction"}
              </button>

              <p className="mt-3 text-center text-xs text-stone-500">
                {transactionLocked
                  ? "Acoustic spoof suspicion triggered safety quarantine."
                  : "Call verified within safe biological acoustic baseline."}
              </p>
            </div>
          </div>

          {/* Dynamic Challenge */}
          <div className="rounded-2xl border border-stone-800/90 bg-[#181614] p-6 shadow-xl shadow-black/30">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-stone-100 text-sm">Active Challenge-Response</h3>
                <p className="text-xs text-stone-400">Anti-replay & latency tripwire</p>
              </div>
              <span className="font-mono text-xs text-amber-400 font-semibold">
                {challenge ? "CHALLENGE PENDING" : "STANDBY"}
              </span>
            </div>

            <div className="rounded-xl border border-dashed border-stone-700/80 bg-[#100f0e] p-6">
              {challenge ? (
                <>
                  <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-amber-400">
                    Acoustic Challenge Verification Prompt:
                  </div>
                  <p className="mt-2 text-xl font-mono font-black leading-relaxed text-stone-100">
                    {challenge}
                  </p>
                  <p className="mt-2 text-xs text-stone-400">
                    Caller must articulate the dynamic phrase above to satisfy the VAD tripwire.
                  </p>
                </>
              ) : (
                <div className="text-center py-3">
                  <p className="text-xs text-stone-400 leading-relaxed">
                    No active challenge requested. When spoof likelihood crosses the suspicion threshold (risk ≥ 35), a dynamic cryptographic challenge phrase is automatically assigned.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Live event log */}
        <section className="rounded-2xl border border-stone-800/90 bg-[#181614] p-6 shadow-xl shadow-black/30">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h3 className="font-bold text-stone-100 text-sm">Live System Audit Log</h3>
              <p className="text-xs text-stone-400">Real-time classification telemetry</p>
            </div>

            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-stone-400">
                {events.length} events logged
              </span>

              {events.length > 5 && (
                <button
                  onClick={() => setShowAllEvents((value) => !value)}
                  className="rounded-lg border border-stone-700 bg-stone-900/80 px-3 py-1 text-xs text-stone-300 hover:border-stone-500 hover:text-white transition cursor-pointer"
                >
                  {showAllEvents ? "Show Recent" : "Show All"}
                </button>
              )}
            </div>
          </div>

          {events.length === 0 ? (
            <div className="rounded-xl border border-dashed border-stone-800/80 p-8 text-center text-xs text-stone-500 font-mono">
              Waiting for incoming audio stream telemetry...
            </div>
          ) : (
            <div className="space-y-2">
              {(showAllEvents ? events : events.slice(0, 5)).map((event) => (
                <div
                  key={event.id}
                  className="flex items-center gap-4 rounded-lg border border-stone-800/80 bg-[#100f0e] px-4 py-2.5 transition"
                >
                  <span
                    className={`h-2 w-2 rounded-full shrink-0 ${
                      event.severity === "critical"
                        ? "bg-rose-500 shadow-sm shadow-rose-500/50"
                        : event.severity === "warning"
                          ? "bg-amber-400 shadow-sm shadow-amber-400/50"
                          : "bg-emerald-400 shadow-sm shadow-emerald-400/50"
                    }`}
                  />

                  <span className="w-20 font-mono text-xs text-stone-500 shrink-0">
                    {event.time}
                  </span>

                  <span className="text-xs text-stone-300 font-medium">
                    {event.message}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function EvidenceCard({
  title,
  value,
  description,
}: {
  title: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-stone-800/80 bg-[#100f0e] p-4 shadow-sm">
      <div className="text-xs font-semibold text-stone-400 tracking-wide">{title}</div>
      <div className="mt-2 text-2xl font-black text-stone-100 tracking-tight">{value}</div>
      <div className="mt-1 text-[11px] text-stone-500">{description}</div>
    </div>
  );
}