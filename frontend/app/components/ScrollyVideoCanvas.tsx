"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Activity,
  Cpu,
  Lock,
  Radio,
} from "lucide-react";

const TOTAL_FRAMES = 220;

// Scrollytelling stage definitions mapped to scroll progress ranges
interface StageInfo {
  id: string;
  step: string;
  title: string;
  subtitle: string;
  badge: string;
  description: string;
  stats: { label: string; value: string }[];
  startProgress: number;
  endProgress: number;
  icon: typeof Activity;
}

const STAGES: StageInfo[] = [
  {
    id: "stage-ingest",
    step: "01",
    badge: "STAGE 01 • INGESTION",
    title: "Edge-First Audio Ingestion",
    subtitle: "Lossless acoustic capture & spectral baseline calibration",
    description:
      "VaniRakshak hooks directly into the incoming raw audio stream, sampling micro-transients and background acoustic ambient profiles with zero buffer lag.",
    stats: [
      { label: "Capture Latency", value: "< 12 ms" },
      { label: "Sample Depth", value: "16-bit PCM" },
      { label: "Channel Isolation", value: "Real-time" },
    ],
    startProgress: 0.02,
    endProgress: 0.24,
    icon: Radio,
  },
  {
    id: "stage-spectral",
    step: "02",
    badge: "STAGE 02 • MULTI-TIER ANALYSIS",
    title: "Deep Spectral Inspection",
    subtitle: "Frequency domain decomposition & artifact tracking",
    description:
      "Instantaneous Short-Time Fourier Transform (STFT) and bi-spectral analysis search for synthetic vocoder phase discontinuities, synthetic smoothing, and neural speech footprint.",
    stats: [
      { label: "Inference Window", value: "1.5s Sliding" },
      { label: "Feature Vectors", value: "Acoustic + FFT" },
      { label: "Detection Engine", value: "AASIST / LFCC" },
    ],
    startProgress: 0.26,
    endProgress: 0.49,
    icon: Activity,
  },
  {
    id: "stage-spoof",
    step: "03",
    badge: "STAGE 03 • BIOMETRIC DEFENSE",
    title: "Acoustic Spoof Detection",
    subtitle: "Synthetic voice scoring & speaker resonance verification",
    description:
      "Cross-analyzes vocal tract resonance and authentic biological micro-jitter against generative voice synthesis models (ElevenLabs, Bark, VALL-E, XTTS).",
    stats: [
      { label: "Spoof Confidence", value: "Multi-Model" },
      { label: "Speaker Sim", value: "Cosine Distance" },
      { label: "False Reject Rate", value: "< 0.4%" },
    ],
    startProgress: 0.51,
    endProgress: 0.74,
    icon: Cpu,
  },
  {
    id: "stage-interlock",
    step: "04",
    badge: "STAGE 04 • AUTONOMOUS ACTION",
    title: "Real-Time Interlock & Quarantine",
    subtitle: "Zero-latency mitigation before transaction commit",
    description:
      "Autonomous risk adjudication immediately enforces security policy: triggering an interactive dynamic challenge phrase or instantly isolating the suspicious channel.",
    stats: [
      { label: "Decision Time", value: "< 45 ms" },
      { label: "Action Options", value: "Allow / Challenge / Block" },
      { label: "Interlock", value: "Hardware & API" },
    ],
    startProgress: 0.76,
    endProgress: 0.98,
    icon: Lock,
  },
];

export default function ScrollyVideoCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Array of loaded images
  const imagesRef = useRef<(HTMLImageElement | null)[]>([]);
  const targetFrameRef = useRef<number>(0);
  const currentFrameRef = useRef<number>(0);
  const animFrameIdRef = useRef<number | null>(null);

  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeStageIndex, setActiveStageIndex] = useState(0);

  // Helper to get formatted frame path
  const getFrameUrl = (frameIndex: number) => {
    const frameNum = Math.min(TOTAL_FRAMES, Math.max(1, frameIndex + 1));
    const padNum = String(frameNum).padStart(4, "0");
    return `/frames/frame_${padNum}.webp`;
  };

  // Draw a specific frame onto the canvas
  const drawFrame = useCallback((frameIndex: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    // Find requested frame or nearest loaded frame
    let img: HTMLImageElement | null = imagesRef.current[frameIndex] || null;
    if (!img || !img.complete || img.naturalWidth === 0) {
      for (let offset = 1; offset < TOTAL_FRAMES; offset++) {
        const left = frameIndex - offset;
        const right = frameIndex + offset;
        if (left >= 0 && imagesRef.current[left]?.complete) {
          img = imagesRef.current[left];
          break;
        }
        if (right < TOTAL_FRAMES && imagesRef.current[right]?.complete) {
          img = imagesRef.current[right];
          break;
        }
      }
    }

    if (!img || !img.complete || img.naturalWidth === 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const canvasWidth = canvas.clientWidth;
    const canvasHeight = canvas.clientHeight;

    // Ensure internal resolution matches client size * dpr
    if (
      canvas.width !== Math.floor(canvasWidth * dpr) ||
      canvas.height !== Math.floor(canvasHeight * dpr)
    ) {
      canvas.width = Math.floor(canvasWidth * dpr);
      canvas.height = Math.floor(canvasHeight * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    // Calculate "cover" scale and positioning
    const imgRatio = img.naturalWidth / img.naturalHeight;
    const canvasRatio = canvasWidth / canvasHeight;

    let drawWidth = canvasWidth;
    let drawHeight = canvasHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (canvasRatio > imgRatio) {
      drawWidth = canvasWidth;
      drawHeight = canvasWidth / imgRatio;
      offsetY = (canvasHeight - drawHeight) / 2;
    } else {
      drawHeight = canvasHeight;
      drawWidth = canvasHeight * imgRatio;
      offsetX = (canvasWidth - drawWidth) / 2;
    }

    // High quality image smoothing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

    // Fully cover the background video's baked-in taskbar with seamless background color
    const bakedNavHeight = Math.max(0, offsetY) + 154 * (drawHeight / 1080);
    ctx.fillStyle = "#F0EDE2";
    ctx.fillRect(0, 0, canvasWidth, bakedNavHeight);

    ctx.restore();
  }, []);

  // Frame loading pipeline
  useEffect(() => {
    imagesRef.current = new Array(TOTAL_FRAMES).fill(null);
    let loadedCount = 0;

    // Priority 1: Load First Frame Immediately
    const firstImg = new Image();
    firstImg.src = getFrameUrl(0);
    firstImg.onload = () => {
      imagesRef.current[0] = firstImg;
      loadedCount++;
      setLoadingProgress(Math.round((loadedCount / TOTAL_FRAMES) * 100));
      setIsReady(true);
      drawFrame(0);

      // Priority 2: Load keyframes across the timeline (every 8th frame)
      const keyframeIndices: number[] = [];
      for (let i = 8; i < TOTAL_FRAMES; i += 8) {
        keyframeIndices.push(i);
      }

      keyframeIndices.forEach((idx) => {
        const img = new Image();
        img.src = getFrameUrl(idx);
        img.onload = () => {
          imagesRef.current[idx] = img;
          loadedCount++;
          setLoadingProgress(Math.round((loadedCount / TOTAL_FRAMES) * 100));
        };
      });

      // Priority 3: Load the rest in batches
      const remainingIndices: number[] = [];
      for (let i = 1; i < TOTAL_FRAMES; i++) {
        if (i % 8 !== 0) {
          remainingIndices.push(i);
        }
      }

      let currentBatchIdx = 0;
      const BATCH_SIZE = 12;

      function loadNextBatch() {
        if (currentBatchIdx >= remainingIndices.length) return;
        const nextIndices = remainingIndices.slice(
          currentBatchIdx,
          currentBatchIdx + BATCH_SIZE
        );
        currentBatchIdx += BATCH_SIZE;

        let batchLoaded = 0;
        nextIndices.forEach((idx) => {
          const img = new Image();
          img.src = getFrameUrl(idx);
          img.onload = () => {
            imagesRef.current[idx] = img;
            loadedCount++;
            setLoadingProgress(Math.round((loadedCount / TOTAL_FRAMES) * 100));
            batchLoaded++;
            if (batchLoaded === nextIndices.length) {
              // Idle callback or small timeout for smooth performance
              if ("requestIdleCallback" in window) {
                (window as Window).requestIdleCallback(loadNextBatch);
              } else {
                setTimeout(loadNextBatch, 16);
              }
            }
          };
          img.onerror = () => {
            batchLoaded++;
            if (batchLoaded === nextIndices.length) {
              setTimeout(loadNextBatch, 16);
            }
          };
        });
      }

      loadNextBatch();
    };

    return () => {
      imagesRef.current = [];
    };
  }, [drawFrame]);

  // Handle Smooth Scroll & Frame Interpolation (Lerping)
  useEffect(() => {
    let lastRenderedFrame = -1;

    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const maxScroll = rect.height - window.innerHeight;

      if (maxScroll <= 0) return;

      const currentScroll = -rect.top;
      const rawProgress = Math.max(0, Math.min(1, currentScroll / maxScroll));
      setScrollProgress(rawProgress);

      const target = Math.min(
        TOTAL_FRAMES - 1,
        Math.max(0, Math.round(rawProgress * (TOTAL_FRAMES - 1)))
      );
      targetFrameRef.current = target;

      // Update active stage indicator
      const activeIdx = STAGES.findIndex(
        (s) => rawProgress >= s.startProgress && rawProgress <= s.endProgress
      );
      if (activeIdx !== -1) {
        setActiveStageIndex(activeIdx);
      }
    };

    // Animation render loop
    const renderLoop = () => {
      // Lerp factor for buttery smooth scrub feel
      const delta = targetFrameRef.current - currentFrameRef.current;
      if (Math.abs(delta) > 0.02) {
        currentFrameRef.current += delta * 0.16;
      } else {
        currentFrameRef.current = targetFrameRef.current;
      }

      const frameToDraw = Math.round(currentFrameRef.current);
      if (frameToDraw !== lastRenderedFrame) {
        drawFrame(frameToDraw);
        lastRenderedFrame = frameToDraw;
      }

      animFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll, { passive: true });
    handleScroll();
    animFrameIdRef.current = requestAnimationFrame(renderLoop);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [drawFrame]);

  // Scroll smoothly to a specific stage in the sequence
  const scrollToStage = (stage: StageInfo) => {
    if (!containerRef.current) return;
    const containerTop =
      containerRef.current.getBoundingClientRect().top + window.scrollY;
    const maxScroll =
      containerRef.current.scrollHeight - window.innerHeight;
    const targetScrollY =
      containerTop + stage.startProgress * maxScroll + 10;

    window.scrollTo({
      top: targetScrollY,
      behavior: "smooth",
    });
  };

  // Jump to live console dashboard
  const scrollToConsole = () => {
    const el = document.getElementById("console-dashboard");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[460vh] bg-slate-950 select-none"
    >
      {/* Sticky Fullscreen Stage */}
      <div className="sticky top-0 h-screen w-full overflow-hidden flex items-center justify-center bg-[#F0EDE2]">
        {/* Canvas Visualizer */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-700"
          style={{ opacity: isReady ? 1 : 0 }}
        />

        {/* Loading Progress Bar (Top edge) */}
        {loadingProgress < 100 && (
          <div className="absolute top-0 inset-x-0 z-40 h-1 bg-stone-300/30 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 via-amber-400 to-emerald-400 transition-all duration-300"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        )}

        {/* Scrollytelling Stage Overlays - Docked to the Right Side */}
        <div className="relative z-20 w-full h-full pointer-events-none">
          {STAGES.map((stage) => {
            // Stage opacity calculation based on scroll progress
            const mid = (stage.startProgress + stage.endProgress) / 2;
            const span = (stage.endProgress - stage.startProgress) / 2;
            const dist = Math.abs(scrollProgress - mid);
            // Smooth bell-curve opacity
            const opacity = Math.max(
              0,
              Math.min(1, 1 - Math.pow(dist / span, 1.8))
            );
            const translateY = (scrollProgress - mid) * 40;

            if (opacity < 0.01) return null;

            const IconComp = stage.icon;

            return (
              <div
                key={stage.id}
                className="absolute right-4 sm:right-8 md:right-12 top-1/2 -translate-y-1/2 w-[calc(100%-2rem)] max-w-[300px] sm:max-w-[330px] transition-transform duration-75 pointer-events-auto z-30"
                style={{
                  opacity,
                  transform: `translateY(calc(-50% + ${translateY}px))`,
                }}
              >
                <div className="rounded-2xl border border-stone-700/70 bg-[#181614]/94 p-4 sm:p-5 backdrop-blur-2xl shadow-2xl shadow-black/80 text-stone-100">
                  {/* Badge */}
                  <div className="flex items-center justify-between gap-2 mb-2.5">
                    <div className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[10px] font-bold tracking-wider text-amber-400 uppercase">
                      <IconComp className="h-3 w-3" />
                      {stage.badge}
                    </div>
                    <span className="text-[11px] font-mono text-stone-400 font-bold">
                      {stage.step} / 04
                    </span>
                  </div>

                  {/* Title & Subtitle */}
                  <h3 className="text-lg sm:text-xl font-black tracking-tight text-stone-100 mb-1 leading-snug">
                    {stage.title}
                  </h3>
                  <p className="text-xs font-semibold text-amber-300/90 mb-2 leading-tight">
                    {stage.subtitle}
                  </p>
                  <p className="text-xs text-stone-300 leading-relaxed mb-4">
                    {stage.description}
                  </p>

                  {/* Micro Stats Grid */}
                  <div className="grid grid-cols-3 gap-1.5 border-t border-stone-800/80 pt-3">
                    {stage.stats.map((st, sIdx) => (
                      <div
                        key={sIdx}
                        className="rounded-lg border border-stone-800/80 bg-[#121110]/90 p-2 text-center backdrop-blur-sm"
                      >
                        <div className="text-[9px] uppercase font-mono tracking-wider text-stone-400 mb-0.5 truncate">
                          {st.label}
                        </div>
                        <div className="text-[11px] sm:text-xs font-bold text-stone-100 truncate">
                          {st.value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>


        {/* Left-Hand Scrubber / Timeline Nav */}
        <div className="absolute left-4 sm:left-8 top-1/2 -translate-y-1/2 z-30 hidden md:flex flex-col items-start gap-3 pointer-events-auto">
          {STAGES.map((st, i) => {
            const isActive = activeStageIndex === i;
            return (
              <button
                key={st.id}
                onClick={() => scrollToStage(st)}
                className="group flex items-center gap-2.5 transition-all text-left"
              >
                <div
                  className={`h-2.5 rounded-full transition-all duration-300 ${
                    isActive
                      ? "w-8 bg-amber-500 shadow-md shadow-amber-500/50"
                      : "w-2.5 bg-stone-400/50 group-hover:bg-stone-600"
                  }`}
                />
                <span
                  className={`text-[11px] font-mono font-bold transition-all ${
                    isActive
                      ? "text-amber-500 font-extrabold translate-x-0 opacity-100"
                      : "text-stone-500 opacity-0 group-hover:opacity-100 group-hover:translate-x-0 -translate-x-2"
                  }`}
                >
                  {st.step} • {st.title.split(" ")[0]}
                </span>
              </button>
            );
          })}
        </div>

        {/* Bottom Seamless Gradient Transition to Dark Console */}
        <div
          className="absolute bottom-0 inset-x-0 h-48 z-20 pointer-events-none transition-opacity duration-300"
          style={{
            background:
              "linear-gradient(to bottom, transparent 0%, rgba(18, 17, 16, 0.4) 40%, rgba(18, 17, 16, 0.95) 85%, #121110 100%)",
            opacity: Math.min(1, Math.max(0, (scrollProgress - 0.75) * 4)),
          }}
        />

        {/* Bottom Control Bar with Frame counter & Scrub percent */}
        <div
          className="absolute bottom-6 inset-x-0 z-30 flex items-center justify-between px-6 sm:px-10 text-xs font-mono pointer-events-none transition-opacity duration-300"
          style={{
            opacity: scrollProgress > 0.95 ? 0 : 1,
          }}
        >
          <div className="ml-12 rounded-full border border-stone-400/30 bg-[#F6F5F2]/90 px-3 py-1 text-stone-800 backdrop-blur-md font-semibold shadow-sm">
            FRAME {String(Math.min(TOTAL_FRAMES, Math.max(1, Math.round(scrollProgress * (TOTAL_FRAMES - 1)) + 1))).padStart(3, "0")} / {TOTAL_FRAMES}
          </div>

          <button
            onClick={scrollToConsole}
            className="pointer-events-auto rounded-full bg-stone-900/90 hover:bg-stone-950 text-stone-100 px-4 py-2 font-sans font-semibold text-xs border border-stone-700/60 backdrop-blur-md shadow-lg transition-all hover:scale-105"
          >
            Launch Live Mic Console ↓
          </button>

          <div className="rounded-full border border-stone-400/30 bg-[#F6F5F2]/90 px-3 py-1 text-stone-800 backdrop-blur-md font-semibold shadow-sm">
            {Math.round(scrollProgress * 100)}% EXPLORED
          </div>
        </div>
      </div>
    </div>
  );
}
