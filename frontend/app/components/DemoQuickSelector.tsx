"use client";

import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, UserX, Play, CheckCircle2 } from "lucide-react";

export interface BenchmarkItem {
  id: string;
  title: string;
  subtitle: string;
  fileName: string;
  audioUrl: string;
  expectedVerdict: "AUTHENTIC_TARGET" | "SYNTHETIC_VOICE_CLONE" | "SUSPICIOUS_SPLICED_AUDIO" | "IMPOSTOR_SPEAKER";
  badge: string;
  color: "emerald" | "rose" | "amber" | "purple";
  description: string;
  keyDetails: string[];
}

export const BENCHMARKS: BenchmarkItem[] = [
  {
    id: "demo_01",
    title: "Authentic target voice.",
    subtitle: "Reference target speaker",
    fileName: "demo_01_genuine.wav",
    audioUrl: "/demo/demo_01_genuine.wav",
    expectedVerdict: "AUTHENTIC_TARGET",
    badge: "BENCHMARK 1 • BONA FIDE",
    color: "emerald",
    description: "Unaltered reference speech recording from the enrolled speaker anchor with natural physiological vocal tract tremor.",
    keyDetails: ["Natural vocal tract micro-tremor", "High anchor similarity (0.72)", "Zero vocoder phase jumps"],
  },
  {
    id: "demo_02",
    title: "Neural vocoder clone.",
    subtitle: "Target voice impersonation",
    fileName: "demo_02_full_clone.wav",
    audioUrl: "/demo/demo_02_full_clone.wav",
    expectedVerdict: "SYNTHETIC_VOICE_CLONE",
    badge: "BENCHMARK 2 • AI CLONE",
    color: "rose",
    description: "Synthetic voice clone mimicking the target voice, characterized by high-band vocoder spectral flattening and collapsed latent variance.",
    keyDetails: ["High-band flatness > 0.70", "Target identity mimicked (0.67)", "Collapsed latent entropy"],
  },
  {
    id: "demo_03",
    title: "Partial splice injection.",
    subtitle: "Genuine / synthetic composite",
    fileName: "demo_03_partial_splice.wav",
    audioUrl: "/demo/demo_03_partial_splice.wav",
    expectedVerdict: "SUSPICIOUS_SPLICED_AUDIO",
    badge: "BENCHMARK 3 • SPLICED",
    color: "amber",
    description: "Authentic recording spliced with synthetic speech insertion; reveals localized vocoder phase discontinuity at 7.4 seconds.",
    keyDetails: ["Phase discontinuity at 7.4s", "Segment risk variance spike", "Dual acoustic signature"],
  },
  {
    id: "demo_04",
    title: "Zero-shot impostor.",
    subtitle: "Unenrolled identity fraud",
    fileName: "demo_04_impostor.wav",
    audioUrl: "/demo/demo_04_impostor.wav",
    expectedVerdict: "IMPOSTOR_SPEAKER",
    badge: "BENCHMARK 4 • IMPOSTOR",
    color: "purple",
    description: "Unenrolled natural speaker attempting to claim the target identity; rejected by ECAPA-TDNN biometric cosine verification.",
    keyDetails: ["Negative cosine similarity (-0.04)", "Biometric mismatch rejected", "Zero-retention logged"],
  },
];

interface DemoQuickSelectorProps {
  selectedId: string | null;
  onSelect: (item: BenchmarkItem) => void;
  isAnalyzing: boolean;
}

export default function DemoQuickSelector({ selectedId, onSelect, isAnalyzing }: DemoQuickSelectorProps) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 border-b border-white/[0.06] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-wider text-[#a1a1a1]">
              STANDARDIZED BENCHMARK SUITE
            </span>
          </div>
          <h3 className="mt-1 text-lg font-semibold tracking-[-0.6px] text-white">
            Pre-configured evaluation files.
          </h3>
          <p className="mt-0.5 text-xs text-[#888888]">
            One-click evaluation files covering bona fide, neural vocoder clone, splice tampering, and biometric impostor vectors.
          </p>
        </div>
        <span className="self-start sm:self-auto rounded-full border border-white/[0.08] bg-[#171717] px-3 py-1 font-mono text-[11px] text-[#a1a1a1]">
          16 kHz mono • Enrolled: Narendra Modi
        </span>
      </div>

      <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-4">
        {BENCHMARKS.map((item) => {
          const isSelected = selectedId === item.id;
          const colorStyles = {
            emerald: {
              border: isSelected ? "border-emerald-500/80 bg-[#111814]" : "border-white/[0.08] hover:border-emerald-500/40 bg-[#121212]",
              badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
              icon: <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />,
              btn: isSelected ? "bg-emerald-500 text-black font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            rose: {
              border: isSelected ? "border-rose-500/80 bg-[#1a1113]" : "border-white/[0.08] hover:border-rose-500/40 bg-[#121212]",
              badge: "border-rose-500/30 bg-rose-500/10 text-rose-400",
              icon: <ShieldAlert className="h-4 w-4 text-rose-400 shrink-0" />,
              btn: isSelected ? "bg-rose-500 text-white font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            amber: {
              border: isSelected ? "border-amber-500/80 bg-[#1a1610]" : "border-white/[0.08] hover:border-amber-500/40 bg-[#121212]",
              badge: "border-amber-500/30 bg-amber-500/10 text-amber-400",
              icon: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />,
              btn: isSelected ? "bg-amber-500 text-black font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            purple: {
              border: isSelected ? "border-purple-500/80 bg-[#17111a]" : "border-white/[0.08] hover:border-purple-500/40 bg-[#121212]",
              badge: "border-purple-500/30 bg-purple-500/10 text-purple-400",
              icon: <UserX className="h-4 w-4 text-purple-400 shrink-0" />,
              btn: isSelected ? "bg-purple-500 text-white font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
          }[item.color];

          return (
            <div
              key={item.id}
              onClick={() => !isAnalyzing && onSelect(item)}
              className={`relative flex flex-col justify-between rounded-lg border p-4 transition-all duration-150 cursor-pointer ${colorStyles.border} ${
                isSelected ? "ring-1 ring-white/20" : ""
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2.5">
                  <span className={`rounded-full border px-2 py-0.5 font-mono text-[9.5px] font-medium tracking-wider uppercase ${colorStyles.badge}`}>
                    {item.badge}
                  </span>
                  {isSelected && (
                    <span className="flex items-center gap-1 font-mono text-[10px] text-white">
                      <CheckCircle2 className="h-3 w-3 text-emerald-400" /> Active
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1 rounded bg-black/40 border border-white/[0.06]">
                    {colorStyles.icon}
                  </div>
                  <div>
                    <h4 className="font-semibold text-sm tracking-[-0.3px] text-white">
                      {item.title}
                    </h4>
                    <p className="font-mono text-[11px] text-[#888888]">
                      {item.subtitle}
                    </p>
                  </div>
                </div>

                <p className="text-xs leading-relaxed text-[#a1a1a1] mt-2.5">
                  {item.description}
                </p>

                <div className="mt-3 space-y-1 border-t border-white/[0.06] pt-2.5 font-mono text-[10.5px] text-[#888888]">
                  {item.keyDetails.map((detail, idx) => (
                    <div key={idx} className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-[#555]" />
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 pt-1">
                <button
                  type="button"
                  disabled={isAnalyzing}
                  className={`w-full flex items-center justify-center gap-1.5 rounded-full py-1.5 px-3 text-xs font-medium transition shadow-sm ${colorStyles.btn} disabled:opacity-40 cursor-pointer`}
                >
                  <Play className="h-3 w-3 fill-current" />
                  <span>{isSelected ? "Re-evaluate" : "Run evaluation"}</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
