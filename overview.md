# VaniRakshak Flowchart Fix - Overview

## What was done
- Repaired all Mermaid syntax in `docs/FLOWCHARTS_CORRECTED.md`.
- Rebuilt `docs/flowcharts.html` using Mermaid **10.9.8** explicitly.
- Corrected both main flowcharts and the condensed ML-only flowchart.

## Key fixes
- Quoted every node label.
- Replaced invalid `br` text with valid `<br/>` line breaks.
- Removed parser-sensitive formulas, parentheses, shorthand junctions, and special symbols from labels.
- Declared nodes within their intended subgraphs.
- Kept the diagrams consistent with the actual audio-forensic VaniRakshak implementation.

## Verification
All three Mermaid blocks were parsed programmatically with the exact `mermaid@10.9.8` package:

- Diagram 1: PASS
- Diagram 2: PASS
- Canonical ML-only diagram: PASS

The standalone HTML viewer is ready for review at `docs/flowcharts.html`.

## README update
- Added the canonical ML-side architecture diagram to `README.md` under **How it works**.
- Added an explicit note that VaniRakshak does not use STT, NER, BERT, emotion classification, or LLM-RAG.
- Verified the README diagram separately: **PASS with Mermaid 10.9.8**.
