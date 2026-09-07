# VaniRakshak Data & Benchmark Assets

This directory houses the standardized forensic evaluation datasets, speaker enrollment anchor profiles, and preprocessing scripts for VaniRakshak.

## Structure

```
data/
├── anchor_speaker_embedding.npy  # 192-dim ECAPA-TDNN reference anchor embedding
├── anchor_speaker_profile.json   # Cryptographic DPDP Act 2023 enrollment receipt & metadata
├── consent/                      # Digital consent and statutory recording disclosures
├── demo/                         # Standardized public benchmark audio suite
│   ├── demo_01_genuine.wav       # 10s bona fide target speech (Narendra Modi)
│   ├── demo_02_full_clone.wav    # 10s vocoded / neural clone speech
│   ├── demo_03_partial_splice.wav# 10s spliced speech (genuine -> clone -> genuine)
│   ├── demo_04_impostor.wav      # 10s zero-shot impostor speech
│   └── ground_truth.json         # Ground-truth forensic annotations & expected verdicts
├── scripts/                      # Data engineering and evaluation split scripts
│   ├── create_demo_splice.py     # Generates benchmark audio files from source recordings
│   ├── make_splits.py            # Generates train/dev/eval splits for ML benchmarks
│   └── standardize.py            # In-memory 16 kHz mono standardization & DSP analysis
└── splits/                       # Generated dataset protocol splits
```

## Privacy & DPDP Compliance

Per **Section 8(7) of the India Digital Personal Data Protection (DPDP) Act 2023**:
- No raw voice recordings are stored or retained on disk during live inference.
- Speaker profiles are stored exclusively as 192-dimensional non-invertible latent embeddings.
- Every biometric transaction emits a SHA-256 hash-chained cryptographic audit receipt.
