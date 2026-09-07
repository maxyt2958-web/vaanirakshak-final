# VaniRakshak Machine Learning & Forensic Benchmark Suite

Target-conditioned neural anti-spoofing models, ASVspoof-5 evaluation metrics, and telephony codec robustness benchmarks.

## Architecture & Modules

```
ml/
├── model.py              # AcousticForensicClassifier (BiGRU + Attentive Stats Pooling)
├── loss.py               # ForensicDetectionCostLoss (smooth minDCF) & AdditiveMarginSoftmaxLoss
├── dataset.py            # ForensicAudioDataset with G.711 / AMR-WB / GSM-EFR augmentations
├── train.py              # End-to-end training and checkpoint saving
├── evaluate.py           # Comprehensive ASVspoof-5 and telephony benchmark runner
├── extract_features.py   # Batch 768-dim WavLM & 192-dim ECAPA vector extraction
├── checkpoints/          # Model weights & calibrated thresholds.json
└── requirements.txt      # PyTorch, NumPy, SciPy dependencies
```

## Running Benchmarks

### 1. Unified Forensic Benchmark Battery
Runs ASVspoof-5 metrics, CPU latency SLA, telephony codec robustness, and demo file ground truth:
```bash
python benchmarks/run_benchmarks.py
```

### 2. ML Evaluation & Threshold Calibration
Evaluates anti-spoofing performance and writes calibrated operating thresholds:
```bash
python ml/evaluate.py
```

### 3. Model Training
```bash
python ml/train.py --epochs 5 --batch-size 16
```

## Evaluated Metrics

- **minDCF**: Minimum normalized detection cost function under ASVspoof 5 parameters ($C_{\text{miss}}=1, C_{\text{fa}}=1, P_{\text{target}}=0.05$).
- **actDCF**: Actual detection cost at theoretical Bayes threshold $\tau_{\text{Bayes}} = -\ln \beta \approx -0.644$.
- **CLLR**: Log-likelihood ratio cost measuring probabilistic calibration.
- **EER**: Equal Error Rate operating point ($P_{\text{miss}} = P_{\text{fa}}$).
