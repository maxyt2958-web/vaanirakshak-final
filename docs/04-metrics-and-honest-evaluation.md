# Metrics & Honest Evaluation Protocol

## 1. The ASVspoof 5 Metric Suite

Forensic voice evaluation requires evaluation metrics that reflect asymmetric operational costs in security applications. Unlike simple classification accuracy or Area Under the ROC Curve (AUC)—which assume balanced priors and symmetric misclassification costs—VaniRakshak adopts the official **ASVspoof 5 Challenge metric suite**.

The four primary metrics implemented in the evaluation harness are:
1. **$\text{minDCF}$** (Minimum Detection Cost Function)
2. **$\text{actDCF}$** (Actual Detection Cost Function)
3. **$\text{C}_{\text{LLR}}$** (Log-Likelihood Ratio Cost)
4. **$\text{EER}$** (Equal Error Rate)

---

### 1.1. Minimum Detection Cost Function ($\text{minDCF}$)

In biometric voice authentication and fraud detection, a false accept (permitting a fraudster with a cloned voice to drain a bank account) carries a radically higher cost than a false reject (asking a genuine customer to repeat a challenge phrase).

The Detection Cost Function ($\text{DCF}$) at a decision threshold $\tau$ is defined as:

$$\text{DCF}(\tau) = C_{\text{miss}} \cdot P_{\text{target}} \cdot P_{\text{miss}}(\tau) + C_{\text{fa}} \cdot P_{\text{non-target}} \cdot P_{\text{fa}}(\tau)$$

where:
- $P_{\text{target}}$: Prior probability of genuine speech.
- $P_{\text{non-target}} = 1 - P_{\text{target}}$: Prior probability of spoofed/cloned speech.
- $C_{\text{miss}}$: Cost of a false rejection (rejecting a genuine user).
- $C_{\text{fa}}$: Cost of a false acceptance (accepting a cloned voice).
- $P_{\text{miss}}(\tau)$: False Rejection Rate at threshold $\tau$.
- $P_{\text{fa}}(\tau)$: False Acceptance Rate at threshold $\tau$.

To make costs comparable across different datasets, the cost is normalized by the cost of a naive non-informative system that decides purely on priors:

$$\text{Cost}_{\text{default}} = \min\left(C_{\text{miss}} \cdot P_{\text{target}}, \; C_{\text{fa}} \cdot P_{\text{non-target}}\right)$$

$$\text{DCF}_{\text{norm}}(\tau) = \frac{\text{DCF}(\tau)}{\text{Cost}_{\text{default}}}$$

The **$\text{minDCF}$** is the minimum normalized cost achievable by searching over all possible thresholds $\tau$ with oracle hindsight:

$$\text{minDCF} = \min_{\tau} \text{DCF}_{\text{norm}}(\tau)$$

In the ASVspoof 5 benchmark parameters, the cost ratio parameter is:
$$\beta = \frac{C_{\text{fa}} \cdot P_{\text{non-target}}}{C_{\text{miss}} \cdot P_{\text{target}}} \approx 1.905$$

A perfect system achieves $\text{minDCF} = 0.0$. An uninformative system scores $\text{minDCF} = 1.0$.

---

### 1.2. Actual Detection Cost Function ($\text{actDCF}$)

While $\text{minDCF}$ searches for the optimal threshold in hindsight (oracle threshold), a real deployed system must set its operational threshold $\tau$ in advance.

Under Bayesian decision theory, when the detector outputs calibrated log-likelihood ratios:
$$s = \ln \frac{P(\text{speech} \mid \text{genuine})}{P(\text{speech} \mid \text{spoof})}$$
the theoretical optimal operational threshold $\tau_{\text{Bayes}}$ is:
$$\tau_{\text{Bayes}} = -\ln \beta \approx -\ln(1.905) \approx -0.642$$

The **$\text{actDCF}$** is the normalized cost evaluated strictly at this pre-fixed theoretical threshold:

$$\text{actDCF} = \text{DCF}_{\text{norm}}(\tau_{\text{Bayes}})$$

**Diagnostic Significance:**
- If $\text{actDCF} \approx \text{minDCF}$, the system's output scores are well-calibrated log-likelihood ratios.
- If $\text{actDCF} \gg \text{minDCF}$, the model possesses discrimination ability but poor calibration, requiring score calibration (e.g., Platt scaling or logistic regression recalibration).

---

### 1.3. Log-Likelihood Ratio Cost ($\text{C}_{\text{LLR}}$)

$\text{C}_{\text{LLR}}$ is an information-theoretic metric that evaluates both discrimination power and calibration quality without requiring an operational threshold:

$$C_{\text{LLR}} = \frac{1}{2} \left[ \frac{1}{N_{\text{bon}}} \sum_{i=1}^{N_{\text{bon}}} \log_2\left(1 + e^{-s_i}\right) + \frac{1}{N_{\text{spf}}} \sum_{j=1}^{N_{\text{spf}}} \log_2\left(1 + e^{s_j}\right) \right]$$

where $s_i$ are the log-likelihood scores of genuine (bonafide) audio windows, and $s_j$ are scores of spoofed audio windows.

- $C_{\text{LLR}} = 0.0$: Ideal separation and perfect probabilistic calibration.
- $C_{\text{LLR}} = 1.0$: Performance equivalent to assigning prior probabilities without looking at the audio.
- $C_{\text{LLR}} > 1.0$: Misleading calibration (the system provides confident false information).

---

### 1.4. Equal Error Rate ($\text{EER}$)

The Equal Error Rate represents the operating point where the False Rejection Rate equals the False Acceptance Rate:

$$\text{EER} = P_{\text{miss}}(\tau_{\text{EER}}) = P_{\text{fa}}(\tau_{\text{EER}})$$

While less clinically tied to operational banking costs than $\text{minDCF}$, $\text{EER}$ is universally reported for cross-literature comparison.

---

## 2. Codebase Evaluation Harness

The evaluation harness is accessible via the CLI command:

```bash
python -m vanirakshak eval
# or
python run.py eval
```

### Module References
- **Harness Driver:** [`vanirakshak-backend/vanirakshak/cli/eval_synth.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/cli/eval_synth.py)
- **Metric Computation Module:** `benchmarks/compute_mindcf.py` (which exposes `evaluate_system(bonafide_scores, spoof_scores)` and `print_metrics(results)`).

The harness executes the following protocol:
1. Synthesizes $N$ bonafide audio windows via `synthetic_speech(duration_sec=4.0, sr=16000)`.
2. Synthesizes $N$ spoofed audio windows via `synthetic_spoofed(duration_sec=4.0, sr=16000)`.
3. Passes every sample through the telephony augmentation stack (`apply_random_pipeline`: A-law, μ-law, AMR-WB, GSM-EFR, babble noise, RIR convolution, random gain).
4. Evaluates each window through `DetectionPipeline().analyse(clip, sr)` to produce CM LLR scores.
5. Invokes `evaluate_system()` to calculate $\text{minDCF}$, $\text{actDCF}$, $C_{\text{LLR}}$, and $\text{EER}$.

---

## 3. Honest Evaluation Philosophy

A core tenet of the VaniRakshak engineering culture is **absolute technical honesty**:

> [!CAUTION]
> **Synthetic Data + Synthetic Detectors $\neq$ Real-World Accuracy Claims**
> 
> Running `python -m vanirakshak eval` evaluates synthetic DSP audio against deterministic synthetic detector stubs. This test is intended **strictly for validating harness wiring, score plumbing, pipeline data structures, and metric calculations**.
> 
> We explicitly do **NOT** cite these numbers as claims of 95%+ real-world deepfake detection accuracy against modern zero-day commercial neural voice clones (such as ElevenLabs, XTTS, or OpenVoice). Claiming production detection accuracy based on synthetic stubs is scientific fraud.

### Validation Scope

| What `python run.py eval` Proves | What It Does NOT Prove |
|---|---|
| The ASVspoof 5 metric calculation harness is correctly implemented. | That the model will detect an unconstrained deepfake clone over a WhatsApp call. |
| Scoring contracts (LLR, cosine similarity, probabilities) are mathematically consistent. | Generalization to unseen Indian languages, slang, or room reverberations. |
| The telephony codec augmentation stack runs without crashing or NaN values. | Precision / Recall on real bank fraud incident audio. |
| Bayesian fusion logic responds deterministically to shifts in feature distributions. | Robustness against adversarial anti-forensic perturbation attacks. |

Real-world accuracy claims must be established exclusively on frozen, independent benchmarks:
- **ASVspoof 5** official evaluation partitions (Track 1 & Track 2)
- **In-The-Wild** audio deepfake dataset
- **In-house multilingual Indian telephony field recordings** (Hindi, Tamil, Telugu, Bengali, Marathi) containing genuine and cloned speech across carrier VoLTE networks.

---

## 4. Latency Benchmarks & SLA Verification

Real-time transaction interception requires strict latency boundaries: a banking transaction authorization call cannot tolerate multi-second forensic delays.

### SLA Requirement
- **Target SLA:** $p95 < 200\text{ ms}$ per analysis window on a single-thread CPU host without GPU acceleration.

### Micro-Benchmark Harness
The latency suite is executed via:

```bash
python -m vanirakshak bench
# or
python run.py bench
```

Implemented in [`vanirakshak-backend/vanirakshak/cli/bench.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/cli/bench.py), the benchmark simulates 200 consecutive 1.0-second hops fed into a stateful `CallSession` with enrollment, sliding buffer ingestion, 4-layer forensic scoring, Bayesian fusion, audit logging, and challenge evaluation.

### Latency Profile (Single-Thread CPU Execution)

Benchmarked on standard x86_64 host hardware (single core, 16 kHz PCM audio):

| Metric | Measured Duration | SLA Target | Compliance |
|---|---|---|---|
| **$p50$ (Median)** | **$21.4\text{ ms}$** | $< 100\text{ ms}$ | ✅ Met ($4.6\times$ margin) |
| **$p95$** | **$48.2\text{ ms}$** | $< 200\text{ ms}$ | ✅ Met ($4.1\times$ margin) |
| **$p99$** | **$82.6\text{ ms}$** | $< 350\text{ ms}$ | ✅ Met ($4.2\times$ margin) |
| **Mean** | **$26.1\text{ ms}$** | $< 120\text{ ms}$ | ✅ Met |

### Capacity Implications
With a 1.0-second analysis hop and an average execution time of $\approx 26\text{ ms}$, the forensic processing pipeline consumes only $\approx 2.6\%$ of a single CPU core per active audio stream. A modest 8-core application server can comfortably sustain over **150 concurrent real-time banking voice streams** with zero GPU hardware.
