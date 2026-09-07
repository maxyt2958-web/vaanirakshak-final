"""ASVspoof 5 metric harness: minDCF, actDCF, CLLR, and EER.

Follows official ASVspoof 5 conventions:
- EER: threshold where FAR (False Acceptance Rate) = FRR (False Rejection Rate)
- minDCF: minimum normalized detection cost over all candidate thresholds
  (Cmiss = 1.0, Cfa = 1.0, Ptar = 0.05)
- actDCF: actual detection cost at a fixed threshold (default 0.5)
- CLLR: log-likelihood ratio cost (BOSARIS / FoCal definition)

Uses numpy only with no torch dependency.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def compute_eer(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the operating threshold where FAR = FRR.

    In countermeasure scoring:
    - Bonafide scores are expected to be higher than spoof scores.
    - At threshold tau:
      * Bonafide < tau is a false rejection (FRR / Pmiss)
      * Spoof >= tau is a false alarm (FAR / Pfa)
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if bon.size == 0 or spf.size == 0:
        return float("nan"), float("nan")

    bon_sorted = np.sort(bon)
    spf_sorted = np.sort(spf)
    n_bon = float(bon_sorted.size)
    n_spf = float(spf_sorted.size)

    # Candidate thresholds spanning all scores and midpoints
    all_scores = np.sort(np.unique(np.concatenate([bon_sorted, spf_sorted])))
    if all_scores.size == 1:
        return 0.5, float(all_scores[0])

    midpoints = (all_scores[:-1] + all_scores[1:]) / 2.0
    thresholds = np.concatenate([
        [all_scores[0] - 1.0],
        all_scores,
        midpoints,
        [all_scores[-1] + 1.0],
    ])
    thresholds = np.sort(np.unique(thresholds))

    # Pmiss (FRR): fraction of bonafide < tau
    p_miss = np.searchsorted(bon_sorted, thresholds, side="left") / n_bon
    # Pfa (FAR): fraction of spoof >= tau
    p_fa = (n_spf - np.searchsorted(spf_sorted, thresholds, side="left")) / n_spf

    # Find the crossing point where FAR <= FRR (p_fa <= p_miss)
    diff = p_fa - p_miss
    idx = np.where(diff <= 0)[0]

    if idx.size == 0:
        return float(p_fa[-1]), float(thresholds[-1])
    first_idx = int(idx[0])
    if first_idx == 0:
        return float(p_miss[0]), float(thresholds[0])

    # Linear interpolation between first_idx - 1 and first_idx
    x0, y0 = p_fa[first_idx - 1], p_miss[first_idx - 1]
    x1, y1 = p_fa[first_idx], p_miss[first_idx]
    t0, t1 = thresholds[first_idx - 1], thresholds[first_idx]

    denom = (x0 - x1) + (y1 - y0)
    if denom > 0:
        alpha = (x0 - y0) / denom
        eer = float(x0 + alpha * (x1 - x0))
        threshold_eer = float(t0 + alpha * (t1 - t0))
    else:
        eer = float(0.5 * (x0 + y0))
        threshold_eer = float(0.5 * (t0 + t1))

    return float(np.clip(eer, 0.0, 1.0)), threshold_eer


def compute_mindcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_tar: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> Tuple[float, float]:
    """Compute minimum normalized detection cost function (minDCF).

    ASVspoof 5 default parameters:
    Cmiss = 1.0, Cfa = 1.0, Ptar = 0.05.
    Normalized DCF = Cdet(tau) / min(Cmiss * Ptar, Cfa * (1 - Ptar)).
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if bon.size == 0 or spf.size == 0:
        return float("nan"), float("nan")

    bon_sorted = np.sort(bon)
    spf_sorted = np.sort(spf)
    n_bon = float(bon_sorted.size)
    n_spf = float(spf_sorted.size)

    all_scores = np.sort(np.unique(np.concatenate([bon_sorted, spf_sorted])))
    midpoints = (all_scores[:-1] + all_scores[1:]) / 2.0 if all_scores.size > 1 else np.array([])
    thresholds = np.concatenate([
        [all_scores[0] - 1.0],
        all_scores,
        midpoints,
        [all_scores[-1] + 1.0],
    ])
    thresholds = np.sort(np.unique(thresholds))

    p_miss = np.searchsorted(bon_sorted, thresholds, side="left") / n_bon
    p_fa = (n_spf - np.searchsorted(spf_sorted, thresholds, side="left")) / n_spf

    c_det = c_miss * p_tar * p_miss + c_fa * (1.0 - p_tar) * p_fa
    c_default = min(c_miss * p_tar, c_fa * (1.0 - p_tar))
    dcf_norm = c_det / c_default

    min_idx = int(np.argmin(dcf_norm))
    min_dcf = float(dcf_norm[min_idx])
    min_dcf = float(np.clip(min_dcf, 0.0, 1.0))
    best_thresh = float(thresholds[min_idx])

    return min_dcf, best_thresh


def compute_act_dcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    threshold: float = 0.5,
    p_tar: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute actual normalized detection cost function (actDCF) at a fixed threshold."""
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if bon.size == 0 or spf.size == 0:
        return float("nan")

    p_miss = float(np.mean(bon < threshold))
    p_fa = float(np.mean(spf >= threshold))

    c_det = c_miss * p_tar * p_miss + c_fa * (1.0 - p_tar) * p_fa
    c_default = min(c_miss * p_tar, c_fa * (1.0 - p_tar))
    act_dcf = c_det / c_default

    return float(act_dcf)


def compute_cllr(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
) -> float:
    """Compute Cost of Log-Likelihood Ratios (CLLR).

    BOSARIS/FoCal definition:
    CLLR = 1 / (2 * ln 2) * [ (1/N_tar) * sum(ln(1 + e^(-s_bon))) + (1/N_non) * sum(ln(1 + e^(s_spf))) ]
    """
    bon = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    spf = np.asarray(spoof_scores, dtype=np.float64).ravel()

    if bon.size == 0 or spf.size == 0:
        return float("nan")

    tar_cost = float(np.mean(np.logaddexp(0.0, -bon)) / np.log(2.0))
    non_cost = float(np.mean(np.logaddexp(0.0, spf)) / np.log(2.0))
    cllr = 0.5 * (tar_cost + non_cost)

    return float(cllr)


def evaluate_system(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    threshold: float = 0.5,
    p_tar: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate system scores and return ASVspoof 5 metrics.

    Parameters:
        bonafide_scores: Array of countermeasure scores for bonafide audio.
        spoof_scores: Array of countermeasure scores for spoof audio.
        threshold: Operating threshold for actDCF (default 0.5).
        p_tar: Prior probability of target class (default 0.05).
        c_miss: Cost of missed target detection (default 1.0).
        c_fa: Cost of false alarm (default 1.0).

    Returns:
        Dict with keys:
        - min_dcf: Minimum normalized detection cost
        - act_dcf: Actual detection cost at fixed threshold
        - cllr: Log-likelihood ratio cost
        - eer: Equal error rate
        (Also includes uppercase/camelCase aliases minDCF, actDCF, CLLR, EER)
    """
    eer, eer_thresh = compute_eer(bonafide_scores, spoof_scores)
    min_dcf, mindcf_thresh = compute_mindcf(
        bonafide_scores, spoof_scores, p_tar=p_tar, c_miss=c_miss, c_fa=c_fa
    )
    act_dcf = compute_act_dcf(
        bonafide_scores, spoof_scores, threshold=threshold, p_tar=p_tar, c_miss=c_miss, c_fa=c_fa
    )
    cllr = compute_cllr(bonafide_scores, spoof_scores)

    return {
        "min_dcf": min_dcf,
        "act_dcf": act_dcf,
        "cllr": cllr,
        "eer": eer,
        "minDCF": min_dcf,
        "actDCF": act_dcf,
        "CLLR": cllr,
        "EER": eer,
        "threshold_eer": eer_thresh,
        "threshold_mindcf": mindcf_thresh,
        "threshold_actdcf": threshold,
    }


def print_metrics(results: Dict[str, Any]) -> None:
    """Print formatted evaluation metrics table."""
    eer = results.get("eer", results.get("EER", float("nan")))
    min_dcf = results.get("min_dcf", results.get("minDCF", float("nan")))
    act_dcf = results.get("act_dcf", results.get("actDCF", float("nan")))
    cllr = results.get("cllr", results.get("CLLR", float("nan")))

    header_line = "=" * 58
    print(f"\n{header_line}")
    print("      ASVspoof 5 Benchmark Metrics Evaluation")
    print(header_line)
    print(f"  {'Metric':<36} | {'Value':>15}")
    print(f"  {'-'*36} | {'-'*15}")
    print(f"  {'Equal Error Rate (EER)':<36} | {eer * 100.0:>14.2f}%")
    print(f"  {'Minimum Detection Cost (minDCF)':<36} | {min_dcf:>15.4f}")
    print(f"  {'Actual Detection Cost (actDCF)':<36} | {act_dcf:>15.4f}")
    print(f"  {'Log-Likelihood Ratio Cost (CLLR)':<36} | {cllr:>15.4f}")
    print(f"{header_line}\n")
