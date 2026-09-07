"""VaniRakshak — ASVspoof-5 Loss Functions & Objective Formulations.

Provides:
  1. AdditiveMarginSoftmaxLoss (AM-Softmax): Enforces an angular separation margin
     between bonafide acoustic vectors and synthetic voice clone vectors.
  2. ForensicDetectionCostLoss: Directly optimizes a smooth differentiable surrogate
     of the ASVspoof-5 Normalized Minimum Detection Cost Function (minDCF).
  3. CalibratedCostWeightedBCELoss: Cost-weighted binary cross entropy with prior adjustment.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveMarginSoftmaxLoss(nn.Module):
    """Additive Margin Softmax (AM-Softmax) for acoustic anti-spoofing.

    Penalizes the cosine angle between feature representations and class weights:
        cos(theta) - margin for the true class.
    Forces bonafide embeddings to form a compact cluster distinct from all spoof styles.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int = 2,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin

        # Normalized class weight vectors
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch_size, in_features) feature embeddings.
            labels: (batch_size,) integer class targets (0=spoof, 1=bonafide).

        Returns:
            Scalar AM-Softmax loss.
        """
        # Normalize weights and feature representations to unit hypersphere
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)

        # Cosine similarity: (batch_size, n_classes)
        cosine = F.linear(x_norm, w_norm)

        # One-hot mask for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply additive margin penalty to ground truth class angle
        output = self.scale * (cosine - one_hot * self.margin)
        return F.cross_entropy(output, labels)


class ForensicDetectionCostLoss(nn.Module):
    """Differentiable surrogate of the ASVspoof 5 Detection Cost Function (DCF).

    Approximates P_miss(tau) and P_fa(tau) using sigmoid relaxation:
        DCF = C_miss * P_tar * P_miss + C_fa * P_non * P_fa
    """

    def __init__(
        self,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
        p_target: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.c_miss = c_miss
        self.c_fa = c_fa
        self.p_target = p_target
        self.p_non_target = 1.0 - p_target
        self.temperature = temperature
        self.beta = (c_fa * self.p_non_target) / (c_miss * p_target)
        self.cost_default = min(c_miss * p_target, c_fa * self.p_non_target)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (batch_size,) or (batch_size, 1) raw log-likelihood scores (positive=bonafide).
            labels: (batch_size,) binary target (0=spoof, 1=bonafide).

        Returns:
            Normalized DCF surrogate loss.
        """
        scores = logits.view(-1)
        targets = labels.view(-1).float()

        bonafide_mask = targets == 1.0
        spoof_mask = targets == 0.0

        n_bon = torch.sum(bonafide_mask).clamp(min=1.0)
        n_spf = torch.sum(spoof_mask).clamp(min=1.0)

        # Theoretical Bayes decision threshold: -ln(beta)
        tau_bayes = -math.log(self.beta)

        # Smooth surrogate indicator functions: sigmoid((tau - score) / temp)
        # For bonafide: miss occurs when score < tau -> sigmoid((tau - score) / temp)
        # For spoof: false alarm occurs when score > tau -> sigmoid((score - tau) / temp)
        diff = (scores - tau_bayes) / self.temperature

        # P_miss approximation over bonafide
        p_miss = torch.sum(torch.sigmoid(-diff) * bonafide_mask) / n_bon

        # P_fa approximation over spoof
        p_fa = torch.sum(torch.sigmoid(diff) * spoof_mask) / n_spf

        # Raw DCF
        dcf = self.c_miss * self.p_target * p_miss + self.c_fa * self.p_non_target * p_fa

        # Normalize by default naive cost
        norm_dcf = dcf / self.cost_default
        return norm_dcf


class CalibratedCostWeightedBCELoss(nn.Module):
    """Prior-adjusted cost-weighted Binary Cross-Entropy."""

    def __init__(self, pos_weight: float = 1.905) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), labels.view(-1).float())
