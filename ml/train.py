"""VaniRakshak — Acoustic Countermeasure Training Pipeline.

Trains the AcousticForensicClassifier using cost-weighted detection loss and AM-Softmax,
computes ASVspoof-5 validation metrics across epochs, and saves the calibrated model weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compute_mindcf import evaluate_system
from ml.dataset import ForensicAudioDataset
from ml.loss import AdditiveMarginSoftmaxLoss, ForensicDetectionCostLoss
from ml.model import AcousticForensicClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vanirakshak.train")


def train_epoch(
    model: nn.Module,
    criterion_dcf: nn.Module,
    criterion_am: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        llr, embed = model(x, return_embedding=True)

        loss_dcf = criterion_dcf(llr, y)
        loss_am = criterion_am(embed, y)
        loss = 0.7 * loss_dcf + 0.3 * loss_am

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    bonafide_scores = []
    spoof_scores = []

    for x, y in loader:
        x = x.to(device)
        llr = model(x).view(-1).cpu().numpy()
        labels = y.numpy()

        for score, label in zip(llr, labels):
            if label == 1:
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)

    bon = np.array(bonafide_scores)
    spf = np.array(spoof_scores)
    metrics = evaluate_system(bon, spf)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train VaniRakshak Acoustic Countermeasure Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--synthetic-samples", type=int, default=160, help="Training synthetic samples")
    parser.add_argument("--out-dir", type=str, default="ml/checkpoints", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    train_dataset = ForensicAudioDataset(synthetic_count=args.synthetic_samples, augment=True)
    val_dataset = ForensicAudioDataset(synthetic_count=max(40, args.synthetic_samples // 4), augment=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model & Criteria
    model = AcousticForensicClassifier(input_dim=768, proj_dim=128, hidden_dim=64, embedding_dim=64).to(device)
    criterion_dcf = ForensicDetectionCostLoss()
    criterion_am = AdditiveMarginSoftmaxLoss(in_features=64, n_classes=2, margin=0.35).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mindcf = float("inf")
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, criterion_dcf, criterion_am, train_loader, optimizer, device)
        val_metrics = evaluate_metrics(model, val_loader, device)
        elapsed = time.time() - t0

        mindcf = val_metrics["minDCF"]
        eer = val_metrics["EER"]
        logger.info(
            f"Epoch {epoch}/{args.epochs} [{elapsed:.1f}s] - Loss: {loss:.4f} | "
            f"minDCF: {mindcf:.4f} | EER: {eer:.2f}% | CLLR: {val_metrics['CLLR']:.4f}"
        )

        if mindcf < best_mindcf:
            best_mindcf = mindcf
            best_metrics = val_metrics
            ckpt_path = out_dir / "countermeasure_cm.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved new best model checkpoint to {ckpt_path}")

    # Save calibrated thresholds
    thresholds = {
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_min_dcf": best_mindcf,
        "validation_eer_pct": best_metrics.get("EER", 0.0),
        "validation_act_dcf": best_metrics.get("actDCF", 0.0),
        "validation_cllr": best_metrics.get("CLLR", 0.0),
        "threshold_allow_max": 35.0,
        "threshold_challenge_max": 69.9,
        "threshold_block_min": 70.0,
        "bayes_operating_threshold": -0.644,
    }

    thresh_path = out_dir / "thresholds.json"
    thresh_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    logger.info(f"Calibrated thresholds saved to {thresh_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
