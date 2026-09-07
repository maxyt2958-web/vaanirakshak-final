"""VaniRakshak — ASV & Anti-Spoofing Dataset Split Generator.

Partitions enrolled target audio and recorded speech into standardized
train, validation (dev), and evaluation splits following ASVspoof protocol conventions:
  - 70% Train (model parameter updates)
  - 15% Development / Validation (threshold tuning & hyperparameter selection)
  - 15% Evaluation (blind test reporting: minDCF, actDCF, CLLR, EER)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "splits"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vanirakshak.make_splits")


def create_standard_splits(
    data_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Scan audio directory and generate reproducible train, dev, and eval JSON splits."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(data_dir.rglob("*.wav")))
    if not wav_files:
        logger.warning(f"No WAV files found in {data_dir}. Generating synthetic split protocol.")
        # Create synthetic protocol entries
        records = [
            {"id": f"sample_{i:04d}", "label": "genuine" if i % 2 == 0 else "spoof", "codec": "pcm"}
            for i in range(100)
        ]
    else:
        records = []
        for p in wav_files:
            name = p.stem.lower()
            label = "genuine" if "genuine" in name else "spoof"
            records.append({
                "id": p.name,
                "path": str(p.relative_to(REPO_ROOT)),
                "label": label,
                "is_clone": "clone" in name or "spoof" in name,
                "is_spliced": "splice" in name,
            })

    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_set = records[:n_train]
    dev_set = records[n_train : n_train + n_dev]
    eval_set = records[n_train + n_dev :]

    for split_name, split_data in [("train", train_set), ("dev", dev_set), ("eval", eval_set)]:
        out_file = out_dir / f"{split_name}.json"
        manifest = {
            "split": split_name,
            "count": len(split_data),
            "files": split_data,
        }
        out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Wrote {split_name} split ({len(split_data)} records) to {out_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create reproducible dataset splits")
    parser.add_argument("--data-dir", type=str, default="data/demo", help="Input audio directory")
    parser.add_argument("--out-dir", type=str, default="data/splits", help="Output split directory")
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    out_dir = REPO_ROOT / args.out_dir

    create_standard_splits(data_dir, out_dir)
    return 0


if __name__ == "__main__":
    main()
