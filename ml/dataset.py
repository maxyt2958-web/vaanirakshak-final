"""VaniRakshak — Forensic Audio Dataset & Augmentation Pipeline.

Provides:
  - ForensicAudioDataset: PyTorch dataset supporting both file-based and in-memory
    synthetic audio streams with telephony codec augmentation (A-law, mu-law, AMR-WB, GSM-EFR).
  - Collate functions for variable-length window batching.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from vanirakshak.data.augment import apply_random_pipeline
from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed


class ForensicAudioDataset(Dataset):
    """Forensic anti-spoofing dataset with telephony channel augmentation."""

    def __init__(
        self,
        samples: Optional[List[Dict[str, Union[str, int, float]]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        synthetic_count: Optional[int] = None,
        sample_rate: int = 16000,
        window_sec: float = 4.04,
        augment: bool = True,
        feature_extractor: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_sec * sample_rate))
        self.augment = augment
        self.feature_extractor = feature_extractor
        self.items: List[Dict[str, Union[str, int, float]]] = []

        # 1. Synthetic generation mode
        if synthetic_count is not None and synthetic_count > 0:
            self.mode = "synthetic"
            self.synthetic_count = synthetic_count
            for i in range(synthetic_count):
                label = 1 if (i % 2 == 0) else 0  # balanced bonafide / spoof
                self.items.append({
                    "id": f"synth_{i:06d}",
                    "label": label,
                    "type": "bonafide" if label == 1 else "spoof",
                })
        # 2. Manifest-based mode
        elif manifest_path is not None:
            self.mode = "file"
            p = Path(manifest_path)
            if not p.is_file():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "files" in data:
                # Format of ground_truth.json
                base_dir = p.parent
                for fname, meta in data["files"].items():
                    fpath = base_dir / fname
                    if fpath.is_file():
                        label = 1 if meta.get("label") == "genuine" else 0
                        self.items.append({
                            "id": fname,
                            "path": str(fpath),
                            "label": label,
                            "expected_verdict": meta.get("expected_verdict", ""),
                        })
            elif isinstance(data, list):
                self.items = data
        # 3. Explicit samples list
        elif samples is not None:
            self.mode = "file"
            self.items = samples
        else:
            raise ValueError("Must provide samples, manifest_path, or synthetic_count.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_or_generate_audio(self, item: Dict[str, Union[str, int, float]]) -> Tuple[np.ndarray, int]:
        label = int(item["label"])

        if self.mode == "synthetic":
            duration_sec = float(self.window_samples) / self.sample_rate
            if label == 1:
                pcm = synthetic_speech(duration_sec=duration_sec, sr=self.sample_rate)
            else:
                pcm = synthetic_spoofed(duration_sec=duration_sec, sr=self.sample_rate)
            # Normalize float32 in [-1, 1]
            x = pcm.astype(np.float32) / 32768.0
        else:
            # File loading
            path = Path(str(item["path"]))
            import wave
            with wave.open(str(path), "rb") as w:
                raw = w.readframes(w.getnframes())
                sr = w.getframerate()
                sw = w.getsampwidth()
            if sw == 2:
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                arr = np.frombuffer(raw, dtype=np.float32)

            # Center crop or pad to window_samples
            if len(arr) > self.window_samples:
                start = random.randint(0, len(arr) - self.window_samples)
                x = arr[start : start + self.window_samples]
            elif len(arr) < self.window_samples:
                pad = self.window_samples - len(arr)
                x = np.pad(arr, (0, pad), mode="constant")
            else:
                x = arr

        # Apply telephony codec augmentation
        if self.augment:
            x = apply_random_pipeline(x, self.sample_rate)

        return x, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        x, label = self._load_or_generate_audio(item)

        # Optional feature extraction
        if self.feature_extractor is not None:
            feats = self.feature_extractor(x, self.sample_rate)
            tensor_x = torch.from_numpy(feats).float()
        else:
            # Default: 768-dim pseudo acoustic feature projection from spectral representations
            # (FFT energy profile interpolated to 768 dimensions)
            n_fft = 512
            hop = 256
            window = np.hanning(n_fft)
            num_frames = (len(x) - n_fft) // hop + 1
            if num_frames > 0:
                frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
                spec = np.abs(np.fft.rfft(frames * window, axis=1))  # (T, 257)
                mean_spec = np.mean(spec, axis=0)  # (257,)
                # Interpolate to 768
                vec_768 = np.interp(np.linspace(0, 1, 768), np.linspace(0, 1, len(mean_spec)), mean_spec)
                vec_768 = (vec_768 - np.mean(vec_768)) / (np.std(vec_768) + 1e-8)
            else:
                vec_768 = np.zeros(768, dtype=np.float32)

            tensor_x = torch.from_numpy(vec_768).float()

        tensor_y = torch.tensor(label, dtype=torch.long)
        return tensor_x, tensor_y
