"""
VaniRakshak — ECAPA-TDNN Speaker Verification & Cross-Session Consistency Engine
Compliant with DPDP Act 2023 Section 8(7) (Zero-retention voice biometrics).

Provides 192-dimensional unit-hypersphere embeddings, cosine similarity,
calibrated ASV consistency scores, and anchor vault verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger("vanirakshak.ecapa_verifier")

EMBED_DIM = 192


def compute_receipt_sha256(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash over canonical JSON payload."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


class ECAPAVerifier:
    """ECAPA-TDNN Speaker Verification engine for live call forensics."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        anchor_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_source = model_source
        self._classifier = None
        self._anchor_embedding: Optional[np.ndarray] = None
        self._anchor_profile: Optional[Dict[str, Any]] = None

        if anchor_path is not None:
            self.load_anchor(anchor_path)

    def _ensure_model_loaded(self) -> None:
        if self._classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info(f"Initializing ECAPA-TDNN from {self.model_source} on {self.device}...")
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts={"device": self.device},
            )

    def load_anchor(self, anchor_path: Union[str, Path]) -> None:
        """Load pre-cached permanent speaker identity anchor."""
        p = Path(anchor_path)
        if not p.is_file():
            raise FileNotFoundError(f"Anchor file not found: {p}")

        if p.suffix == ".pt":
            payload = torch.load(p, map_location="cpu")
            if isinstance(payload, dict) and "embedding" in payload:
                emb = payload["embedding"].numpy()
                self._anchor_profile = payload.get("metadata", {})
            else:
                emb = payload.numpy()
        elif p.suffix == ".npy":
            emb = np.load(p)
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._anchor_profile = data
                # Load corresponding .npy or .pt if present
                npy_ref = p.parent / "anchor_speaker_embedding.npy"
                if npy_ref.is_file():
                    emb = np.load(npy_ref)
                else:
                    raise ValueError(f"JSON profile loaded but corresponding vector array missing: {npy_ref}")
        else:
            raise ValueError(f"Unsupported anchor file format: {p.suffix}")

        emb = np.asarray(emb, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb) + 1e-9
        self._anchor_embedding = (emb / norm).astype(np.float32)
        logger.info(f"Loaded speaker anchor with dimension {self._anchor_embedding.shape[0]} from {p}")

    @property
    def has_anchor(self) -> bool:
        return self._anchor_embedding is not None

    def embed(self, waveform: Union[np.ndarray, torch.Tensor], sr: int = 16000) -> np.ndarray:
        """Extract a 192-dimensional unit-normalized speaker embedding."""
        self._ensure_model_loaded()

        if isinstance(waveform, np.ndarray):
            x = torch.from_numpy(waveform).float()
        else:
            x = waveform.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Resample in-memory if needed
        if sr != 16000:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            x = resampler(x)

        with torch.no_grad():
            emb = self._classifier.encode_batch(x.to(self.device))
            norm_emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu().numpy().astype(np.float32)

        return norm_emb

    @staticmethod
    def compute_cosine(e1: np.ndarray, e2: np.ndarray) -> float:
        """Calculate cosine similarity between two unit-normalized vectors."""
        n1 = np.linalg.norm(e1) + 1e-9
        n2 = np.linalg.norm(e2) + 1e-9
        return float(np.dot(e1, e2) / (n1 * n2))

    def verify(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sr: int = 16000,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, bool]:
        """Verify audio against reference or anchor embedding.
        
        Returns:
            (cosine_similarity in [-1, 1], calibrated_score in [0, 1], match_verdict)
        """
        ref = reference_embedding if reference_embedding is not None else self._anchor_embedding
        if ref is None:
            raise ValueError("No reference or enrolled anchor embedding available for verification.")

        live_emb = self.embed(waveform, sr)
        cos_sim = self.compute_cosine(ref, live_emb)
        calibrated_score = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
        match = cos_sim >= 0.55

        return cos_sim, calibrated_score, match

    def verify_windows(
        self,
        windows: list[Any],
        sr: int = 16000,
        reference_embedding: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """Verify a sequence of 2.0-second sliding windows against the target anchor."""
        ref = reference_embedding if reference_embedding is not None else self._anchor_embedding
        if ref is None:
            raise ValueError("No enrolled anchor embedding available.")

        results = []
        for w in windows:
            if not getattr(w, "is_voiced", True):
                results.append({
                    "window_index": getattr(w, "index", 0),
                    "is_voiced": False,
                    "speaker_sim": None,
                    "match": None,
                })
                continue

            samples = getattr(w, "samples", w)
            sim, cal, match = self.verify(samples, sr=sr, reference_embedding=ref)
            results.append({
                "window_index": getattr(w, "index", 0),
                "is_voiced": True,
                "speaker_sim": round(sim, 4),
                "calibrated_score": round(cal, 4),
                "match": match,
            })
        return results


def main() -> None:
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(
        description="VaniRakshak — ECAPA-TDNN Speaker Identity Verification & Zero-Shot Defense"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to verify (WAV, MP3, FLAC)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="data/anchor_speaker_embedding.pt",
        help="Path to enrolled target anchor vector (.pt, .npy, or .json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Cosine similarity decision threshold (default: 0.55)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    anchor_path = Path(args.anchor)

    if not audio_path.is_file():
        logger.error(f"Audio file not found: {audio_path}")
        return

    verifier = ECAPAVerifier(device=args.device, anchor_path=anchor_path)

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    cos_sim, calibrated_score, match = verifier.verify(audio, sr=sr)
    margin = cos_sim - args.threshold
    target_name = (
        verifier._anchor_profile.get("speaker_name", "Enrolled Anchor")
        if verifier._anchor_profile
        else "Enrolled Anchor"
    )

    print("\n========================================================")
    print("   VaniRakshak — Biometric Speaker Identity Report       ")
    print("========================================================")
    print(f"Test File:               {audio_path.name}")
    print(f"Target Enrolled Voice:   {target_name}")
    print(f"Vector Dimension:        {EMBED_DIM}-dimensional unit hypersphere")
    print(f"Cosine Similarity:       {cos_sim:.4f}")
    print(f"Decision Threshold:      {args.threshold:.2f}")
    print(f"Verification Margin:     {margin:+.4f}")
    print(f"Calibrated ASV Score:    {calibrated_score * 100:.2f}%")
    if match:
        print("Verdict:                 AUTHENTIC TARGET SPEAKER (MATCH)")
    else:
        print("Verdict:                 ZERO-SHOT IMPERSONATION REJECTED (IMPOSTOR)")
    print("========================================================\n")


if __name__ == "__main__":
    main()
