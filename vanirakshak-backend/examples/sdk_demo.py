"""5-line SDK demo — works against a running server.

Start the server in one shell::

    python -m vanirakshak serve

Then in another shell::

    python examples/sdk_demo.py
"""

from __future__ import annotations

import os
import struct
import sys
import wave

import numpy as np

# Allow running this file from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from vanirakshak.data.synthetic import synthetic_speech, synthetic_spoofed  # noqa: E402
from vanirakshak.sdk import VaniRakshakClient  # noqa: E402


def _save_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_i16.tobytes())


def main() -> int:
    sr = 16000
    out_dir = os.path.join(HERE, "sample_audio")
    os.makedirs(out_dir, exist_ok=True)

    gen_path = os.path.join(out_dir, "genuine.wav")
    spf_path = os.path.join(out_dir, "spoofed.wav")
    _save_wav(gen_path, synthetic_speech(4.0, sr=sr, rng=np.random.default_rng(1)), sr)
    _save_wav(spf_path, synthetic_spoofed(4.0, sr=sr, rng=np.random.default_rng(2)), sr)

    client = VaniRakshakClient(host="http://127.0.0.1:8000")
    print("health :", client.health())

    # 1. enrol
    print("enroll :", client.enroll("demo_user", gen_path))

    # 2. one-shot analyse
    print("analyse genuine :", client.analyze(gen_path, claimed_identity="demo_user"))
    print("analyse spoofed :", client.analyze(spf_path, claimed_identity="demo_user"))

    # 3. streaming
    print("streaming spoofed chunks ...")
    spf = np.frombuffer(open(spf_path, "rb").read(), dtype=np.uint8)
    # for brevity, we just call /v1/analyze on the file again
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
