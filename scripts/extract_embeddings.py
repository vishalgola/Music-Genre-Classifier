"""
extract_embeddings.py - Stage 2b: Extract YAMNet embeddings from audio files.

Run:
    python scripts/extract_embeddings.py

Creates one embedding file per track in data/embeddings_yamnet/{track_id}.npy
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    EMBED_DIR,
    FMA_AUDIO_DIR,
    LOG_DIR,
    SAMPLE_RATE,
    SPLITS_FILE,
    YAMNET_HANDLE,
    YAMNET_SAMPLE_RATE,
    YAMNET_SEG_SECONDS,
    YAMNET_SEG_HOP,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def track_id_to_path(track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return FMA_AUDIO_DIR / tid[:3] / f"{tid}.mp3"


def _load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=YAMNET_SAMPLE_RATE, mono=True, duration=30.0)
    return y.astype(np.float32)


def _segment_audio(y: np.ndarray) -> list[np.ndarray]:
    seg_len = int(YAMNET_SEG_SECONDS * YAMNET_SAMPLE_RATE)
    hop = int(YAMNET_SEG_HOP * YAMNET_SAMPLE_RATE)
    if seg_len <= 0:
        return [y]
    segments = []
    for start in range(0, max(1, len(y) - seg_len + 1), hop):
        seg = y[start : start + seg_len]
        if len(seg) < seg_len:
            break
        segments.append(seg)
    if not segments:
        segments = [y[:seg_len]]
    return segments


def extract_all() -> None:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found. Run extract_features.py first.")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLITS_FILE.read_text())
    all_ids = list(splits["train"].keys()) + list(splits["val"].keys()) + list(splits["test"].keys())
    all_ids = sorted(set(int(x) for x in all_ids))

    log.info(f"Loading YAMNet from {YAMNET_HANDLE}")
    yamnet = hub.load(YAMNET_HANDLE)

    missing = 0
    processed = 0
    for tid in tqdm(all_ids, desc="Embeddings"):
        out_path = EMBED_DIR / f"{tid}.npy"
        if out_path.exists():
            continue
        audio_path = track_id_to_path(tid)
        if not audio_path.exists():
            missing += 1
            continue
        try:
            waveform = _load_audio(audio_path)
            segments = _segment_audio(waveform)
            seg_embs = []
            for seg in segments:
                scores, embeddings, spectrogram = yamnet(seg)
                emb = tf.reduce_mean(embeddings, axis=0)
                seg_embs.append(emb)
            emb = tf.reduce_mean(tf.stack(seg_embs, axis=0), axis=0).numpy().astype(np.float32)
            np.save(out_path, emb)
            processed += 1
        except Exception as exc:
            log.warning(f"Failed {tid}: {exc}")

    log.info(f"Embeddings done. new={processed}, missing_audio={missing}")


if __name__ == "__main__":
    extract_all()
