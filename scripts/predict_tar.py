"""
predict_tar.py - Stream inference directly from Jamendo TARs using a trained YAMNet classifier.

Run:
    python scripts/predict_tar.py --metadata /path/to/metadata.tsv --tars-dir /path/to/tars --track-id 12345
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    MODEL_PATH_YAMNET,
    YAMNET_HANDLE,
    YAMNET_SAMPLE_RATE,
    SPLITS_FILE,
    YAMNET_SEG_SECONDS,
    YAMNET_SEG_HOP,
)
from scripts.jamendo_utils import load_metadata, normalize_ids, normalize_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _load_audio_bytes(raw: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=YAMNET_SAMPLE_RATE, mono=True, duration=30.0)
    Path(tmp_path).unlink(missing_ok=True)
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


def _find_tar(tars_dir: Path, rel_path: str) -> Path | None:
    tar_files = list(tars_dir.glob("*.tar")) + list(tars_dir.glob("*.tar.gz")) + list(tars_dir.glob("*.tgz"))
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r:*") as tar:
            try:
                tar.getmember(rel_path)
                return tar_path
            except KeyError:
                continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Jamendo metadata TSV/CSV.")
    parser.add_argument("--tars-dir", required=True, help="Directory containing tar/tar.gz files.")
    parser.add_argument("--track-id", required=True, help="Track id to predict.")
    args = parser.parse_args()

    if not MODEL_PATH_YAMNET.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH_YAMNET}. Train yamnet first.")

    df = load_metadata(Path(args.metadata))
    ids = normalize_ids(df)
    paths = normalize_paths(df)
    id_to_path = dict(zip(ids, paths))

    rel_path = id_to_path.get(str(args.track_id))
    if not rel_path:
        raise ValueError(f"Track id not found in metadata: {args.track_id}")

    tar_path = _find_tar(Path(args.tars_dir), rel_path)
    if not tar_path:
        raise FileNotFoundError(f"Could not find {rel_path} in tar files.")

    with tarfile.open(tar_path, "r:*") as tar:
        member = tar.getmember(rel_path)
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Could not read {rel_path} from {tar_path}")
        raw = f.read()

    waveform = _load_audio_bytes(raw)
    yamnet = hub.load(YAMNET_HANDLE)
    segments = _segment_audio(waveform)
    seg_embs = []
    for seg in segments:
        scores, embeddings, spectrogram = yamnet(seg)
        seg_embs.append(tf.reduce_mean(embeddings, axis=0))
    emb = tf.reduce_mean(tf.stack(seg_embs, axis=0), axis=0).numpy().astype(np.float32)

    model = tf.keras.models.load_model(MODEL_PATH_YAMNET)
    probs = model.predict(emb[None, :], verbose=0)[0]

    label_map = {}
    if Path(SPLITS_FILE).exists():
        splits = json.loads(Path(SPLITS_FILE).read_text())
        lm: dict[str, int] = splits["label_map"]
        label_map = {v: k for k, v in lm.items()}

    top_idx = int(np.argmax(probs))
    top_label = label_map.get(top_idx, str(top_idx))
    log.info(f"Top label: {top_label} (p={probs[top_idx]:.4f})")


if __name__ == "__main__":
    import json
    main()
