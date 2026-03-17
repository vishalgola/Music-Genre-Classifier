"""
evaluate_tar.py - Stream evaluation directly from Jamendo TARs using a trained YAMNet classifier.

Run:
    python scripts/evaluate_tar.py --metadata /path/to/metadata.tsv --tars-dir /path/to/tars
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    MODEL_PATH_YAMNET,
    SPLITS_FILE,
    YAMNET_HANDLE,
    YAMNET_SAMPLE_RATE,
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


def build_tar_index(tars_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    tar_files = list(tars_dir.glob("*.tar")) + list(tars_dir.glob("*.tar.gz")) + list(tars_dir.glob("*.tgz"))
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    index[member.name] = tar_path
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Jamendo metadata TSV/CSV.")
    parser.add_argument("--tars-dir", required=True, help="Directory containing tar/tar.gz files.")
    args = parser.parse_args()

    if not MODEL_PATH_YAMNET.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH_YAMNET}. Train yamnet first.")
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found. Build splits first.")

    splits = json.loads(Path(SPLITS_FILE).read_text())
    test_map = splits["test"]

    df = load_metadata(Path(args.metadata))
    ids = normalize_ids(df)
    paths = normalize_paths(df)
    id_to_path = dict(zip(ids, paths))

    log.info(f"Indexing tar files in {args.tars_dir}...")
    index = build_tar_index(Path(args.tars_dir))

    yamnet = hub.load(YAMNET_HANDLE)
    model = tf.keras.models.load_model(MODEL_PATH_YAMNET)

    y_true = []
    y_pred = []
    missing = 0

    for tid, label in tqdm(test_map.items(), desc="Eval"):
        rel_path = id_to_path.get(str(tid))
        if not rel_path:
            missing += 1
            continue
        tar_path = index.get(rel_path)
        if not tar_path:
            missing += 1
            continue
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                member = tar.getmember(rel_path)
                f = tar.extractfile(member)
                if f is None:
                    missing += 1
                    continue
                raw = f.read()
            waveform = _load_audio_bytes(raw)
            segments = _segment_audio(waveform)
            seg_embs = []
            for seg in segments:
                scores, embeddings, spectrogram = yamnet(seg)
                seg_embs.append(tf.reduce_mean(embeddings, axis=0))
            emb = tf.reduce_mean(tf.stack(seg_embs, axis=0), axis=0).numpy().astype(np.float32)
            probs = model.predict(emb[None, :], verbose=0)[0]
            y_true.append(label)
            y_pred.append(int(np.argmax(probs)))
        except Exception:
            missing += 1
            continue

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    log.info(f"Eval done. acc={acc:.4f} f1={f1:.4f} missing={missing}")


if __name__ == "__main__":
    main()
