"""
extract_embeddings_tar.py - Extract YAMNet embeddings directly from Jamendo TAR archives.

Run:
    python scripts/extract_embeddings_tar.py --metadata path/to/metadata.tsv --tars-dir path/to/tars
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
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    EMBED_DIR,
    LOG_DIR,
    SPLITS_FILE,
    YAMNET_HANDLE,
    YAMNET_SAMPLE_RATE,
    YAMNET_SEG_SECONDS,
    YAMNET_SEG_HOP,
)
from scripts.jamendo_utils import load_metadata, normalize_ids, normalize_paths

LOG_DIR.mkdir(parents=True, exist_ok=True)
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
        mode = "r:*"
        with tarfile.open(tar_path, mode) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    index[member.name] = tar_path
    return index


def extract_all(metadata_path: Path, tars_dir: Path) -> None:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found. Build splits first.")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLITS_FILE.read_text())
    all_ids = list(splits["train"].keys()) + list(splits["val"].keys()) + list(splits["test"].keys())
    all_ids = sorted(set(str(x) for x in all_ids))

    df = load_metadata(metadata_path)
    ids = normalize_ids(df)
    paths = normalize_paths(df)
    id_to_path = dict(zip(ids, paths))

    log.info(f"Indexing tar files in {tars_dir} (this can take a while)...")
    index = build_tar_index(tars_dir)

    log.info(f"Loading YAMNet from {YAMNET_HANDLE}")
    yamnet = hub.load(YAMNET_HANDLE)

    missing = 0
    processed = 0

    for tid in tqdm(all_ids, desc="Embeddings"):
        out_path = EMBED_DIR / f"{tid}.npy"
        if out_path.exists():
            continue
        rel_path = id_to_path.get(tid)
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
                emb = tf.reduce_mean(embeddings, axis=0)
                seg_embs.append(emb)
            emb = tf.reduce_mean(tf.stack(seg_embs, axis=0), axis=0).numpy().astype(np.float32)
            np.save(out_path, emb)
            processed += 1
        except Exception as exc:
            log.warning(f"Failed {tid}: {exc}")

    log.info(f"Embeddings done. new={processed}, missing={missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Jamendo metadata TSV/CSV.")
    parser.add_argument("--tars-dir", required=True, help="Directory containing tar/tar.gz files.")
    args = parser.parse_args()
    extract_all(Path(args.metadata), Path(args.tars_dir))
