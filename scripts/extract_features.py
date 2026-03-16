"""
extract_features.py — Stage 2: Convert FMA Medium MP3s to Mel Spectrogram tensors.

Run independently:
    python scripts/extract_features.py

Idempotent: skips tracks whose .npy file already exists.
Corrupted/unreadable tracks are logged to data/skip_list.txt and skipped.
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    DURATION,
    EXTRACT_LOG,
    FMA_META_DIR,
    HOP_LENGTH,
    IMG_SIZE,
    LOG_DIR,
    N_FFT,
    N_MELS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    SKIP_LIST_FILE,
    SPLITS_FILE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    FMA_AUDIO_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EXTRACT_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def track_id_to_path(track_id: int) -> Path:
    """Convert integer track_id to the FMA directory path.

    FMA stores files like: fma_medium/000/000002.mp3
    """
    tid = f"{track_id:06d}"
    return FMA_AUDIO_DIR / tid[:3] / f"{tid}.mp3"


def audio_to_melspec(audio_path: Path) -> np.ndarray:
    """Load audio, compute Mel Spectrogram, and resize to (IMG_SIZE, IMG_SIZE).

    Returns:
        np.ndarray of shape (IMG_SIZE, IMG_SIZE), dtype float32
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # z-score normalise
    mean, std = mel_db.mean(), mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std

    # resize from (128, ~1292) → (IMG_SIZE, IMG_SIZE) for MobileNetV2
    scale_y = IMG_SIZE / mel_db.shape[0]
    scale_x = IMG_SIZE / mel_db.shape[1]
    mel_resized = zoom(mel_db, (scale_y, scale_x)).astype(np.float32)

    return mel_resized


def _process_one(track_id: int) -> tuple[int, bool, str]:
    """Worker function — extract one track. Returns (id, success, msg)."""
    out_path = PROCESSED_DIR / f"{track_id}.npy"
    if out_path.exists():
        return track_id, True, "cached"

    audio_path = track_id_to_path(track_id)
    if not audio_path.exists():
        return track_id, False, f"audio not found: {audio_path}"

    try:
        mel = audio_to_melspec(audio_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mel)
        return track_id, True, "ok"
    except Exception as exc:
        return track_id, False, str(exc)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_track_metadata() -> pd.DataFrame:
    """Load FMA tracks.csv and return rows with a valid top-level genre."""
    tracks_path = FMA_META_DIR / "tracks.csv"
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # Keep only tracks that belong to the medium subset
    subset = tracks["set", "subset"] == "medium"
    has_genre = tracks["track", "genre_top"].notna()
    df = tracks[subset & has_genre].copy()

    log.info(f"Tracks with valid genre in medium subset: {len(df)}")
    return df


def build_splits(track_ids: list[int], genre_labels: list[str]) -> dict:
    """Create stratified 70/15/15 train/val/test splits."""
    ids_train, ids_tmp, g_train, g_tmp = train_test_split(
        track_ids, genre_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=genre_labels,
        random_state=RANDOM_SEED,
    )
    rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, g_tmp,
        test_size=(1 - rel_val),
        stratify=g_tmp,
        random_state=RANDOM_SEED,
    )

    # Build genre → int label map
    genres = sorted(set(genre_labels))
    label_map = {g: i for i, g in enumerate(genres)}

    splits = {
        "label_map": label_map,
        "train": {str(tid): label_map[g] for tid, g in zip(ids_train, g_train)},
        "val":   {str(tid): label_map[g] for tid, g in zip(ids_val,   g_tmp[:len(ids_val)])},
        "test":  {str(tid): label_map[g] for tid, g in zip(ids_test,  g_tmp[len(ids_val):])},
    }
    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_all(max_workers: int = 4) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_track_metadata()
    track_ids = df.index.tolist()
    genre_labels = df["track", "genre_top"].tolist()

    skip_list: list[str] = []
    successful_ids: list[int] = []
    successful_genres: list[str] = []

    genre_by_id = dict(zip(track_ids, genre_labels))

    log.info(f"Extracting features for {len(track_ids)} tracks using {max_workers} workers…")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, tid): tid for tid in track_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            tid, ok, msg = fut.result()
            if ok:
                successful_ids.append(tid)
                successful_genres.append(genre_by_id[tid])
            else:
                log.warning(f"Skipped track {tid}: {msg}")
                skip_list.append(f"{tid}: {msg}")

    # Write skip list
    SKIP_LIST_FILE.write_text("\n".join(skip_list))
    log.info(f"Skipped {len(skip_list)} tracks → {SKIP_LIST_FILE}")

    # Build and save splits
    if SPLITS_FILE.exists():
        log.info(f"splits.json already exists, skipping split generation.")
    else:
        splits = build_splits(successful_ids, successful_genres)
        SPLITS_FILE.write_text(json.dumps(splits, indent=2))
        log.info(
            f"Splits saved to {SPLITS_FILE} | "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    log.info("Feature extraction complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()
    extract_all(max_workers=args.workers)
