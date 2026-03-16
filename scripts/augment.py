"""
augment.py — Stage 3: Data augmentation on training-set Mel Spectrograms.

Run independently:
    python scripts/augment.py

Generates 2 augmented variants per training clip:
  - Time-stretched spectrogram
  - Pitch-shifted spectrogram
  - (Gaussian noise is applied on-the-fly during training via the data generator)

Augmented files are saved as aug_<track_id>_<type>.npy beside the originals.
The splits.json is updated in-place to include augmented samples in the train set.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    AUGMENT_LOG,
    AUG_NOISE_STD,
    AUG_PITCH_STEPS,
    AUG_TIME_RATES,
    DURATION,
    HOP_LENGTH,
    IMG_SIZE,
    LOG_DIR,
    N_FFT,
    N_MELS,
    PROCESSED_DIR,
    SAMPLE_RATE,
    SPLITS_FILE,
    FMA_AUDIO_DIR,
)
from scripts.extract_features import audio_to_melspec, track_id_to_path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(AUGMENT_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-augmentation processors (all return np.ndarray shape (IMG_SIZE, IMG_SIZE))
# ---------------------------------------------------------------------------

def _time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Speed audio up/down by *rate* without changing pitch."""
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    mel = librosa.feature.melspectrogram(
        y=y_stretched, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean, std = mel_db.mean(), mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std
    sy, sx = IMG_SIZE / mel_db.shape[0], IMG_SIZE / mel_db.shape[1]
    return zoom(mel_db, (sy, sx)).astype(np.float32)


def _pitch_shift(y: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    """Shift pitch by *n_steps* semitones without changing speed."""
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    mel = librosa.feature.melspectrogram(
        y=y_shifted, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean, std = mel_db.mean(), mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std
    sy, sx = IMG_SIZE / mel_db.shape[0], IMG_SIZE / mel_db.shape[1]
    return zoom(mel_db, (sy, sx)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def augment_training_set() -> None:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"splits.json not found at {SPLITS_FILE}. Run extract_features.py first."
        )

    splits = json.loads(SPLITS_FILE.read_text())
    train_map: dict[str, int] = splits["train"]

    new_entries: dict[str, int] = {}
    skipped = 0

    for track_id_str, label in tqdm(train_map.items(), desc="Augmenting"):
        audio_path = track_id_to_path(int(track_id_str))
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        except Exception as exc:
            log.warning(f"Could not load {audio_path}: {exc}")
            skipped += 1
            continue

        # Time-stretch variants
        for rate in AUG_TIME_RATES:
            tag = f"ts{int(rate * 100)}"
            out_path = PROCESSED_DIR / f"aug_{track_id_str}_{tag}.npy"
            if not out_path.exists():
                try:
                    arr = _time_stretch(y, sr, rate)
                    np.save(out_path, arr)
                except Exception as exc:
                    log.warning(f"Time-stretch failed for {track_id_str} rate={rate}: {exc}")
                    continue
            new_entries[f"aug_{track_id_str}_{tag}"] = label

        # Pitch-shift variants
        for steps in AUG_PITCH_STEPS:
            tag = f"ps{steps:+d}"
            out_path = PROCESSED_DIR / f"aug_{track_id_str}_{tag}.npy"
            if not out_path.exists():
                try:
                    arr = _pitch_shift(y, sr, steps)
                    np.save(out_path, arr)
                except Exception as exc:
                    log.warning(f"Pitch-shift failed for {track_id_str} steps={steps}: {exc}")
                    continue
            new_entries[f"aug_{track_id_str}_{tag}"] = label

    splits["train"].update(new_entries)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))

    log.info(
        f"Augmentation done. Added {len(new_entries)} samples. "
        f"Skipped {skipped} tracks. "
        f"New training set size: {len(splits['train'])}"
    )


if __name__ == "__main__":
    augment_training_set()
