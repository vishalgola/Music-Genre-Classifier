"""Generate and visualize a mel spectrogram for one audio file."""

from __future__ import annotations

from pathlib import Path
import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def create_sample_audio(path: Path, sr: int = TARGET_SR, duration: float = 5.0) -> None:
    """Create a synthetic audio file so the pipeline is runnable in an empty workspace."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
    )
    # Simple fade in/out to avoid clicks at boundaries.
    fade_len = int(0.05 * sr)
    fade = np.linspace(0, 1, fade_len)
    signal[:fade_len] *= fade
    signal[-fade_len:] *= fade[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sr)


def mel_spectrogram(
    audio_path: Path,
    out_path: Path,
    sr: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    normalize: bool = False,
) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if normalize:
        mean = mel_db.mean()
        std = mel_db.std() + 1e-8
        mel_db = (mel_db - mean) / std

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {audio_path.name}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return mel_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("data/sample.wav"),
        help="Path to a source audio file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mel_spectrogram.png"),
        help="Output image path for spectrogram.",
    )
    parser.add_argument(
        "--create-sample-if-missing",
        action="store_true",
        help="Generate synthetic sample audio when --audio does not exist.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply per-spectrogram z-score normalization after dB conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        if args.create_sample_if_missing:
            create_sample_audio(args.audio)
        else:
            raise FileNotFoundError(
                f"Audio file not found: {args.audio}. "
                "Provide --audio or pass --create-sample-if-missing."
            )

    mel_db = mel_spectrogram(args.audio, args.out, normalize=args.normalize)
    print(f"Audio source: {args.audio}")
    print(f"Mel spectrogram saved to: {args.out}")
    print(f"Mel tensor shape (n_mels, frames): {mel_db.shape}")
    print(f"Mel tensor dtype: {mel_db.dtype}")
    print(f"Mel tensor stats: min={mel_db.min():.3f}, max={mel_db.max():.3f}")


if __name__ == "__main__":
    main()
