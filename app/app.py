"""
app.py - Flask web server: serves the UI and exposes the /predict endpoint.

Run independently (models must exist in outputs/):
    python app/app.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from scipy.ndimage import zoom

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    FLASK_HOST,
    FLASK_PORT,
    HOP_LENGTH,
    IMG_SIZE,
    MAX_UPLOAD_MB,
    MODEL_PATH_CNN,
    MODEL_PATH_MNET,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    SPLITS_FILE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "flac"}

# Globals - loaded once at startup
_models: dict[str, tf.keras.Model] = {}
_label_map: dict[int, str] = {}


def _load_models() -> None:
    """Load trained models and genre label map into module globals."""
    global _models, _label_map

    model_paths = {
        "mobilenetv2": MODEL_PATH_MNET,
        "custom": MODEL_PATH_CNN,
    }
    _models = {}
    for name, path in model_paths.items():
        if not path.exists():
            log.warning(f"Model not found at {path} - skipping {name}.")
            continue
        log.info(f"Loading model '{name}' from {path}...")
        _models[name] = tf.keras.models.load_model(path)
        log.info(f"Model '{name}' loaded.")

    if not _models:
        raise FileNotFoundError("No models found. Train a model before running the app.")

    if SPLITS_FILE.exists():
        splits = json.loads(SPLITS_FILE.read_text())
        lm: dict[str, int] = splits["label_map"]
        _label_map = {v: k for k, v in lm.items()}
    else:
        log.warning("splits.json not found - genre names will fall back to indices.")


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _audio_to_spectrogram(audio_bytes: bytes) -> np.ndarray:
    """Convert raw audio bytes -> (1, IMG_SIZE, IMG_SIZE, 3) tensor."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=30.0)
    Path(tmp_path).unlink(missing_ok=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean, std = mel_db.mean(), mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std

    sy = IMG_SIZE / mel_db.shape[0]
    sx = IMG_SIZE / mel_db.shape[1]
    mel_resized = zoom(mel_db, (sy, sx)).astype(np.float32)

    # -> (224, 224, 3) -> (1, 224, 224, 3)
    tensor = np.stack([mel_resized, mel_resized, mel_resized], axis=-1)
    return np.expand_dims(tensor, axis=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model", "").strip().lower()
    if not model_name:
        return jsonify({"error": "Model selection required."}), 400
    if model_name not in _models:
        available = ", ".join(sorted(_models.keys()))
        return jsonify({"error": f"Model '{model_name}' not available. Available: {available}"}), 400

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use MP3, WAV, OGG, or FLAC."}), 400

    try:
        audio_bytes = file.read()
        tensor = _audio_to_spectrogram(audio_bytes)
        probs = _models[model_name].predict(tensor, verbose=0)[0]  # (n_classes,)

        probabilities = {
            _label_map.get(i, str(i)): float(round(float(p), 4))
            for i, p in enumerate(probs)
        }
        top_idx = int(np.argmax(probs))
        top_genre = _label_map.get(top_idx, str(top_idx))
        confidence = float(round(float(probs[top_idx]), 4))

        return jsonify({
            "model": model_name,
            "top_genre": top_genre,
            "confidence": confidence,
            "probabilities": probabilities,
        })

    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_models()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
