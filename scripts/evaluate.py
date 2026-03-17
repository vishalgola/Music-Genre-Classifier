"""
evaluate.py - Stage 5: Evaluate a trained model on the held-out test set.

Run:
    python scripts/evaluate.py --model mobilenetv2
    python scripts/evaluate.py --model custom
    python scripts/evaluate.py --model transformer
    python scripts/evaluate.py --model yamnet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    CONFUSION_PATH_CNN,
    CONFUSION_PATH_MNET,
    CONFUSION_PATH_TRANS,
    CONFUSION_PATH_YAMNET,
    EVAL_LOG_CNN,
    EVAL_LOG_MNET,
    EVAL_LOG_TRANS,
    EVAL_LOG_YAMNET,
    LOG_DIR,
    METRICS_EVAL_CNN,
    METRICS_EVAL_MNET,
    METRICS_EVAL_TRANS,
    METRICS_EVAL_YAMNET,
    MODEL_PATH_CNN,
    MODEL_PATH_MNET,
    MODEL_PATH_TRANS,
    MODEL_PATH_YAMNET,
    PROCESSED_DIR,
    EMBED_DIR,
    SPLITS_FILE,
)
from scripts.gpu_utils import setup_gpu

log = logging.getLogger(__name__)


def _configure_logging(log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for handler in list(log.handlers):
        log.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.setLevel(logging.INFO)


def load_test_data(test_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    missing = 0
    for key, label in test_map.items():
        npy_path = PROCESSED_DIR / f"{key}.npy"
        if not npy_path.exists():
            missing += 1
            continue
        arr = np.load(npy_path)                       # (224, 224) float32
        arr = np.stack([arr, arr, arr], axis=-1)      # -> (224, 224, 3)
        X.append(arr)
        y.append(label)
    if missing:
        log.warning(f"{missing} test tracks missing .npy files - skipped.")
    return np.array(X, dtype=np.float32), np.array(y)


def load_test_embeddings(test_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    missing = 0
    for key, label in test_map.items():
        npy_path = EMBED_DIR / f"{key}.npy"
        if not npy_path.exists():
            missing += 1
            continue
        arr = np.load(npy_path)                       # (1024,)
        X.append(arr)
        y.append(label)
    if missing:
        log.warning(f"{missing} test embeddings missing .npy files - skipped.")
    return np.array(X, dtype=np.float32), np.array(y)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Genre", fontsize=13)
    ax.set_ylabel("True Genre", fontsize=13)
    ax.set_title("Confusion Matrix - Music Genre Classifier", fontsize=15)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {out_path}")


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate(model_name: str = "mobilenetv2") -> None:
    model_name = model_name.lower().strip()
    if model_name not in {"mobilenetv2", "custom", "transformer", "yamnet"}:
        raise ValueError("model_name must be 'mobilenetv2', 'custom', 'transformer', or 'yamnet'")

    if model_name == "mobilenetv2":
        model_path = MODEL_PATH_MNET
        confusion_path = CONFUSION_PATH_MNET
        log_path = EVAL_LOG_MNET
        metrics_path = METRICS_EVAL_MNET
    elif model_name == "custom":
        model_path = MODEL_PATH_CNN
        confusion_path = CONFUSION_PATH_CNN
        log_path = EVAL_LOG_CNN
        metrics_path = METRICS_EVAL_CNN
    elif model_name == "transformer":
        model_path = MODEL_PATH_TRANS
        confusion_path = CONFUSION_PATH_TRANS
        log_path = EVAL_LOG_TRANS
        metrics_path = METRICS_EVAL_TRANS
    else:
        model_path = MODEL_PATH_YAMNET
        confusion_path = CONFUSION_PATH_YAMNET
        log_path = EVAL_LOG_YAMNET
        metrics_path = METRICS_EVAL_YAMNET

    _configure_logging(log_path)
    setup_gpu(log)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found. Run extract_features.py first.")

    splits = json.loads(SPLITS_FILE.read_text())
    label_map: dict[str, int] = splits["label_map"]
    idx_to_genre = {v: k for k, v in label_map.items()}
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    log.info("Loading test data...")
    if model_name == "yamnet":
        X_test, y_true = load_test_embeddings(splits["test"])
    else:
        X_test, y_true = load_test_data(splits["test"])
    log.info(f"Test set: {len(X_test)} samples, {len(class_names)} classes")

    log.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    log.info("Running predictions...")
    y_probs = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)

    log.info(f"\n{'='*60}")
    log.info(f"Model         : {model_name}")
    log.info(f"Test Accuracy : {acc:.4f}")
    log.info(f"Weighted F1   : {f1_weighted:.4f}")
    log.info(f"\n{report}")
    log.info(f"{'='*60}\n")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, confusion_path)

    _write_metrics(
        metrics_path,
        {
            "model": model_name,
            "accuracy": round(float(acc), 6),
            "f1_weighted": round(float(f1_weighted), 6),
            "test_samples": int(len(y_true)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model",
        choices=["mobilenetv2", "custom", "transformer", "yamnet"],
        default="mobilenetv2",
        help="Which model to evaluate.",
    )
    args = parser.parse_args()
    evaluate(args.model)
