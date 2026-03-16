"""
train.py - Stage 4: Train model(s) on Mel Spectrogram tensors.

Run:
    python scripts/train.py --model mobilenetv2
    python scripts/train.py --model custom

MobileNetV2 uses two-phase training:
  Phase 1: freeze base, train head
  Phase 2: unfreeze top layers, fine-tune at lower LR

Custom CNN uses a single-phase training loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    AUG_NOISE_STD,
    BATCH_SIZE,
    CNN_BASE_FILTERS,
    CNN_DROPOUT,
    CNN_EPOCHS,
    CNN_L2,
    CNN_LR,
    DENSE_1,
    DENSE_2,
    DROPOUT_1,
    DROPOUT_2,
    EARLY_STOP_PATIENCE,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LOG_DIR,
    METRICS_TRAIN_CNN,
    METRICS_TRAIN_MNET,
    MODEL_PATH_CNN,
    MODEL_PATH_MNET,
    OUTPUTS_DIR,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE1_REDUCE_LR_PATIENCE,
    PHASE2_EPOCHS,
    PHASE2_LR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SPLITS_FILE,
    TRAIN_HISTORY_CNN,
    TRAIN_HISTORY_MNET,
    TRAIN_LOG_CNN,
    TRAIN_LOG_MNET,
    UNFREEZE_LAYERS,
)
from scripts.gpu_utils import setup_gpu

log = logging.getLogger(__name__)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

class SpectrogramDataset(tf.keras.utils.Sequence):
    """Keras-compatible data generator for Mel Spectrogram .npy files."""

    def __init__(
        self,
        sample_map: dict[str, int],
        batch_size: int = BATCH_SIZE,
        n_classes: int = 16,
        augment: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.keys = list(sample_map.keys())
        self.labels = [sample_map[k] for k in self.keys]
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.keys) / self.batch_size))

    def on_epoch_end(self) -> None:
        self.indices = np.arange(len(self.keys))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, y = [], []
        for i in batch_idx:
            key = self.keys[i]
            npy_path = PROCESSED_DIR / f"{key}.npy"
            arr = np.load(npy_path)                      # (224, 224) float32
            arr = np.stack([arr, arr, arr], axis=-1)     # -> (224, 224, 3)
            if self.augment:
                arr += np.random.normal(0, AUG_NOISE_STD, arr.shape).astype(np.float32)
            X.append(arr)
            y.append(self.labels[i])
        X = np.array(X, dtype=np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mobilenetv2(n_classes: int) -> tuple[keras.Model, keras.Model]:
    """Build MobileNetV2-based classifier."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Phase 1: freeze entire base

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(DENSE_1, activation="relu")(x)
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(DENSE_2, activation="relu")(x)
    x = layers.Dropout(DROPOUT_2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base


def build_custom_cnn(n_classes: int) -> keras.Model:
    """Build a 4-block custom CNN classifier."""
    filters = [
        CNN_BASE_FILTERS,
        CNN_BASE_FILTERS * 2,
        CNN_BASE_FILTERS * 4,
        CNN_BASE_FILTERS * 8,
    ]
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for f in filters:
        x = layers.Conv2D(
            f,
            kernel_size=3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(CNN_L2),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(CNN_DROPOUT)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(CNN_DROPOUT)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logging(log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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


def _write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model_name: str = "mobilenetv2") -> None:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found. Run extract_features.py first.")

    model_name = model_name.lower().strip()
    if model_name not in {"mobilenetv2", "custom"}:
        raise ValueError("model_name must be 'mobilenetv2' or 'custom'")

    log_path = TRAIN_LOG_MNET if model_name == "mobilenetv2" else TRAIN_LOG_CNN
    history_path = TRAIN_HISTORY_MNET if model_name == "mobilenetv2" else TRAIN_HISTORY_CNN
    model_path = MODEL_PATH_MNET if model_name == "mobilenetv2" else MODEL_PATH_CNN
    metrics_path = METRICS_TRAIN_MNET if model_name == "mobilenetv2" else METRICS_TRAIN_CNN

    _configure_logging(log_path)
    setup_gpu(log)

    splits = json.loads(SPLITS_FILE.read_text())
    label_map: dict[str, int] = splits["label_map"]
    n_classes = len(label_map)
    log.info(f"Number of genres: {n_classes} - {list(label_map.keys())}")

    train_map = splits["train"]
    val_map   = splits["val"]
    test_map  = splits["test"]

    log.info(f"Samples - train: {len(train_map)}, val: {len(val_map)}, test: {len(test_map)}")

    # Class weights for imbalance
    all_labels = np.array(list(train_map.values()))
    cw = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    class_weight = {i: w for i, w in enumerate(cw)}

    train_gen = SpectrogramDataset(train_map, n_classes=n_classes, augment=True)
    val_gen   = SpectrogramDataset(val_map,   n_classes=n_classes, augment=False, shuffle=False)

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=1,
        ),
        CSVLogger(str(history_path), append=False),
    ]

    start = time.perf_counter()

    if model_name == "mobilenetv2":
        model, base = build_mobilenetv2(n_classes)
        model.summary(print_fn=log.info)

        log.info("=== Phase 1: Warming up classification head ===")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=PHASE1_EPOCHS,
            class_weight=class_weight,
            callbacks=[
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=REDUCE_LR_FACTOR,
                    patience=PHASE1_REDUCE_LR_PATIENCE,
                    verbose=1,
                )
            ],
            verbose=1,
        )

        log.info(f"=== Phase 2: Fine-tuning top {UNFREEZE_LAYERS} MobileNetV2 layers ===")
        base.trainable = True
        for layer in base.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=PHASE2_EPOCHS,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model = build_custom_cnn(n_classes)
        model.summary(print_fn=log.info)
        log.info("=== Training custom CNN ===")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CNN_LR),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=CNN_EPOCHS,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

    train_seconds = round(time.perf_counter() - start, 2)
    _write_metrics(
        metrics_path,
        {
            "model": model_name,
            "train_seconds": train_seconds,
            "train_samples": len(train_map),
            "val_samples": len(val_map),
            "classes": n_classes,
        },
    )

    log.info(f"Training complete. Best model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a music genre classifier.")
    parser.add_argument(
        "--model",
        choices=["mobilenetv2", "custom"],
        default="mobilenetv2",
        help="Which model to train.",
    )
    args = parser.parse_args()
    train(args.model)
