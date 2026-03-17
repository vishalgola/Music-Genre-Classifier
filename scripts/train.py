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
    METRICS_TRAIN_TRANS,
    METRICS_TRAIN_YAMNET,
    EMBED_DIR,
    MODEL_PATH_CNN,
    MODEL_PATH_MNET,
    MODEL_PATH_TRANS,
    MODEL_PATH_YAMNET,
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
    TRAIN_HISTORY_TRANS,
    TRAIN_HISTORY_YAMNET,
    TRAIN_LOG_CNN,
    TRAIN_LOG_MNET,
    TRAIN_LOG_TRANS,
    TRAIN_LOG_YAMNET,
    UNFREEZE_LAYERS,
    SPEC_TIME_MASKS,
    SPEC_FREQ_MASKS,
    SPEC_TIME_MAX,
    SPEC_FREQ_MAX,
    MIXUP_ALPHA,
    MIXUP_PROB,
    TRANS_EPOCHS,
    TRANS_LR,
    TRANS_PATCH,
    TRANS_DEPTH,
    TRANS_HEADS,
    TRANS_MLP_DIM,
    TRANS_DROPOUT,
    YAMNET_EPOCHS,
    YAMNET_LR,
    YAMNET_DROPOUT,
    YAMNET_DENSE,
    YAMNET_EMB_DIM,
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
            if self.augment:
                arr = _spec_augment(arr)
            arr = np.stack([arr, arr, arr], axis=-1)     # -> (224, 224, 3)
            if self.augment:
                arr += np.random.normal(0, AUG_NOISE_STD, arr.shape).astype(np.float32)
            X.append(arr)
            y.append(self.labels[i])
        X = np.array(X, dtype=np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.augment:
            X, y = _mixup_batch(X, y)
        return X, y


class EmbeddingDataset(tf.keras.utils.Sequence):
    """Dataset for precomputed embeddings."""

    def __init__(
        self,
        sample_map: dict[str, int],
        batch_size: int = BATCH_SIZE,
        n_classes: int = 16,
        shuffle: bool = True,
    ) -> None:
        self.keys = list(sample_map.keys())
        self.labels = [sample_map[k] for k in self.keys]
        self.batch_size = batch_size
        self.n_classes = n_classes
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
            npy_path = EMBED_DIR / f"{key}.npy"
            arr = np.load(npy_path)  # (1024,)
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
        x = layers.SpatialDropout2D(0.1)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(CNN_DROPOUT)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(CNN_DROPOUT)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_transformer(n_classes: int) -> keras.Model:
    """Build a ViT-style transformer for spectrogram images."""
    patch_size = TRANS_PATCH
    num_patches = (IMG_SIZE // patch_size) * (IMG_SIZE // patch_size)
    embed_dim = TRANS_MLP_DIM

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embedding",
    )(inputs)
    x = layers.Reshape((num_patches, embed_dim))(x)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    x = x + pos_embed
    x = layers.Dropout(TRANS_DROPOUT)(x)

    for i in range(TRANS_DEPTH):
        # LayerNorm + MHA
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=TRANS_HEADS,
            key_dim=embed_dim // TRANS_HEADS,
            dropout=TRANS_DROPOUT,
            name=f"mha_{i}",
        )(x1, x1)
        x2 = layers.Add(name=f"skip_attn_{i}")([x, attn])
        # MLP block
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"ln_2_{i}")(x2)
        mlp = layers.Dense(TRANS_MLP_DIM * 2, activation="gelu")(x3)
        mlp = layers.Dropout(TRANS_DROPOUT)(mlp)
        mlp = layers.Dense(embed_dim)(mlp)
        x = layers.Add(name=f"skip_mlp_{i}")([x2, mlp])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(TRANS_DROPOUT)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_yamnet_classifier(n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(YAMNET_EMB_DIM,))
    x = layers.Dense(YAMNET_DENSE, activation="relu")(inputs)
    x = layers.Dropout(YAMNET_DROPOUT)(x)
    x = layers.Dense(YAMNET_DENSE // 2, activation="relu")(x)
    x = layers.Dropout(YAMNET_DROPOUT)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def _spec_augment(arr: np.ndarray) -> np.ndarray:
    """Apply SpecAugment time/frequency masking to a single spectrogram."""
    out = arr.copy()
    n_freq, n_time = out.shape
    # Frequency masks
    for _ in range(SPEC_FREQ_MASKS):
        f = np.random.randint(0, min(SPEC_FREQ_MAX, n_freq) + 1)
        if f == 0:
            continue
        f0 = np.random.randint(0, max(1, n_freq - f + 1))
        out[f0 : f0 + f, :] = 0.0
    # Time masks
    for _ in range(SPEC_TIME_MASKS):
        t = np.random.randint(0, min(SPEC_TIME_MAX, n_time) + 1)
        if t == 0:
            continue
        t0 = np.random.randint(0, max(1, n_time - t + 1))
        out[:, t0 : t0 + t] = 0.0
    return out


def _mixup_batch(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply mixup to a batch with probability MIXUP_PROB."""
    if MIXUP_ALPHA <= 0 or np.random.rand() > MIXUP_PROB:
        return X, y
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    idx = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix.astype(np.float32), y_mix.astype(np.float32)


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
    if model_name not in {"mobilenetv2", "custom", "transformer", "yamnet"}:
        raise ValueError("model_name must be 'mobilenetv2', 'custom', 'transformer', or 'yamnet'")

    if model_name == "mobilenetv2":
        log_path = TRAIN_LOG_MNET
        history_path = TRAIN_HISTORY_MNET
        model_path = MODEL_PATH_MNET
        metrics_path = METRICS_TRAIN_MNET
    elif model_name == "custom":
        log_path = TRAIN_LOG_CNN
        history_path = TRAIN_HISTORY_CNN
        model_path = MODEL_PATH_CNN
        metrics_path = METRICS_TRAIN_CNN
    elif model_name == "transformer":
        log_path = TRAIN_LOG_TRANS
        history_path = TRAIN_HISTORY_TRANS
        model_path = MODEL_PATH_TRANS
        metrics_path = METRICS_TRAIN_TRANS
    else:
        log_path = TRAIN_LOG_YAMNET
        history_path = TRAIN_HISTORY_YAMNET
        model_path = MODEL_PATH_YAMNET
        metrics_path = METRICS_TRAIN_YAMNET

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

    if model_name == "yamnet":
        train_gen = EmbeddingDataset(train_map, n_classes=n_classes, shuffle=True)
        val_gen   = EmbeddingDataset(val_map,   n_classes=n_classes, shuffle=False)
    else:
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
        if model_name == "custom":
            model = build_custom_cnn(n_classes)
        elif model_name == "transformer":
            model = build_transformer(n_classes)
        else:
            model = build_yamnet_classifier(n_classes)
        model.summary(print_fn=log.info)
        log.info(f"=== Training {model_name} ===")
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=CNN_LR if model_name == "custom" else (TRANS_LR if model_name == "transformer" else YAMNET_LR)
            ),
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=LABEL_SMOOTHING if model_name != "yamnet" else 0.0
            ),
            metrics=["accuracy"],
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=CNN_EPOCHS if model_name == "custom" else (TRANS_EPOCHS if model_name == "transformer" else YAMNET_EPOCHS),
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
        choices=["mobilenetv2", "custom", "transformer", "yamnet"],
        default="mobilenetv2",
        help="Which model to train.",
    )
    args = parser.parse_args()
    train(args.model)
