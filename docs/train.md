# `train.py` — Stage 4: Training the Model

## Code Walkthrough

This script supports **two models** to classify music genres from Mel Spectrogram images:
1. **MobileNetV2 (transfer learning)**
2. **Custom 4-block CNN (trained from scratch)**

Run:
```bash
python scripts/train.py --model mobilenetv2
python scripts/train.py --model custom
```

### `SpectrogramDataset` — the data generator

Rather than loading all 70,000+ spectrogram arrays into RAM at once (which would require ~40+ GB), this class loads them **batch by batch on demand**. It inherits from `tf.keras.utils.Sequence`, which Keras knows how to use during `.fit()`.

- `__len__()` returns the number of batches
- `__getitem__(idx)` loads one batch: reads `.npy` files, stacks them into `(B, 224, 224, 3)`, and optionally adds Gaussian noise
- `on_epoch_end()` shuffles the index array so the model sees examples in a different order each epoch

**3-channel conversion:**
```python
arr = np.stack([arr, arr, arr], axis=-1)  # (224, 224) → (224, 224, 3)
```
MobileNetV2 expects 3-channel input (RGB). By stacking the spectrogram with itself 3 times we create a valid 3-channel grayscale image without losing any information.

### `build_mobilenetv2(n_classes)` — transfer learning architecture

Creates a MobileNetV2 base (weights frozen in Phase 1) and attaches a new classification head:

```
Inputs → MobileNetV2 (frozen) → GlobalAveragePooling2D
       → BatchNormalization → Dense(512) → Dropout(0.4)
       → Dense(256) → Dropout(0.3) → Dense(n_genres, softmax)
```

### `train()` — training loops

**Phase 1 – Warm-Up (5 epochs)**
- MobileNetV2 base is frozen (`base.trainable = False`)
- Only the new Dense head is trained
- LR = 0.001 — high enough to converge quickly

**Phase 2 – Fine-Tuning (up to 40+ epochs)**
- The top 50 MobileNetV2 layers are unfrozen
- LR drops to 0.00001 — must be tiny to avoid destroying pre-trained features
- All 4 callbacks are active

**Custom CNN (single phase)**
- 4 convolutional blocks with BatchNorm, MaxPool, and Dropout
- GlobalAveragePooling → Dense → Softmax
- Trained with early stopping and ReduceLROnPlateau

**Class weights** are passed to `.fit()`. These are computed by `compute_class_weight("balanced", ...)` which assigns higher weight to under-represented genres so the model doesn't just memorise majority classes.

---

## Theory Behind It

### What is Transfer Learning?

Transfer learning means taking a model that was trained on one task and **reusing its learned representations** for a different but related task.

MobileNetV2 was trained on **ImageNet** — 1.2 million images across 1000 different object categories. In doing so, it learned to detect edges, textures, patterns, and shapes in 2D images. These low-level features are **also useful** for detecting patterns in spectrograms (which are just 2D images with frequency on one axis and time on the other).

Training from scratch on 70,000 spectrograms would require much more data and compute to learn these fundamental features. Transfer learning shortens that to a few epochs of warm-up.

### Why Two Phases?

If we unfreeze all MobileNetV2 layers and start training with LR = 0.001, the gradients are large and will **destroy the pre-trained weights** — undoing all the learned ImageNet knowledge. The two-phase approach avoids this:

1. **Phase 1:** Train only the new head at a normal LR so it quickly learns to map MobileNetV2 features to genres
2. **Phase 2:** Unfreeze top layers and use a very small LR (10–100× smaller) so we can *gently nudge* the pre-trained features to specialise for spectrograms without erasing them

### What is GlobalAveragePooling?

After MobileNetV2's final convolutional layer, we have a 3D tensor of shape `(7, 7, 1280)`. GlobalAveragePooling takes the **spatial average** across the 7×7 grid for each of the 1280 channels, producing a `(1280,)` vector. This:

- Dramatically reduces parameters (vs Flatten → Dense)
- Provides a global summary of "what features are present"
- Reduces overfitting

### Why Dropout?

Dropout randomly sets a fraction of neuron outputs to zero during training. This prevents the network from becoming over-reliant on any single neuron (co-adaptation), forcing it to learn **redundant representations** that generalise better. At inference time, Dropout is disabled.

### What do the callbacks do?

| Callback | Purpose |
|---|---|
| `ModelCheckpoint` | Saves the model only when val_accuracy improves — ensures we keep the best version |
| `EarlyStopping` | Stops training if val_accuracy hasn't improved in 10 epochs — prevents wasted compute and overfitting |
| `ReduceLROnPlateau` | Halves the learning rate when val_loss stalls — helps escape local minima |
| `CSVLogger` | Records per-epoch metrics to a CSV for later analysis |

### What is Categorical Cross-Entropy?

For a 16-class classification problem, the model outputs a probability for each class (via softmax). Cross-entropy measures how different those predicted probabilities are from the true one-hot label. If the model assigns 90% probability to the correct class, loss is low. If it's only 10%, loss is high. The optimizer minimises this loss.

```
Loss = -Σ y_true × log(y_pred)
```
