# `extract_features.py` — Stage 2: Audio → Mel Spectrogram Tensors

## Code Walkthrough

This script converts every 30-second MP3 clip in `data/fma_medium/` into a 2D NumPy array representing its Mel Spectrogram, then saves it as a `.npy` file. It also builds the train/val/test split file.

### `track_id_to_path(track_id)`

FMA stores files in subfolders named by the first 3 digits of the zero-padded track ID:

```
track_id = 5 → "000005" → data/fma_medium/000/000005.mp3
track_id = 12345 → "012345" → data/fma_medium/012/012345.mp3
```

### `audio_to_melspec(audio_path)` — the core function

1. **Load audio** with `librosa.load()` at 22,050 Hz, forced to mono, up to 30s
2. **Compute Mel Spectrogram** via `librosa.feature.melspectrogram()`
3. **Convert to dB scale** with `librosa.power_to_db()` — logarithmic, matches human hearing
4. **Normalise** with z-score so every spectrogram has ~0 mean and ~1 std deviation
5. **Resize** from `(128, ~1292)` to `(224, 224)` using `scipy.ndimage.zoom`

### `_process_one(track_id)` — the parallel worker

Wrapped in `try/except`; any exception (corrupted MP3, truncated file, wrong format) is caught and the track ID + error message is recorded to `skip_list.txt`. The function returns a `(id, success, msg)` tuple.

### `ProcessPoolExecutor` — why parallel?

Feature extraction is CPU-bound (FFT calculations). Using `ProcessPoolExecutor` with 4 workers can cut total extraction time by 3–4×, since each worker runs on a separate CPU core.

### `load_track_metadata()`

Reads FMA's `tracks.csv` (a multi-level header CSV), keeps only rows where:
- `set.subset == "medium"` — confirms the track is in our chosen subset
- `track.genre_top is not null` — the track has a confirmed genre label

### `build_splits(track_ids, genre_labels)`

Uses scikit-learn's `train_test_split` with `stratify=genre_labels`. Stratification ensures each genre is represented in the same proportion across train, val, and test sets — without this, rare genres might end up almost entirely in one split.

---

## Theory Behind It

### What is a Mel Spectrogram?

A **spectrogram** shows how the frequency content of audio changes over time. The x-axis is time, the y-axis is frequency, and the colour (or brightness) shows energy at each frequency at each moment.

A **Mel Spectrogram** uses the **Mel scale** for the frequency axis — a perceptual scale that spaces frequencies the way humans actually hear them. Lower frequencies are spread out (we're very sensitive there), while higher frequencies are compressed (we distinguish them less precisely).

```
Linear frequency:  100 Hz → 200 Hz → 400 Hz → 800 Hz → 1600 Hz ...
Mel scale:         humans perceive all of these as ~equal steps
```

This makes Mel Spectrograms much more informative for music than a raw FFT, because they emphasise the frequency regions that matter most for musical content like melody, timbre, and rhythm.

### What is the Short-Time Fourier Transform (STFT)?

Librosa computes the spectrogram using the STFT:
1. Slide a window (`N_FFT = 2048 samples ≈ 93 ms`) over the audio
2. Apply Fourier Transform to each window → frequency magnitudes
3. Move the window forward by `HOP_LENGTH = 512 samples`
4. Stack all windows side by side → 2D time-frequency grid

### Why dB scale?

Raw power values span many orders of magnitude (e.g. 0.0001 to 1000). The dB conversion compresses this to a human-readable logarithmic range (typically –80 to 0 dB), which makes patterns in the spectrogram visually and numerically easier to learn from.

```
dB = 10 × log₁₀(power / reference_power)
```

### Why z-score normalisation?

Without normalisation, spectrograms from loud recordings have very different value ranges than quiet ones. Z-score normalisation (`(x - mean) / std`) brings every spectrogram to the same scale, so the model doesn't confuse loudness with genre.

### Why resize to 224×224?

MobileNetV2 was pre-trained on ImageNet images of exactly **224×224 pixels**. Feeding it tensors of a different shape would require retraining the entire network from scratch — we'd lose all the learned features. By resizing our `(128, ~1292)` spectrograms to `(224, 224)` we can leverage all that pre-trained knowledge.
