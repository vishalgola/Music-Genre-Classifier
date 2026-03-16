# `augment.py` — Stage 3: Data Augmentation

## Code Walkthrough

This script generates additional training examples by applying audio transformations to each training clip, then saves the new spectrograms and updates `splits.json`.

### `_time_stretch(y, sr, rate)`

Uses `librosa.effects.time_stretch(y, rate=rate)` to speed up or slow down the audio without changing pitch. A `rate > 1.0` speeds up, `rate < 1.0` slows down. The result is then converted to a Mel Spectrogram and resized, exactly like in `extract_features.py`.

The config defines two rates: `[0.9, 1.1]` — 10% slower and 10% faster.

### `_pitch_shift(y, sr, n_steps)`

Uses `librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)` to shift the pitch up or down by a given number of **semitones** without changing the speed.

The config defines two values: `[-2, 2]` — down 2 semitones and up 2 semitones.

### Gaussian Noise — handled in the training generator

Rather than storing noise-augmented files on disk (`disk_size × 2`), Gaussian noise is added **on-the-fly** inside the `SpectrogramDataset` generator in `train.py`. This saves disk space while still providing the augmentation benefit.

### Output filename convention

```
aug_{track_id}_{tag}.npy

Examples:
  aug_12345_ts90.npy   → time-stretch at 0.9x
  aug_12345_ts110.npy  → time-stretch at 1.1x
  aug_12345_ps-2.npy   → pitch-shift down 2 semitones
  aug_12345_ps+2.npy   → pitch-shift up 2 semitones
```

### Updating `splits.json`

After generating all augmented files, the script updates `splits["train"]` in-place by adding the new `aug_*` keys with the same label as their source track. Only the training set is augmented — validation and test sets are kept clean for unbiased evaluation.

---

## Theory Behind It

### What is data augmentation?

Data augmentation is the practice of **artificially expanding the training dataset** by creating plausible variations of existing examples. The model sees more variety, so it learns more robust features rather than memorising specific recordings.

Without augmentation, a model trained on 17,500 clips might learn to recognise a particular recording style (microphone, room acoustics, EQ) rather than the underlying genre. With augmentation, it's forced to learn features that are stable across these variations.

### Why time-stretch for music genre?

The genre of a song doesn't change if you play it 10% faster or slower — a jazz track played slightly faster is still jazz. But the spectrogram looks different (frames are compressed/stretched in time). Training on both speeds teaches the model that **temporal scale is not a genre-defining feature**.

### Why pitch-shift?

Similarly, a rock song transposed up 2 semitones is still rock. By training on pitch-shifted versions, the model learns to recognise **timbral patterns and rhythmic structures** rather than absolute pitch positions, which makes it more robust across songs in different keys.

### Why ±2 semitones, not more?

Beyond ~3–4 semitones, librosa's pitch-shifting algorithm starts to introduce audible artefacts (phase smearing, frequency leakage). The generated spectrograms would look unnatural and could confuse the model. ±2 is the sweet spot between meaningfully different and still realistic.

### Why only augment the training set?

The validation and test sets must remain **identical to real-world data** — if we augment them, our accuracy metrics would be measured on "easier" artificial variants rather than genuine tracks. The model should be evaluated on data it has never seen, in its original form.

### Effect on dataset size

| Set | Before augmentation | After (×4 per track) |
|---|---|---|
| Train | ~17,500 | ~70,000 |
| Val | ~3,750 | 3,750 (unchanged) |
| Test | ~3,750 | 3,750 (unchanged) |
