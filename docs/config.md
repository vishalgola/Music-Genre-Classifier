# `config.py` — Shared Configuration

## Code Walkthrough

`config.py` is the single source of truth for every path and hyperparameter in the project. No other script should ever hardcode a path or number — they import from here instead.

### Key sections

| Section | What it defines |
|---|---|
| **Paths** | Root directory, all `data/`, `outputs/`, and `logs/` paths as `pathlib.Path` objects |
| **Download URLs** | Official FMA server links for the audio and metadata zips |
| **Audio / Feature params** | Sample rate, FFT settings, mel bins, image size |
| **Augmentation** | Time-stretch rates, pitch-shift steps, noise level |
| **Data Split** | Train/val/test ratios, random seed |
| **Training** | Batch size, epochs, learning rates, dropout rates, callback patience values |
| **Flask** | Host, port, max upload size |

### Why `pathlib.Path` instead of strings?

`Path` objects let you combine parts with `/` (`ROOT_DIR / "data"`) rather than string concatenation, work correctly on both Windows and Linux, and expose useful methods like `.exists()`, `.mkdir()`, `.glob()`.

### How `ROOT_DIR` is computed

```python
ROOT_DIR = Path(__file__).resolve().parent.parent
```

`__file__` is the path of `config.py` itself. `.parent` goes up one level (to `scripts/`), and `.parent` again reaches the project root. This means the config works regardless of which directory you `cd` into before running a script.

---

## Theory Behind It

### Why a central config is essential

When constants are scattered across scripts, a small change (e.g. resizing images from 224 to 128) requires editing multiple files, introduces subtle mismatches, and makes debugging much harder. A single config file means:

- **One edit → all scripts updated automatically**
- **Documentation in one place** — any reader can see the full design at a glance
- **No silent divergence** — extract, train, and evaluate all use the *exact same* `IMG_SIZE`

### Why `RANDOM_SEED = 42`?

Setting a fixed random seed makes all stochastic operations (data splits, weight initialisation, shuffle order) **reproducible**. Anyone running the pipeline gets identical train/val/test splits, making results comparable.

### Why these specific audio parameters?

| Parameter | Value | Reason |
|---|---|---|
| `SAMPLE_RATE` | 22,050 Hz | Standard for music analysis; covers full human hearing range (20 Hz–20 kHz) via Nyquist |
| `N_FFT` | 2048 | ~93 ms window at 22 kHz — large enough to capture low frequencies cleanly |
| `HOP_LENGTH` | 512 | 25% overlap between frames; balances time resolution vs computation |
| `N_MELS` | 128 | 128 frequency bands on the mel scale; standard in music ML research |
| `IMG_SIZE` | 224 | Required by MobileNetV2 (pre-trained on ImageNet at 224×224) |
