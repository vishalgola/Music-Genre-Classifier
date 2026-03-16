# `app.py` — Flask Server & Prediction API

## Code Walkthrough

This script runs the web server that ties the trained model to the browser UI.

### Module globals: `_model` and `_label_map`

The TensorFlow model is loaded **once at startup** into a module-level variable. If it were reloaded on every request, inference would take 10–30 seconds per call instead of under 1 second. `_label_map` converts integer class indices back to genre name strings.

### `_load_model()`

Called once when the server starts (or by `main.py`). If `outputs/model.keras` doesn't exist, it raises a clear `FileNotFoundError` rather than crashing silently later.

### `_allowed_file(filename)`

Validates the file extension before processing. Accepted types: `.mp3`, `.wav`, `.ogg`, `.flac`. This prevents users from accidentally uploading a video file or a PDF, which would cause confusing errors from librosa.

### `_audio_to_spectrogram(audio_bytes)`

The core prediction pre-processing step:
1. Write audio bytes to a temporary file (librosa needs a path, not a buffer)
2. Load with `librosa.load()` — identical settings as `extract_features.py`
3. Compute and normalise Mel Spectrogram — identical to training
4. Resize to 224×224 with scipy zoom
5. Stack to 3 channels → add batch dimension → return `(1, 224, 224, 3)` tensor

**Critically, this must match `extract_features.py` exactly.** Any difference (different sample rate, different normalisation formula, different resize method) would cause the model to receive inputs it was never trained on, resulting in garbage predictions.

### `POST /predict`

```
Request:  multipart/form-data with key "audio"
Response: JSON { top_genre, confidence, probabilities }
```

Returns HTTP 400 for missing/invalid files, 503 if the model isn't loaded, and 500 for unexpected errors — each with a clear error message.

### `app.config["MAX_CONTENT_LENGTH"]`

Flask enforces this automatically — requests larger than 50 MB are rejected with a 413 error before any Python code runs.

---

## Theory Behind It

### What is a REST API?

REST (Representational State Transfer) is an architectural style for web APIs. Key principle: each request contains all information needed to process it — the server doesn't remember anything between requests (stateless).

Our `/predict` endpoint is a simple REST resource:
- **Resource:** A genre prediction
- **Method:** `POST` (sends data to create a new result)
- **Format:** JSON response (language-agnostic, parseable by any JS frontend)

### Why Flask over FastAPI or Django?

Flask is a **micro-framework** — it provides just routing and request/response handling, nothing more. For a single-endpoint ML inference server, this is ideal:

- Minimal dependencies
- Easier to bundle into a PyInstaller `.exe` later
- No ORM, authentication, or admin overhead needed
- Runs synchronously, which is fine for local use (one user at a time)

### On-the-fly inference vs. pre-computed features

During **training**, we pre-compute and cache spectrograms as `.npy` files because each track is processed many times (multiple epochs). During **inference**, each audio file is processed exactly once, so computing the spectrogram on-the-fly is perfectly efficient.

### Why `tempfile` for the audio bytes?

We receive an audio file as raw bytes from the HTTP request. Librosa's `load()` function only accepts file paths or file objects — not raw byte strings. Writing to a `tempfile` and passing the path is the simplest cross-platform solution. The temp file is deleted immediately after loading.
