# `download_data.py` — Stage 1: Downloading the FMA Dataset

## Code Walkthrough

This script downloads and extracts the two FMA zips needed for the project: the audio files and the metadata.

### `_download_file(url, dest)`

Uses `requests.get(..., stream=True)` to download in **1 MiB chunks** rather than loading the entire file into memory first. A `tqdm` progress bar is attached to the `content-length` header so you can see how far through the ~22 GB download you are.

### `_extract_zip(zip_path, extract_to)`

Opens the zip file and iterates over each member, extracting one at a time with a `tqdm` bar. This is slower than a single `extractall()` call but gives you live progress on what can be a 20+ minute operation.

### `_stage(url, zip_dest, extract_dir, sentinel_file)`

The idempotency logic lives here. Before downloading or extracting, it checks for a **sentinel file** — a file that only exists after a successful extraction. If the sentinel exists, the whole stage is silently skipped. This means you can safely re-run the script if a download was interrupted.

| Archive | Sentinel checked |
|---|---|
| `fma_medium.zip` | `data/fma_medium/000/` (first audio subfolder) |
| `fma_metadata.zip` | `data/fma_metadata/tracks.csv` |

### `download_all()`

The public entry point. Calls `_stage()` for both archives, then logs that everything is ready.

---

## Theory Behind It

### What is the FMA dataset?

The **Free Music Archive (FMA)** is an open-access dataset released specifically for Music Information Retrieval (MIR) research. It contains audio clips from the [Free Music Archive](https://freemusicarchive.org/), all licensed under Creative Commons.

`fma_medium` specifically contains:
- **25,000 tracks**, each trimmed to **30 seconds**
- Audio encoded at **128 kbps MP3**
- **16 top-level genres** (e.g. Electronic, Rock, Hip-Hop, Folk, Jazz…)
- Accompanying metadata: titles, artists, genres, play counts

### Why streaming download?

A standard HTTP request like `requests.get(url).content` loads the entire response into RAM before writing. For a 22 GB file this would crash most machines. Streaming mode (`stream=True`) downloads and writes one chunk at a time, keeping memory usage near-constant at ~1 MiB regardless of file size.

### Why idempotency matters

Long-running data pipelines get interrupted — power cuts, network drops, keyboard interrupts. If each stage checks for its own output before running, re-executing the pipeline is safe and fast: completed stages are skipped instantly, and only the interrupted stage is retried.
