"""
download_data.py — Stage 1: Download FMA Medium audio and metadata.

Run independently:
    python scripts/download_data.py

Idempotent: skips files that have already been downloaded and extracted.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── add project root to path so config is importable ─────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.config import (
    DATA_DIR,
    DOWNLOAD_LOG,
    FMA_AUDIO_DIR,
    FMA_AUDIO_URL,
    FMA_META_DIR,
    FMA_META_URL,
    LOG_DIR,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DOWNLOAD_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest*, showing a tqdm progress bar."""
    log.info(f"Downloading {url} → {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
            fh.write(chunk)
            bar.update(len(chunk))

    log.info(f"Download complete: {dest}")


def _is_valid_zip(zip_path: Path) -> bool:
    """Return True if *zip_path* has a readable central directory.

    Uses only the zip's table-of-contents (central directory), which is
    parsed in milliseconds regardless of archive size.  A full CRC scan
    via ``testzip()`` would read every byte of a 22 GB archive and appear
    to hang for many minutes.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, EOFError, OSError):
        return False


def _extract_zip(zip_path: Path, extract_to: Path, stamp: Path) -> None:
    """Extract *zip_path* into *extract_to*, showing a tqdm progress bar.

    Writes *stamp* only after every member has been extracted successfully.
    An interrupted extraction leaves no stamp so the next run re-extracts.
    """
    log.info(f"Extracting {zip_path} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    stamp.unlink(missing_ok=True)  # clear any stale stamp before starting

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zf.extract(member, extract_to)

    stamp.touch()  # written ONLY after full successful extraction
    log.info(f"Extraction complete: {extract_to}")


def _stage(
    url: str,
    zip_dest: Path,
    extract_root: Path,
    stamp: Path,
) -> None:
    """
    Download + extract one FMA zip if not already present.

    *extract_root* — directory the zip is extracted INTO.
    *stamp*        — fixed file touched only after successful extraction;
                     lives in DATA_DIR so it never moves with config changes.
    """
    if stamp.exists():
        log.info(f"Already fully extracted ({zip_dest.stem}). Skipping.")
        return

    if zip_dest.exists():
        if _is_valid_zip(zip_dest):
            log.info(f"Zip already present and valid ({zip_dest}). Skipping download.")
        else:
            log.warning(
                f"Zip present but corrupt or incomplete ({zip_dest}). "
                "Deleting and re-downloading."
            )
            zip_dest.unlink()
            _download_file(url, zip_dest)
    else:
        _download_file(url, zip_dest)

    _extract_zip(zip_dest, extract_root, stamp)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def download_all() -> None:
    """Download and extract both the FMA Medium audio and metadata zips."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Stamps live in DATA_DIR with fixed names — independent of FMA_*_DIR.
    _stage(
        url=FMA_AUDIO_URL,
        zip_dest=DATA_DIR / "fma_medium.zip",
        extract_root=DATA_DIR,
        stamp=DATA_DIR / ".fma_medium_done",
    )

    _stage(
        url=FMA_META_URL,
        zip_dest=DATA_DIR / "fma_metadata.zip",
        extract_root=DATA_DIR,
        stamp=DATA_DIR / ".fma_metadata_done",
    )

    log.info("All data ready.")


if __name__ == "__main__":
    download_all()
