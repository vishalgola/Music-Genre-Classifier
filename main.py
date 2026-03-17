"""
main.py - Pipeline orchestrator and application entry point.

Running this file:
    python main.py

Will check which stages have already been completed and run
only the ones that are missing, then launch the Flask UI.

Stages:
    1. Download   -> data/fma_medium/ and data/fma_metadata/
    2. Extract    -> data/processed/*.npy + data/splits.json
    3. Augment    -> augmented .npy files added to splits.json train set
    4. Train      -> outputs/model_mobilenetv2.keras + outputs/model_custom_cnn.keras
    5. Evaluate   -> outputs/confusion_matrix_*.png
    6. Compare    -> outputs/comparison.csv + outputs/comparison.md
    7. Serve      -> Flask app at http://localhost:5000
"""

from __future__ import annotations

import importlib
import logging
import sys
import webbrowser
from pathlib import Path
from threading import Timer

# Ensure scripts/ is importable
sys.path.append(str(Path(__file__).resolve().parent))

from scripts.config import (
    COMPARE_CSV,
    COMPARE_MD,
    CONFUSION_PATH_CNN,
    CONFUSION_PATH_MNET,
    CONFUSION_PATH_TRANS,
    CONFUSION_PATH_YAMNET,
    FMA_AUDIO_DIR,
    FMA_META_DIR,
    FLASK_HOST,
    FLASK_PORT,
    MODEL_PATH_CNN,
    MODEL_PATH_MNET,
    MODEL_PATH_TRANS,
    MODEL_PATH_YAMNET,
    PROCESSED_DIR,
    SPLITS_FILE,
    EMBED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DIVIDER = "=" * 60


def _stage(name: str, check: bool, module: str, fn: str, **kwargs) -> None:
    """Run *module.fn(**kwargs)* if *check* is False (stage not done yet)."""
    if check:
        log.info(f"[SKIP]  {name} - already complete.")
        return
    log.info(f"\n{DIVIDER}\n[RUN]   {name}\n{DIVIDER}")
    mod = importlib.import_module(module)
    getattr(mod, fn)(**kwargs)
    log.info(f"[DONE]  {name}\n")


def main() -> None:
    log.info(f"\n{'#'*60}\n  Music Genre Classifier - Pipeline Orchestrator\n{'#'*60}\n")

    # -- Stage 1: Download ---------------------------------------------------
    _stage(
        name="Download FMA Medium",
        check=(FMA_AUDIO_DIR / "000").exists() and (FMA_META_DIR / "tracks.csv").exists(),
        module="scripts.download_data",
        fn="download_all",
    )

    # -- Stage 2: Feature Extraction ----------------------------------------
    _stage(
        name="Extract Mel Spectrograms",
        check=SPLITS_FILE.exists() and any(PROCESSED_DIR.glob("*.npy")),
        module="scripts.extract_features",
        fn="extract_all",
    )

    _stage(
        name="Extract YAMNet Embeddings",
        check=EMBED_DIR.exists() and any(EMBED_DIR.glob("*.npy")),
        module="scripts.extract_embeddings",
        fn="extract_all",
    )

    # -- Stage 3: Augmentation -----------------------------------------------
    _stage(
        name="Data Augmentation",
        check=any(PROCESSED_DIR.glob("aug_*.npy")),
        module="scripts.augment",
        fn="augment_training_set",
    )

    # -- Stage 4: Training ---------------------------------------------------
    _stage(
        name="Train MobileNetV2 Model",
        check=MODEL_PATH_MNET.exists(),
        module="scripts.train",
        fn="train",
        model_name="mobilenetv2",
    )

    _stage(
        name="Train Custom CNN Model",
        check=MODEL_PATH_CNN.exists(),
        module="scripts.train",
        fn="train",
        model_name="custom",
    )

    _stage(
        name="Train Transformer Model",
        check=MODEL_PATH_TRANS.exists(),
        module="scripts.train",
        fn="train",
        model_name="transformer",
    )

    _stage(
        name="Train YAMNet Classifier",
        check=MODEL_PATH_YAMNET.exists(),
        module="scripts.train",
        fn="train",
        model_name="yamnet",
    )

    # -- Stage 5: Evaluation -------------------------------------------------
    _stage(
        name="Evaluate MobileNetV2 Model",
        check=CONFUSION_PATH_MNET.exists(),
        module="scripts.evaluate",
        fn="evaluate",
        model_name="mobilenetv2",
    )

    _stage(
        name="Evaluate Custom CNN Model",
        check=CONFUSION_PATH_CNN.exists(),
        module="scripts.evaluate",
        fn="evaluate",
        model_name="custom",
    )

    _stage(
        name="Evaluate Transformer Model",
        check=CONFUSION_PATH_TRANS.exists(),
        module="scripts.evaluate",
        fn="evaluate",
        model_name="transformer",
    )

    _stage(
        name="Evaluate YAMNet Model",
        check=CONFUSION_PATH_YAMNET.exists(),
        module="scripts.evaluate",
        fn="evaluate",
        model_name="yamnet",
    )

    # -- Stage 6: Compare ----------------------------------------------------
    _stage(
        name="Compare Models",
        check=COMPARE_CSV.exists() and COMPARE_MD.exists(),
        module="scripts.compare_models",
        fn="main",
    )

    # -- Stage 7: Launch Flask UI -------------------------------------------
    log.info(f"\n{DIVIDER}")
    log.info(f"  Launching UI at http://{FLASK_HOST}:{FLASK_PORT}")
    log.info(f"{DIVIDER}\n")

    url = f"http://{FLASK_HOST}:{FLASK_PORT}"
    # Open browser after a short delay so Flask has time to start
    Timer(1.5, lambda: webbrowser.open(url)).start()

    # Import here to avoid heavy TF import at top
    from app.app import app, _load_models
    _load_models()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    main()
