"""
config.py — Single source of truth for all paths and hyperparameters.

Import this module in every other script instead of hardcoding values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).resolve().parent.parent  # project root

DATA_DIR       = ROOT_DIR / "data"
# Audio zip was extracted into data/fma_medium/ (before fix) → double-nested
FMA_AUDIO_DIR  = DATA_DIR / "fma_medium" / "fma_medium"
# Metadata zip was re-extracted into data/ (after fix) → single-nested
FMA_META_DIR   = DATA_DIR / "fma_metadata"
PROCESSED_DIR  = DATA_DIR / "processed"
EMBED_DIR      = DATA_DIR / "embeddings_yamnet"
SPLITS_FILE    = DATA_DIR / "splits.json"
SKIP_LIST_FILE = DATA_DIR / "skip_list.txt"

OUTPUTS_DIR    = ROOT_DIR / "outputs"
MODEL_PATH     = OUTPUTS_DIR / "model.keras"
MODEL_PATH_MNET = OUTPUTS_DIR / "model_mobilenetv2.keras"
MODEL_PATH_CNN  = OUTPUTS_DIR / "model_custom_cnn.keras"
MODEL_PATH_TRANS = OUTPUTS_DIR / "model_transformer.keras"
MODEL_PATH_YAMNET = OUTPUTS_DIR / "model_yamnet.keras"
CONFUSION_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CONFUSION_PATH_MNET = OUTPUTS_DIR / "confusion_matrix_mobilenetv2.png"
CONFUSION_PATH_CNN  = OUTPUTS_DIR / "confusion_matrix_custom_cnn.png"
CONFUSION_PATH_TRANS = OUTPUTS_DIR / "confusion_matrix_transformer.png"
CONFUSION_PATH_YAMNET = OUTPUTS_DIR / "confusion_matrix_yamnet.png"

LOG_DIR        = OUTPUTS_DIR / "logs"
DOWNLOAD_LOG   = LOG_DIR / "download.log"
EXTRACT_LOG    = LOG_DIR / "extract.log"
AUGMENT_LOG    = LOG_DIR / "augment.log"
TRAIN_LOG      = LOG_DIR / "train.log"
TRAIN_HISTORY  = LOG_DIR / "train_history.csv"
TRAIN_LOG_MNET     = LOG_DIR / "train_mobilenetv2.log"
TRAIN_LOG_CNN      = LOG_DIR / "train_custom_cnn.log"
TRAIN_LOG_TRANS    = LOG_DIR / "train_transformer.log"
TRAIN_LOG_YAMNET   = LOG_DIR / "train_yamnet.log"
TRAIN_HISTORY_MNET = LOG_DIR / "train_history_mobilenetv2.csv"
TRAIN_HISTORY_CNN  = LOG_DIR / "train_history_custom_cnn.csv"
TRAIN_HISTORY_TRANS = LOG_DIR / "train_history_transformer.csv"
TRAIN_HISTORY_YAMNET = LOG_DIR / "train_history_yamnet.csv"
EVAL_LOG       = LOG_DIR / "evaluate.log"
EVAL_LOG_MNET  = LOG_DIR / "evaluate_mobilenetv2.log"
EVAL_LOG_CNN   = LOG_DIR / "evaluate_custom_cnn.log"
EVAL_LOG_TRANS = LOG_DIR / "evaluate_transformer.log"
EVAL_LOG_YAMNET = LOG_DIR / "evaluate_yamnet.log"

METRICS_TRAIN_MNET = OUTPUTS_DIR / "metrics_train_mobilenetv2.json"
METRICS_TRAIN_CNN  = OUTPUTS_DIR / "metrics_train_custom_cnn.json"
METRICS_TRAIN_TRANS = OUTPUTS_DIR / "metrics_train_transformer.json"
METRICS_TRAIN_YAMNET = OUTPUTS_DIR / "metrics_train_yamnet.json"
METRICS_EVAL_MNET  = OUTPUTS_DIR / "metrics_eval_mobilenetv2.json"
METRICS_EVAL_CNN   = OUTPUTS_DIR / "metrics_eval_custom_cnn.json"
METRICS_EVAL_TRANS = OUTPUTS_DIR / "metrics_eval_transformer.json"
METRICS_EVAL_YAMNET = OUTPUTS_DIR / "metrics_eval_yamnet.json"

COMPARE_CSV = OUTPUTS_DIR / "comparison.csv"
COMPARE_MD  = OUTPUTS_DIR / "comparison.md"

# ---------------------------------------------------------------------------
# Download URLs  (official FMA GitHub release)
# ---------------------------------------------------------------------------
FMA_AUDIO_URL = (
    "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
)
FMA_META_URL = (
    "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
)

# ---------------------------------------------------------------------------
# Audio / Feature-Extraction Hyperparameters
# ---------------------------------------------------------------------------
SAMPLE_RATE  = 22_050      # Hz — librosa default; good for music
DURATION     = 30.0        # seconds per clip (FMA clips are exactly 30s)
N_FFT        = 2048        # FFT window size
HOP_LENGTH   = 512         # samples between frames
N_MELS       = 128         # mel filterbank bins
IMG_SIZE     = 224         # resize spectrogram to 224×224 for MobileNetV2

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
AUG_TIME_RATES   = [0.9, 1.1]   # speed multipliers (time-stretch)
AUG_PITCH_STEPS  = [-2, 2]      # semitones (pitch-shift)
AUG_NOISE_STD    = 0.005        # Gaussian noise standard deviation
SPEC_TIME_MASKS  = 2            # SpecAugment time masks per sample
SPEC_FREQ_MASKS  = 2            # SpecAugment freq masks per sample
SPEC_TIME_MAX    = 24           # Max time-mask width (pixels)
SPEC_FREQ_MAX    = 16           # Max freq-mask width (pixels)
MIXUP_ALPHA      = 0.4          # Mixup beta distribution alpha
MIXUP_PROB       = 0.5          # Probability of applying mixup per batch

# ---------------------------------------------------------------------------
# Data Split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 16
PHASE1_EPOCHS   = 18         # longer warm-up to improve head convergence
PHASE2_EPOCHS   = 45         # fine-tune
PHASE1_LR       = 3e-4
PHASE2_LR       = 1e-5
UNFREEZE_LAYERS = 60         # unfreeze more layers for better adaptation
DROPOUT_1       = 0.5        # stronger regularization
DROPOUT_2       = 0.5        # stronger regularization
DENSE_1         = 256        # fewer parameters to prevent memorization
DENSE_2         = 128        # fewer parameters
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE    = 10
REDUCE_LR_PATIENCE     = 3   # drop LR faster when validation stalls
REDUCE_LR_FACTOR       = 0.5
PHASE1_REDUCE_LR_PATIENCE = 2

CNN_EPOCHS      = 50
CNN_LR          = 1e-4
CNN_DROPOUT     = 0.5
CNN_L2          = 2e-4
CNN_BASE_FILTERS = 24

# ---------------------------------------------------------------------------
# Transformer (ViT-style) Hyperparameters
# ---------------------------------------------------------------------------
TRANS_EPOCHS    = 40
TRANS_LR        = 2e-4
TRANS_PATCH     = 16
TRANS_DEPTH     = 6
TRANS_HEADS     = 4
TRANS_MLP_DIM   = 256
TRANS_DROPOUT   = 0.2

# ---------------------------------------------------------------------------
# YAMNet Embeddings + Classifier
# ---------------------------------------------------------------------------
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16000
YAMNET_EMB_DIM = 1024
YAMNET_EPOCHS = 30
YAMNET_LR = 1e-3
YAMNET_DROPOUT = 0.4
YAMNET_DENSE = 256
YAMNET_SEG_SECONDS = 3.0
YAMNET_SEG_HOP = 1.5

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
MAX_UPLOAD_MB = 50          # maximum audio file size the UI will accept
