# 🎵 Music Genre Classification using Deep Learning

## 📌 Project Overview

This project automatically classifies music tracks into different genres using a Deep Learning model. Audio files are processed to extract Mel Spectrogram images, which are then used to fine-tune a MobileNetV2 CNN. Results are served through an animated web interface.

**Dataset:** FMA Medium (25,000 tracks, 16 genres, ~22 GB)  
**Models:** MobileNetV2 with Transfer Learning + Custom CNN + Transformer + YAMNet embeddings  
**UI:** Animated Flask Web App (drag & drop → waveform → genre chart)
=======
This project automatically classifies music tracks into different genres using a Deep Learning model. Audio files are processed to extract important features such as MFCCs and Mel Spectrograms, which are then used to train a neural network for genre prediction.
>>>>>>> 2155cfaa1d8fbe5af83e7b2e784a6ccfcefb5a6b

---

## 🚀 Features

<<<<<<< HEAD
- Automatic music genre prediction from MP3/WAV/OGG/FLAC files
- Mel Spectrogram feature extraction with z-score normalization
- MobileNetV2 transfer learning with two-phase fine-tuning
- Custom 4-block CNN baseline (trained from scratch)
- Transformer (ViT-style) baseline on spectrograms
- YAMNet embeddings + classifier (pretrained audio model)
- Data augmentation (time-stretch, pitch-shift, Gaussian noise)
- Side-by-side model comparison metrics (accuracy, F1, training time)
- Animated web UI with waveform visualizer and genre probability chart
- Model selector to compare predictions from different architectures
- Fully modular pipeline — each stage independently runnable
- Detailed explanation docs for every module in `docs/`
=======
* Automatic music genre prediction
* Audio preprocessing and feature extraction
* Deep Learning model for classification
* Model evaluation using performance metrics
* Visualization of audio features
>>>>>>> 2155cfaa1d8fbe5af83e7b2e784a6ccfcefb5a6b

---

## 🛠️ Technologies Used

<<<<<<< HEAD
- **Python 3.10+**
- **TensorFlow / Keras** — Model training and inference
- **Librosa** — Audio loading and Mel Spectrogram extraction
- **Flask** — Web server and REST API
- **NumPy, Pandas, Scikit-learn** — Data processing and splitting
- **Matplotlib, Seaborn** — Evaluation visualization
- **WaveSurfer.js** — Browser audio waveform rendering
- **Chart.js** — Animated genre probability bar chart

---

## 📂 Project Structure

```
Music-Genre-Classifier/
├── scripts/
│   ├── config.py            # Shared paths & hyperparameters
│   ├── download_data.py     # Stage 1: Download FMA Medium
│   ├── extract_features.py  # Stage 2: Audio → Mel Spectrogram .npy
│   ├── augment.py           # Stage 3: Time-stretch + pitch-shift augmentation
│   ├── train.py             # Stage 4: Train MobileNetV2, Custom CNN, Transformer, or YAMNet classifier
│   ├── extract_embeddings.py# Stage 2b: YAMNet embeddings
│   ├── evaluate.py          # Stage 5: Accuracy, F1, confusion matrix
│   ├── compare_models.py    # Stage 6: Comparison report (CSV/MD)
│   └── mel_preview.py       # Utility: preview a single spectrogram
├── app/
│   ├── app.py               # Flask server + /predict endpoint
│   ├── static/
│   │   ├── style.css        # Dark theme + all CSS animations
│   │   └── main.js          # WaveSurfer, Chart.js, fetch API logic
│   └── templates/
│       └── index.html
├── docs/                    # Plain-English explanation per module
│   ├── config.md
│   ├── download_data.md
│   ├── extract_features.md  # Covers Mel Spectrogram & FFT theory
│   ├── augment.md           # Covers data augmentation theory
│   ├── train.md             # Covers transfer learning theory
│   ├── evaluate.md          # Covers accuracy vs. F1, confusion matrix
│   ├── app.md               # Covers Flask and REST API design
│   └── frontend.md          # Covers WaveSurfer, Chart.js, CSS animations
├── data/                    # (gitignored) Downloaded audio + processed features
├── outputs/                 # (gitignored) Trained model + logs
├── main.py                  # Pipeline orchestrator → auto-launches UI
├── requirements.txt
├── .gitignore
└── Readme.md
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

**GPU note (Windows):** TensorFlow GPU works best in **WSL2** with CUDA + cuDNN installed. Native Windows installs typically run on CPU unless you use a legacy CUDA stack.

### 2. Run everything (recommended)
```bash
python main.py
```
This will check which stages are already done and run only what's missing, then open the UI at `http://localhost:5000`.
The UI requires selecting a model before prediction.

### 3. Run stages independently (for debugging)
```bash
python scripts/download_data.py      # Download FMA Medium (~22 GB)
python scripts/extract_features.py   # Extract Mel Spectrograms
python scripts/extract_embeddings.py # Extract YAMNet embeddings
python scripts/augment.py            # Generate augmented training data

### MTG-Jamendo (train directly from TARs)
```bash
python scripts/build_splits_jamendo.py --metadata /path/to/metadata.tsv --label-col genre
python scripts/extract_embeddings_tar.py --metadata /path/to/metadata.tsv --tars-dir /path/to/tars
python scripts/train.py --model yamnet
python scripts/evaluate_tar.py --metadata /path/to/metadata.tsv --tars-dir /path/to/tars
```
python scripts/train.py --model mobilenetv2   # Train MobileNetV2
python scripts/train.py --model custom        # Train Custom CNN
python scripts/train.py --model transformer   # Train Transformer
python scripts/train.py --model yamnet        # Train YAMNet classifier
python scripts/evaluate.py --model mobilenetv2
python scripts/evaluate.py --model custom
python scripts/evaluate.py --model transformer
python scripts/evaluate.py --model yamnet
python scripts/compare_models.py              # Generate comparison report
python app/app.py                    # Run UI (model must exist)
```

---

## 📊 Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 1 | `download_data.py` | `data/fma_medium/` + `data/fma_metadata/` |
| 2 | `extract_features.py` | `data/processed/*.npy` + `data/splits.json` |
| 3 | `augment.py` | `data/processed/aug_*.npy` (training only) |
| 4 | `train.py` | `outputs/model_mobilenetv2.keras`, `outputs/model_custom_cnn.keras`, `outputs/model_transformer.keras`, `outputs/model_yamnet.keras` |
| 5 | `evaluate.py` | `outputs/confusion_matrix_*.png` + accuracy/F1 |
| 6 | `compare_models.py` | `outputs/comparison.csv` + `outputs/comparison.md` |
| 7 | `app.py` | Web UI at `http://localhost:5000` |

---

## 📚 Documentation

Every module has a dedicated explanation file in `docs/` covering:
1. **Code Walkthrough** — what each function does and why
2. **Theory** — the ML/DSP concepts that motivated the design

Start with [`docs/extract_features.md`](docs/extract_features.md) for Mel Spectrogram theory and [`docs/train.md`](docs/train.md) for transfer learning.
=======
* Python
* TensorFlow / Keras
* Librosa
* NumPy & Pandas
* Matplotlib

---

## 📂 Project Workflow

1. Audio Data Collection
2. Audio Preprocessing
3. Feature Extraction (MFCC / Mel Spectrogram)
4. Model Training using Deep Learning
5. Model Evaluation
6. Genre Prediction for new audio files

---

## 📊 Results

The trained model learns patterns from audio features and predicts the genre of unseen music tracks with good accuracy.

---

## ▶️ How to Run the Project

1. Clone the repository
2. Install required libraries
3. Run the training script
4. Use the trained model to classify new music files

---

## 📚 Future Improvements

* Improve model accuracy with larger datasets
* Add more genres for classification
* Deploy the model as a web application
>>>>>>> 2155cfaa1d8fbe5af83e7b2e784a6ccfcefb5a6b

---

## 👨‍💻 Author

**Vishal Prajapati and Kumar Aditya**

Data Science | Artificial Intelligence | Machine Learning
