# GWO-Tuned CNN-BLSTM-Attention for Speech Emotion Recognition

> **A self-optimizing deep learning pipeline** that automatically discovers the best architecture and hyperparameters for speech emotion recognition — no manual tuning required.

---

## Overview

This project applies **Speech Emotion Recognition (SER)** on the [RAVDESS](https://zenodo.org/record/1188976) dataset using a hybrid deep learning model: a **Convolutional Neural Network (CNN)** for spatial feature extraction, a **Bidirectional LSTM (BLSTM)** for temporal modelling, and a **soft Attention mechanism** for context-weighted classification.

What makes this pipeline *self-configuring* is the **Grey Wolf Optimizer (GWO)** — a bio-inspired metaheuristic algorithm — which automatically searches for optimal hyperparameters (learning rate, dropout, filter sizes, LSTM hidden units, optimizer type, batch size) before the final training run begins.

The dataset is expanded **5× via audio augmentation** (noise injection, time shifting, pitch shifting, time stretching) to improve generalization on the relatively small RAVDESS corpus.

---

## Key Features

- **Hybrid Architecture** — CNN → BLSTM → Attention → Softmax, designed for audio spectrogram classification
- **Grey Wolf Optimizer (GWO)** — metaheuristic hyperparameter search; no manual grid search needed
- **5× Audio Augmentation** — four augmentation strategies applied per sample to combat data scarcity
- **Mel-Spectrogram Representation** — 128×128 log-mel spectrograms as visual audio fingerprints
- **End-to-End in PyTorch** — GPU-accelerated, fully reproducible (seeded)
- **RAVDESS Compatible** — built around the standard 8-emotion RAVDESS filename convention

---

## Architecture

```
Input (1 × 128 × 128 Mel-Spectrogram)
        │
        ▼
┌───────────────────────────────┐
│  CNN Block × 3                │
│  Conv2d → BatchNorm → ReLU    │
│  → MaxPool2d                  │
│  Filters: 32 → 64 → 128      │
└───────────────┬───────────────┘
                │  Reshape to sequence
                ▼
┌───────────────────────────────┐
│  Bidirectional LSTM           │
│  hidden_dim = 128 (× 2 dirs) │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Soft Attention               │
│  Linear → Softmax → Weighted  │
│  Sum over time steps          │
└───────────────┬───────────────┘
                │
                ▼
         Dropout → FC → Logits
                │
                ▼
        Emotion Class (8)
```

> Filter sizes, LSTM hidden units, and dropout are **automatically tuned by GWO**.

---

## Grey Wolf Optimizer

GWO mimics the social hierarchy and hunting behaviour of grey wolves. In this pipeline:

- Each **wolf** represents a candidate set of hyperparameters
- **Fitness** is measured as validation accuracy after a quick 3-epoch training run
- The **alpha wolf** (best solution) guides the pack toward better regions of the search space over several iterations
- After GWO converges, the best parameters are used for a full **50-epoch** final training run

**Search Space:**

| Hyperparameter | Range |
|---|---|
| Learning Rate | 1e-4 → 1e-2 (log-uniform) |
| Weight Decay | 1e-6 → 1e-3 (log-uniform) |
| Dropout Rate | 0.2 → 0.5 |
| LSTM Hidden Units | 64 / 96 / 128 / 160 |
| CNN Filter Sets | 16 / 32 / 48 / 64 / 96 / 128 |
| Optimizer | Adam / AdamW / RMSprop |
| Batch Size | 16 / 32 / 64 |

---

## Audio Augmentation (5×)

Each original `.wav` file produces 5 samples total:

| # | Augmentation | Details |
|---|---|---|
| 1 | **Original** | Raw audio, no modification |
| 2 | **Gaussian Noise** | σ = 0.005 additive white noise |
| 3 | **Time Shift** | Random ±25% circular shift |
| 4 | **Pitch Shift** | ±2 semitones via librosa |
| 5 | **Time Stretch** | 0.8× or 1.2× playback rate |

All augmented signals are converted to **128×128 log-Mel spectrograms** entirely in RAM to avoid I/O bottlenecks with mounted drives.

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 male, 12 female)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Filename format: `03-01-**05**-02-02-01-12.wav` → emotion ID in position 3 (1-indexed)
- Download: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

---

## Requirements

```
Python >= 3.8
torch >= 1.12
torchaudio
librosa
numpy
scikit-learn
scikit-image
matplotlib
```

Install dependencies:

```bash
pip install torch torchaudio librosa numpy scikit-learn scikit-image matplotlib
```

---

## Usage

### 1. Set the dataset path

Edit this line in the script to point to your downloaded RAVDESS folder:

```python
audio_dir = "/content/drive/MyDrive/audio_speech_actors_01-24"
```

### 2. Run the pipeline

```bash
python ser_gwo_cnn_blstm.py
```

The pipeline will:

1. Walk the dataset directory and load all `.wav` files
2. Apply 4× augmentation per file (producing 5× total samples)
3. Convert all audio to 128×128 log-Mel spectrograms
4. Split data into 80% train / 20% test (stratified)
5. Run GWO hyperparameter search (4 wolves × 3 iterations × 3 epochs each)
6. Train the final model for 50 epochs using the best discovered parameters
7. Plot train vs. test accuracy curves

### 3. Expected output

```
Using device: cuda
Building 5x augmented dataset from audio...
Total samples after 5x augmentation: 7200
Specs shape: (7200, 1, 128, 128)  Labels shape: (7200,)
Train: (5760, 1, 128, 128)  Test: (1440, 1, 128, 128)
Num classes: 8

GWO Iter 1/3 | Best Acc so far: 61.32%
GWO Iter 2/3 | Best Acc so far: 65.78%
GWO Iter 3/3 | Best Acc so far: 67.15%

Best Params Found by GWO: {'lr': 0.00082, 'weight_decay': 3.1e-05, ...}

Epoch 1/50  | Train Acc: 34.21% | Test Acc: 31.87%
...
Epoch 50/50 | Train Acc: 91.44% | Test Acc: 74.58%
```

---

## Project Structure

```
├── ser_gwo_cnn_blstm.py      # Main script (single file)
├── README.md
└── audio_speech_actors_01-24/
    ├── Actor_01/
    │   └── *.wav
    ├── Actor_02/
    │   └── *.wav
    └── ...
```

---

## Results

Training is plotted automatically at the end of the run:

- X-axis: Epoch (1–50)
- Y-axis: Accuracy (%)
- Two curves: Train Accuracy and Test Accuracy

The GWO pre-search ensures the 50-epoch run starts with well-tuned parameters rather than arbitrary defaults, reducing wasted training time and improving final test accuracy.

---

## Configuration

Key constants at the top of the script:

```python
seed         = 42       # Global random seed for reproducibility
sample_rate  = 16000    # Audio resampling rate (Hz)
resize_dim   = 128      # Mel-spectrogram output size (128×128)
epochs       = 50       # Final training epochs
n_wolves     = 4        # GWO population size
iterations   = 3        # GWO search iterations
epochs_per_wolf = 3     # Quick evaluation epochs per wolf candidate
```

---

## How GWO Works Here (Brief)

```
Initialize N wolves with random hyperparameter sets
For each GWO iteration:
    Evaluate fitness (val accuracy) of each wolf
    Rank wolves: α (best) > β > δ > ω
    Update each wolf's position toward α using:
        A = 2·a·r₁ − a          (convergence factor)
        D = |C·Xα − Xᵢ|        (distance to alpha)
        Xᵢ ← Xα − A·D          (position update)
    Clamp updated values to valid ranges
    Keep top-N wolves across old and new populations
Return α wolf's parameters → use for final training
```

The parameter `a` decreases linearly from 2 → 0 across iterations, shifting wolves from exploration to exploitation.

---

## Limitations & Future Work

- GWO search is lightweight (3 wolves × 3 iters × 3 epochs) for speed — increasing these will yield better hyperparameters at the cost of more compute
- Augmentation is applied once at data-loading time (static augmentation); online augmentation per epoch could further improve robustness
- The model currently ignores speaker identity — speaker-independent cross-validation would give a more realistic performance estimate
- Possible extensions: add MFCC or delta features alongside Mel spectrograms, try Transformer encoder in place of BLSTM, or experiment with contrastive pre-training

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Citation

If you use this work, please cite the RAVDESS dataset:

> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE* 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

---

## Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976) — S. Livingstone & F. Russo
- [librosa](https://librosa.org/) — audio feature extraction
- [PyTorch](https://pytorch.org/) — deep learning framework
- Grey Wolf Optimizer — Mirjalili et al. (2014), *Advances in Engineering Software*
