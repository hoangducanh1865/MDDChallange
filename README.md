# MDD Challenge 2025 — Vietnamese Mispronunciation Detection and Diagnosis

## Problem Description

Given a student's speech recording and the canonical (reference) phoneme sequence of the target utterance, predict the actual phoneme sequence produced by the student. Mispronounced phonemes are detected and diagnosed by comparing the predicted output against the canonical reference.

**Dataset**: 3,180 Vietnamese speech samples with phoneme-level transcriptions and tone markers (e.g., `aː-0`, `ɓ`, `t͡ɕ`, `ŋmz`).

## Challenge Metric

Leaderboard ranking is based on:

```
Score = 0.5 × F1 + 0.4 × (1 − DER) + 0.1 × (1 − PER)
```

| Metric | Weight | Description |
|--------|--------|-------------|
| F1     | 0.5    | Phoneme-level mispronunciation detection F1 |
| DER    | 0.4    | Diagnosis Error Rate (wrong error type assigned) |
| PER    | 0.1    | Phoneme Error Rate (overall recognition accuracy) |

## Architecture

- **Ensemble** of 3 independently fine-tuned backbones: `wav2vec2-base-100h`, `wav2vec2-base-vietnamese-250h`, `hubert-base-ls960`
- **Multi-task heads**: CTC (primary) + binary error detection (auxiliary)
- **Focal CTC Loss** (γ=2) to handle phoneme imbalance
- **SpecAugment** + **audiomentations** for data augmentation
- **LLRD** (Layer-wise Learning Rate Decay) for stable fine-tuning
- **Temperature Scaling** calibration before ensemble
- **Optuna** ensemble weight search (100 trials)
- **5-Fold Cross Validation** for robust evaluation

## Installation

**Step 1 — Conda environment (recommended):**
```bash
conda env create -f environment.yml
conda activate MDDChallange
```

Or with pip:
```bash
pip install torch torchaudio transformers librosa audiomentations
pip install scikit-learn pandas tqdm optuna pyctcdecode jiwer matplotlib soundfile sentencepiece
```

**Step 2 — Download HuBERT model weights (large files, not in repo):**

Download the two files below from Google Drive and place them in `models/hubert-base-ls960/`:

```
https://drive.google.com/drive/folders/1yLVFtPcz33Qp8bizN45cYHzy5ZZXekj1?usp=sharing
```

Files to download:
- `pytorch_model.bin` → `models/hubert-base-ls960/pytorch_model.bin`
- `tf_model.h5`       → `models/hubert-base-ls960/tf_model.h5`

After downloading, the directory should look like:
```
models/
└── hubert-base-ls960/
    ├── config.json              ✓ already in repo
    ├── preprocessor_config.json ✓ already in repo
    ├── pytorch_model.bin        ← download from Drive (360 MB)
    └── tf_model.h5              ← download from Drive (360 MB)
```

> Only `pytorch_model.bin` is required for training; `tf_model.h5` can be skipped.

## Usage

```bash
# Cross-validation (eval mode) — trains all 3 models across 5 folds
python main.py --mode eval \
               --data_dir ./data/MDD-Challenge-2025-training-set \
               --n_folds 5 --epochs 30 --batch_size 16 --lr 1e-4

# Inference on unlabelled test set → produces predictions.csv
python main.py --mode test \
               --data_dir ./data/MDD-Challenge-2025-test-set \
               --checkpoint_dir ./outputs/checkpoints
```

## Project Structure

```
.
├── main.py                        # Entry point (eval / test modes)
├── src/
│   ├── dataset.py                 # MDDDataset, DataLoader, augmentation, K-fold splits
│   ├── model.py                   # MDDModel (3 backbones), EnsembleModel
│   ├── train.py                   # 5-fold CV, FocalCTCLoss, LLRD, Score-based checkpointing
│   ├── inference.py               # Temperature scaling, Optuna ensemble, predictions.csv
│   ├── utils.py                   # Vocab building, greedy decode, Score computation, plots
│   └── evaluation/
│       └── evaluate.py            # F1 / PER / DER metric implementation
├── data/
│   └── MDD-Challenge-2025-training-set/
│       ├── audio_data/train/      # 3,180 WAV files (16 kHz)
│       └── metadata/
│           ├── train_phones.csv   # IPA phoneme sequences (canonical + transcript)
│           └── lexicon_vmd.txt    # Vietnamese phoneme dictionary
├── outputs/
│   ├── vocab.json                 # Auto-built Vietnamese phoneme vocabulary
│   └── checkpoints/               # best_fold{k}_{model_name}.pt (15 files for 5-fold × 3 models)
├── experiment/                    # Training curves and CV summary plots
└── predictions.csv                # Final output (id, predict) — only in test mode
```

## Notes

- Vocab is built automatically from `train_phones.csv` on first run and cached at `outputs/vocab.json`.
- Checkpoints are saved per fold and per model, named `best_fold{k}_{model_name}.pt`.
- All plots are saved to `experiment/` during cross-validation.
- For Colab with T4 GPU, use `--batch_size 8` and one model at a time if memory is tight.
