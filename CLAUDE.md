# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSBrain is a Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding (NeurIPS 2025 Spotlight). It's a PyTorch-based framework that pretrains on unlabeled EEG data via masked autoencoding, then fine-tunes on 16 downstream EEG tasks (emotion recognition, motor imagery, sleep staging, seizure detection, regression).

## Commands

### Pretraining
```bash
bash sh/pretrain_CSBrain.sh
```

### Fine-tuning (16 dataset scripts in `sh/`)
```bash
bash sh/finetune_CSBrain_<DATASET>.sh
# e.g., bash sh/finetune_CSBrain_BCIC.sh
```

### Running directly
```bash
# Pretrain
python pretrain_main.py --dataset_dir <path> --model_dir <path> [options]

# Fine-tune
python finetune_main.py --downstream_dataset FACED --datasets_dir <path> --model_dir <path> --use_pretrained_weights --foundation_dir pth/CSBrain.pth [options]
```

### Environment setup
Follow [CBraMod](https://github.com/wjq-learning/CBraMod) for environment installation. Key dependencies: torch, numpy, scipy, scikit-learn, lmdb, einops, tqdm, matplotlib, umap-learn.

## Architecture

### Core Model (`models/`)
- **CSBrain.py** — Main foundation model: PatchEmbedding (Conv2D + positional + spectral FFT features), Cross-scale Temporal Embedding (multi-scale convolutions with kernels 1/3/5), Brain Region Embedding, and CSBrain_TransformerEncoder
- **CSBrain_transformerlayer.py** — Custom transformer layer with inter-window temporal attention (window_size=5) and inter-region spatial attention with RegionAttentionMaskBuilder
- **CSBrain_transformer.py** — Supporting modules: TemEmbedEEGLayer, BrainEmbedEEGLayer, LayerNorm, FeedForward
- **Task-specific model files** (e.g., `model_for_faced.py`) — Wrap CSBrain backbone with classification/regression heads

### Brain Region Mapping (5 regions, 19 electrodes)
- Region 0 (Frontal): FP1, FP2, F3, F4, F7, F8, FZ
- Region 1 (Parietal): P3, P4, PZ
- Region 2 (Temporal): T3, T4, T5, T6
- Region 3 (Occipital): O1, O2
- Region 4 (Central): C3, C4, CZ

### Training Pipeline
- **pretrain_main.py / pretrain_trainer.py** — MAE pretraining (50% mask ratio, MSELoss, AdamW lr=5e-4, 40 epochs, batch 128)
- **finetune_main.py / finetune_trainer.py** — Fine-tuning with multi-class (CrossEntropyLoss + label smoothing 0.1), binary (BCEWithLogitsLoss), or regression (MSELoss)
- **finetune_evaluator.py** — Metrics: balanced accuracy, kappa, F1 (multi-class); balanced accuracy, ROC-AUC, PR-AUC (binary); Pearson correlation, R², RMSE (regression)

### Data Layer (`datasets/`)
- LMDB-based dataset storage
- Input shape: (batch, channels, patches, patch_size) — e.g., (64, 19, 30, 200)
- Each downstream dataset has its own loader module
- `preprocessing/` has processing scripts for HMC, Siena, TUSL datasets

### Utilities (`utils/`)
- `signaltools.py` — FFT-based EEG signal resampling (PyTorch reimplementation of scipy.signal.resample)
- `masking.py` — Mask generation for MAE pretraining

## Key Model Parameters
- d_model=200, dim_feedforward=800, n_layer=12, nhead=8, seq_len=30
- Default fine-tune: lr=1e-4, weight_decay=0.05, epochs=50, batch_size=64, dropout=0.1

## Pretrained Weights
Download from Google Drive (link in README). Place pretrained weights at `pth/CSBrain.pth`, downstream weights in `pth_downtasks/`.
