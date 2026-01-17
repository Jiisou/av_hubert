# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AV-HuBERT (Audio-Visual Hidden Unit BERT) is a self-supervised representation learning framework for audio-visual speech. It achieves state-of-the-art results in lip reading, ASR, and audio-visual speech recognition on the LRS3 benchmark.

## Installation

```bash
conda create -n avhubert python=3.8 -y
conda activate avhubert
git submodule init && git submodule update
pip install -r requirements.txt
cd fairseq && pip install --editable ./
```

## Common Commands

All training/inference commands must be run from the `avhubert/` directory.

### Pre-training
```bash
cd avhubert
fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

### Fine-tuning
```bash
cd avhubert
fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`
```

### Decoding (Inference)
```bash
cd avhubert
python -B infer_s2s.py --config-dir ./conf/ --config-name s2s_decode \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/results \
  override.modalities=['video'] common.user_dir=`pwd`
```

Modality options: `['video']` (lip reading), `['audio']` (ASR), `['audio','video']` (AV speech recognition)

### Load Pretrained Model (Python)
```python
import fairseq
import hubert_pretraining, hubert
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

## Architecture

### Core Components (`avhubert/`)

- **hubert.py**: Main `AVHubertModel` combining audio and visual modalities. Uses `ConvFeatureExtractionModel` for audio and `ResEncoder` for video (mouth ROI). Features are fused in a Transformer encoder.

- **hubert_pretraining.py**: `AVHubertPretrainingTask` for self-supervised training with masked cluster prediction. Uses frame-level pseudo-labels from k-means clustering.

- **hubert_asr.py**: `AVHubertAsrConfig` for Seq2Seq fine-tuning. Adds Transformer decoder for transcription tasks.

- **hubert_criterion.py**: Loss function with masked/unmasked frame prediction.

- **hubert_dataset.py**: Data loading with audio-visual alignment, augmentation (crop, flip), and noise mixing.

- **resnet.py**: `ResEncoder` - 2D ResNet for extracting visual features from mouth ROI.

- **sequence_generator.py**: Beam search decoder for inference.

### Data Pipeline

1. **preparation/**: Preprocessing scripts for LRS3/VoxCeleb2
   - Detect facial landmarks with dlib
   - Crop mouth ROI (`align_mouth.py`)
   - Count frames, create manifests

2. **clustering/**: Pseudo-label generation for self-supervised training
   - Extract MFCC or HuBERT features
   - Train k-means (default 1000 clusters)
   - Generate `.km` label files

### Data Formats

- **Manifest (.tsv)**: Tab-separated with root dir on line 1, then `id video_path audio_path video_frames audio_frames`
- **Labels (.km)**: Space-separated cluster IDs per frame
- **Transcripts (.wrd)**: Word-level text labels

### Configuration

Configs use Hydra/OmegaConf in `avhubert/conf/`:
- `pretrain/`: Pre-training configs (base/large, iteration 1-5)
- `finetune/`: Fine-tuning configs
- `s2s_decode.yaml`: Decoding hyperparameters (beam size, length penalty)

## Key Dependencies

- **fairseq**: Core training framework (installed as submodule in editable mode)
- **dlib**: Face detection/landmark prediction (for data preparation only)
- **opencv-python**: Video processing
- **sentencepiece**: BPE tokenization

## Notes

- Label rate must match feature frame rate (100Hz for MFCC, 25Hz for AV-HuBERT features)
- Pre-training uses iterative refinement: iteration 1 uses MFCC labels, subsequent iterations use HuBERT features
- No test suite exists; evaluation is done via WER/CER during decoding
