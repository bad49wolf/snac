# SNAC 24kHz Speech Training Guide

This guide explains how to train the SNAC 24kHz speech model from scratch using the provided training script.

## Overview

The training implementation follows the paper "SNAC: Multi-Scale Neural Audio Codec" and includes:

- **SNAC Generator**: 24kHz speech-optimized configuration
- **Multi-Period Discriminator (MPD)**: From HiFi-GAN
- **Multi-Scale STFT Discriminator (MSD)**: Complex spectrogram discriminator
- **Loss Functions**: Adversarial + Feature Matching + Mel + Multi-Resolution STFT

## Model Configuration (24kHz Speech)

Based on the paper specifications:

```python
{
    "sampling_rate": 24000,
    "encoder_rates": [2, 4, 8, 8],      # Downsampling rates
    "decoder_rates": [8, 8, 4, 2],      # Upsampling rates (mirror)
    "vq_strides": [4, 2, 1],            # 3 VQ levels (vs 4 for music)
    "codebook_size": 4096,              # 12-bit codes
    "attn_window_size": None,           # No attention for speech
    "depthwise": True,                   # Depthwise convolutions
    "noise": True                        # Noise blocks in decoder
}
```

**Resulting Model:**
- Token rates: ~12, 23, 47 Hz
- Bitrate: ~0.98 kbps
- Parameters: ~19.8M (6.7M encoder + 13.0M decoder)

## Training Hyperparameters

Following the paper:

```python
{
    "batch_size": 16,
    "learning_rate": 6e-4,              # Higher LR enabled by depthwise conv
    "lr_decay": 0.999994,               # Per-iteration decay
    "num_iterations": 800000,
    "optimizer": "AdamW",
    "betas": [0.8, 0.99],
    "weight_decay": 1e-2
}
```

## Dataset Requirements

The training script now uses Hugging Face datasets with audio fields. You have several options:

### Option 1: Local Audio Folder
```
/path/to/speech/dataset/
├── speaker1/
│   ├── utterance1.wav
│   ├── utterance2.wav
│   └── ...
├── speaker2/
│   └── ...
└── ...
```

### Option 2: Hugging Face Dataset
Use any HF dataset with an "audio" column:
- `mozilla-foundation/common_voice_16_1`
- `librispeech_asr` 
- `vctk`
- Your custom uploaded dataset

### Option 3: Custom Dataset Format
Upload your dataset to Hugging Face Hub with audio files in "audio" column.

**Supported formats:** .wav, .flac, .mp3  
**Segment length:** 0.8 seconds (19,200 samples at 24kHz)  
**Preprocessing:** Automatic resampling, mono conversion, normalization

**Streaming vs Non-Streaming:**
- **Non-streaming** (`streaming: False`): Loads entire dataset into memory. Better for shuffling and random access. Good for datasets < 10GB.
- **Streaming** (`streaming: True`): Loads data on-demand. Essential for large datasets > 10GB. No random shuffling but more memory efficient.

## Usage

### 1. Setup Environment

```bash
pip install -r requirements.txt
pip install -r requirements_training.txt
```

### 2. Configure Dataset

Update the dataset configuration in the training script:

```python
"data": {
    "dataset_path": "/hugginfacerepodataset/dataset",  # Or local path
    "audio_column": "audio",       # Column name with audio data
    "split": "train",              # Dataset split to use
    "streaming": False,            # True for large datasets
    "segment_length": 0.8,
}
```

**Examples:**
```python
# Local folder
"dataset_path": "/path/to/local/audio/folder"

# HF Hub dataset
"dataset_path": "mozilla-foundation/common_voice_16_1"

# Custom dataset
"dataset_path": "your-username/your-speech-dataset"
```

### 3. Start Training

```bash
python train_snac_24khz.py
```

The script will:
- Create `./checkpoints` and `./logs` directories
- Save checkpoints every 10,000 iterations
- Log progress every 100 iterations
- Train for 800,000 iterations (can be adjusted)

### 4. Monitor Training

Key metrics to watch:
- **G_loss**: Generator loss (should decrease)
- **D_loss**: Discriminator loss (should stabilize)
- **Mel**: Mel-spectrogram reconstruction loss
- **STFT**: Multi-resolution STFT loss
- **LR**: Learning rate (should decay gradually)

### 5. Inference

Test your trained model:

```bash
python inference.py \
    --checkpoint ./checkpoints/snac_24khz_final.pt \
    --input test_audio.wav \
    --output reconstructed.wav
```

## Loss Function Details

The training uses multiple loss components:

1. **Adversarial Loss**: Generator vs discriminators (MPD + MSD)
2. **Feature Matching**: L1 loss between discriminator features
3. **Mel-Spectrogram**: L1 loss in mel domain (80 mels, 24kHz)
4. **Multi-Resolution STFT**: L1 loss in STFT domain (3 resolutions)

**Loss weights:**
- Adversarial: 1.0
- Feature Matching: 2.0  
- Mel: 45.0
- STFT: 45.0

## Key Implementation Details

### Multi-Scale Vector Quantization
```python
vq_strides = [4, 2, 1]  # Temporal strides for each VQ level
# Level 0: Every 4th frame (coarse, ~6 Hz)
# Level 1: Every 2nd frame (medium, ~12 Hz) 
# Level 2: Every frame (fine, ~24 Hz)
```

### Depthwise Convolutions
- Reduces parameters and improves training stability
- Applied everywhere except embedding/output layers
- Enables higher learning rates (6e-4 vs typical 1e-4)

### Noise Blocks
- Add input-dependent stochastic noise: `x + Linear(x) ⊙ ε`
- Improves reconstruction quality and codebook utilization
- Applied after each decoder upsampling layer

## Expected Results

After training, the model should achieve:
- **MUSHRA Score**: ~88 (vs reference at 99.5)
- **Compression**: ~49x ratio at 0.98 kbps
- **Quality**: Near-transparent speech reconstruction
- **Efficiency**: Outperforms other codecs at same bitrate

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (RTX 3070/4060 Ti)
- RAM: 16GB
- Storage: 100GB for checkpoints

**Recommended:**  
- GPU: 16GB+ VRAM (RTX 4080/A100)
- RAM: 32GB
- Storage: 500GB SSD

Training time: ~3-5 days on single RTX 4090.

## Troubleshooting

**Common Issues:**

1. **OOM Error**: Reduce batch_size from 16 to 8 or 4
2. **NaN Loss**: Check audio normalization, reduce learning rate
3. **Slow Training**: Ensure data on fast SSD, increase num_workers
4. **Poor Quality**: Verify dataset quality, check loss weights

**Training Tips:**

- Monitor discriminator/generator balance (D_loss should not dominate)
- Save intermediate checkpoints for evaluation
- Use mixed precision (fp16) for memory efficiency
- Consider gradient accumulation for larger effective batch sizes

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{siuzdak2024snac,
  title={SNAC: Multi-Scale Neural Audio Codec},
  author={Siuzdak, Hubert and Grötschla, Florian and Lanzendörfer, Luca A},
  booktitle={Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation},
  year={2024}
}
```
