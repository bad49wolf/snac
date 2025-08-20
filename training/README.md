# SNAC Training Module

This modular training system provides organized components for training SNAC models.

## Structure

```
training/
├── __init__.py          # Package initialization and exports
├── dataset.py           # AudioDataset class for HF datasets
├── discriminators.py    # Multi-Period and Multi-Scale STFT discriminators
├── losses.py           # All loss functions (adversarial, reconstruction)
├── config.py           # Model configurations and presets
├── train.py            # Main training loop and utilities
└── README.md           # This file
```

## Components

### 1. Dataset (`dataset.py`)
- **AudioDataset**: Loads audio from Hugging Face datasets
- Supports both local folders and HF Hub datasets
- Automatic resampling, mono conversion, normalization
- Streaming support for large datasets

### 2. Discriminators (`discriminators.py`)
- **MultiPeriodDiscriminator**: HiFi-GAN style MPD with periods [2,3,5,7,11]
- **MultiScaleSTFTDiscriminator**: STFT-based discriminator with 3 scales
- **PeriodDiscriminator**: Single period discriminator component
- **STFTDiscriminator**: Single STFT discriminator component

### 3. Loss Functions (`losses.py`)
- **feature_matching_loss**: L1 loss between discriminator features
- **discriminator_loss**: MSE loss for real/fake classification
- **generator_loss**: MSE adversarial loss for generator
- **mel_spectrogram_loss**: L1 loss in mel domain (80 mels)
- **stft_loss**: Multi-resolution STFT reconstruction loss

### 4. Configuration (`config.py`)
- **create_snac_24khz_config**: Speech model configuration
- **create_snac_32khz_config**: Music model configuration (32kHz)
- **create_snac_44khz_config**: High-quality music model (44kHz)
- **create_training_config**: Complete training configuration
- **DATASET_CONFIGS**: Preset configurations for common datasets

### 5. Training Loop (`train.py`)
- **main**: Complete training function
- **setup_models**: Initialize generator and discriminators
- **setup_optimizers**: Create AdamW optimizers and schedulers
- **setup_dataset**: Create dataset and dataloader
- **train_step**: Single training iteration
- **save_checkpoint**: Checkpoint saving utility

## Usage

### Option 1: Use the simplified training script
```bash
# Default configuration
python train_snac.py

# Use preset dataset
python train_snac.py --dataset ljspeech

# Custom configuration
python train_snac.py --dataset_path "your-dataset" --batch_size 8 --streaming

# Load from JSON config
python train_snac.py --config my_config.json
```

### Option 2: Use the training module directly
```python
from training import train_main, create_training_config

# Create configuration
config = create_training_config(
    dataset_path="bad49wolf/ljspeech-simple",
    batch_size=16,
    learning_rate=6e-4,
    streaming=False
)

# Start training
train_main(config)
```

### Option 3: Import individual components
```python
from training import AudioDataset, MultiPeriodDiscriminator
from training.losses import mel_spectrogram_loss
from training.config import create_snac_24khz_config

# Use components separately for custom training loops
dataset = AudioDataset("path/to/dataset")
discriminator = MultiPeriodDiscriminator()
config = create_snac_24khz_config()
```

## Configuration Examples

### Speech Training (24kHz)
```python
config = create_training_config(
    dataset_path="bad49wolf/ljspeech-simple",
    batch_size=16,
    streaming=False
)
```

### Large Dataset Training
```python
config = create_training_config(
    dataset_path="mozilla-foundation/common_voice_16_1",
    batch_size=8,
    streaming=True  # Important for large datasets
)
```

### Custom Model Configuration
```python
from training.config import create_snac_24khz_config

# Get base config and modify
model_config = create_snac_24khz_config()
model_config["codebook_size"] = 8192  # Larger codebook
model_config["encoder_dim"] = 128     # Larger model

config = {
    "model": model_config,
    "training": {...},
    "data": {...}
}
```

## Available Presets

The following dataset presets are available:

- **ljspeech**: LJSpeech dataset (clean single speaker)
- **common_voice**: Mozilla Common Voice (multilingual, streaming)
- **librispeech**: LibriSpeech clean-100 (streaming)
- **vctk**: VCTK corpus (multi-speaker)

## Benefits of Modular Structure

1. **Maintainability**: Each component is isolated and focused
2. **Reusability**: Components can be imported and used independently
3. **Testing**: Individual components can be unit tested
4. **Extensibility**: Easy to add new discriminators, losses, or datasets
5. **Debugging**: Easier to isolate issues in specific components
6. **Code Organization**: Clear separation of concerns

## Migration from Original Script

The original monolithic `train_snac_24khz.py` has been split into:
- Dataset logic → `training/dataset.py`
- Discriminators → `training/discriminators.py`
- Loss functions → `training/losses.py`
- Configuration → `training/config.py`
- Training loop → `training/train.py`

All functionality is preserved while improving code organization and reusability.
