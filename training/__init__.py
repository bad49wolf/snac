"""
SNAC Training Package

This package contains all the components needed for training SNAC models:
- Dataset handling for Hugging Face datasets
- Discriminator models (Multi-Period and Multi-Scale STFT)
- Loss functions (adversarial, feature matching, reconstruction)
- Training configuration and main training loop
"""

from .dataset import AudioDataset
from .discriminators import MultiPeriodDiscriminator, MultiScaleSTFTDiscriminator
from .losses import (
    feature_matching_loss,
    discriminator_loss, 
    generator_loss,
    mel_spectrogram_loss,
    stft_loss
)
from .config import create_snac_24khz_config, create_training_config
from .train import main as train_main

__version__ = "1.0.0"

__all__ = [
    "AudioDataset",
    "MultiPeriodDiscriminator", 
    "MultiScaleSTFTDiscriminator",
    "feature_matching_loss",
    "discriminator_loss",
    "generator_loss", 
    "mel_spectrogram_loss",
    "stft_loss",
    "create_snac_24khz_config",
    "create_training_config",
    "train_main"
]
