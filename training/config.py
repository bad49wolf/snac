#!/usr/bin/env python3
"""
SNAC Training Configuration

This module contains configuration functions and default settings for SNAC training.
"""


def create_snac_24khz_config():
    """Create SNAC configuration for 24kHz speech model based on paper."""
    return {
        "sampling_rate": 24000,
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 8, 8],  # Speech-specific rates
        "latent_dim": None,
        "decoder_dim": 1536,
        "decoder_rates": [8, 8, 4, 2],  # Mirror of encoder
        "attn_window_size": None,  # No attention for speech model
        "codebook_size": 4096,
        "codebook_dim": 8,
        "vq_strides": [4, 2, 1],  # 3 levels instead of 4 for speech
        "noise": True,
        "depthwise": True,
    }


def create_training_config(
    dataset_path: str = "bad49wolf/ljspeech-simple",
    batch_size: int = 16,
    learning_rate: float = 6e-4,
    num_iterations: int = 800000,
    streaming: bool = False,
):
    """Create a complete training configuration."""
    return {
        "model": create_snac_24khz_config(),
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_decay": 0.999994,
            "num_iterations": num_iterations,
            "save_interval": 10000,
            "log_interval": 100,
            "validation_interval": 1000,
        },
        "data": {
            "dataset_path": dataset_path,  # HF dataset path or local path
            "audio_column": "audio",  # Column name containing audio data
            "split": "train",  # Dataset split to use
            "streaming": streaming,  # Set to True for large datasets
            "segment_length": 0.8,  # seconds
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        }
    }


def create_snac_32khz_config():
    """Create SNAC configuration for 32kHz music model."""
    return {
        "sampling_rate": 32000,
        "encoder_dim": 64,
        "encoder_rates": [3, 3, 7, 7],  # Music-specific rates
        "latent_dim": None,
        "decoder_dim": 1536,
        "decoder_rates": [7, 7, 3, 3],  # Mirror of encoder
        "attn_window_size": 32,  # With attention for music
        "codebook_size": 4096,
        "codebook_dim": 8,
        "vq_strides": [8, 4, 2, 1],  # 4 levels for music
        "noise": True,
        "depthwise": True,
    }


def create_snac_44khz_config():
    """Create SNAC configuration for 44kHz high-quality music model."""
    return {
        "sampling_rate": 44100,
        "encoder_dim": 64,
        "encoder_rates": [3, 3, 7, 7],  # Music-specific rates
        "latent_dim": None,
        "decoder_dim": 1536,
        "decoder_rates": [7, 7, 3, 3],  # Mirror of encoder
        "attn_window_size": 32,  # With attention for music
        "codebook_size": 4096,
        "codebook_dim": 8,
        "vq_strides": [8, 4, 2, 1],  # 4 levels for music
        "noise": True,
        "depthwise": True,
    }


# Preset configurations for common datasets
DATASET_CONFIGS = {
    "ljspeech": {
        "dataset_path": "bad49wolf/ljspeech-simple",
        "streaming": False,
        "model_config": "24khz",
    },
    "common_voice": {
        "dataset_path": "mozilla-foundation/common_voice_16_1",
        "streaming": True,
        "model_config": "24khz",
    },
    "librispeech": {
        "dataset_path": "librispeech_asr",
        "streaming": True,
        "model_config": "24khz",
        "split": "train.clean.100",
    },
    "vctk": {
        "dataset_path": "vctk",
        "streaming": False,
        "model_config": "24khz",
    },
}
