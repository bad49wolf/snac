#!/usr/bin/env python3
"""
SNAC Training Quick Start Examples

This script shows how to quickly modify the training configuration
for different dataset types.
"""

# Import the main training function
import json

def create_config_for_common_voice():
    """Configuration for Mozilla Common Voice dataset (streaming)"""
    return {
        "model": {
            "sampling_rate": 24000,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "latent_dim": None,
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "attn_window_size": None,
            "codebook_size": 4096,
            "codebook_dim": 8,
            "vq_strides": [4, 2, 1],
            "noise": True,
            "depthwise": True,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 6e-4,
            "lr_decay": 0.999994,
            "num_iterations": 800000,
            "save_interval": 10000,
            "log_interval": 100,
            "validation_interval": 1000,
        },
        "data": {
            "dataset_path": "mozilla-foundation/common_voice_16_1",
            "audio_column": "audio",
            "split": "train",
            "streaming": True,  # Large dataset - use streaming
            "segment_length": 0.8,
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        }
    }

def create_config_for_librispeech():
    """Configuration for LibriSpeech dataset"""
    return {
        "model": {
            "sampling_rate": 24000,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "latent_dim": None,
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "attn_window_size": None,
            "codebook_size": 4096,
            "codebook_dim": 8,
            "vq_strides": [4, 2, 1],
            "noise": True,
            "depthwise": True,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 6e-4,
            "lr_decay": 0.999994,
            "num_iterations": 800000,
            "save_interval": 10000,
            "log_interval": 100,
            "validation_interval": 1000,
        },
        "data": {
            "dataset_path": "librispeech_asr",
            "audio_column": "audio",
            "split": "train.clean.100",  # Clean 100-hour subset
            "streaming": True,
            "segment_length": 0.8,
        },
        "paths": {
            "checkpoint_dir": "./checkpoints_librispeech",
            "log_dir": "./logs_librispeech",
        }
    }

def create_config_for_local_folder():
    """Configuration for local audio folder"""
    return {
        "model": {
            "sampling_rate": 24000,
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "latent_dim": None,
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "attn_window_size": None,
            "codebook_size": 4096,
            "codebook_dim": 8,
            "vq_strides": [4, 2, 1],
            "noise": True,
            "depthwise": True,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 6e-4,
            "lr_decay": 0.999994,
            "num_iterations": 800000,
            "save_interval": 10000,
            "log_interval": 100,
            "validation_interval": 1000,
        },
        "data": {
            "dataset_path": "/path/to/your/audio/folder",  # UPDATE THIS PATH
            "audio_column": "audio",
            "split": "train",
            "streaming": False,  # Local folder - can load in memory
            "segment_length": 0.8,
        },
        "paths": {
            "checkpoint_dir": "./checkpoints_local",
            "log_dir": "./logs_local",
        }
    }

def save_config(config, filename):
    """Save configuration to JSON file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {filename}")

if __name__ == "__main__":
    print("SNAC Training Configuration Examples")
    print("=" * 40)
    
    # Generate example configurations
    configs = {
        "common_voice": create_config_for_common_voice(),
        "librispeech": create_config_for_librispeech(), 
        "local_folder": create_config_for_local_folder(),
    }
    
    # Save configurations
    for name, config in configs.items():
        filename = f"config_{name}.json"
        save_config(config, filename)
    
    print("\nTo use a configuration:")
    print("1. Edit the config JSON file with your specific paths")
    print("2. Load it in your training script:")
    print("   with open('config_common_voice.json') as f:")
    print("       config = json.load(f)")
    print("3. Or modify train_snac_24khz.py directly")
    
    print(f"\nDataset paths to update:")
    print(f"- Common Voice: Already configured")
    print(f"- LibriSpeech: Already configured")
    print(f"- Local folder: Update path in config_local_folder.json")
