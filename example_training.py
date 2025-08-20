#!/usr/bin/env python3
"""
SNAC Training Examples

Examples showing different ways to use the modular SNAC training system.
"""

import json
from training import train_main, create_training_config, AudioDataset
from training.config import DATASET_CONFIGS, create_snac_24khz_config


def example_1_basic_training():
    """Example 1: Basic training with default settings"""
    print("Example 1: Basic training with LJSpeech dataset")
    
    config = create_training_config(
        dataset_path="bad49wolf/ljspeech-simple",
        batch_size=16,
        streaming=False
    )
    
    # Reduce iterations for quick testing
    config["training"]["num_iterations"] = 1000
    config["training"]["save_interval"] = 500
    
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Iterations: {config['training']['num_iterations']}")
    
    # Uncomment to start training
    # train_main(config)


def example_2_custom_config():
    """Example 2: Custom model configuration"""
    print("\nExample 2: Custom model configuration")
    
    # Start with base config and modify
    model_config = create_snac_24khz_config()
    model_config["codebook_size"] = 8192  # Larger codebook
    model_config["encoder_dim"] = 128     # Larger encoder
    
    config = {
        "model": model_config,
        "training": {
            "batch_size": 8,  # Smaller batch for larger model
            "learning_rate": 3e-4,  # Lower learning rate
            "lr_decay": 0.999994,
            "num_iterations": 1000,
            "save_interval": 500,
            "log_interval": 50,
            "validation_interval": 200,
        },
        "data": {
            "dataset_path": "bad49wolf/ljspeech-simple",
            "audio_column": "audio",
            "split": "train",
            "streaming": False,
            "segment_length": 0.8,
        },
        "paths": {
            "checkpoint_dir": "./checkpoints_custom",
            "log_dir": "./logs_custom",
        }
    }
    
    print(f"Model codebook size: {model_config['codebook_size']}")
    print(f"Model encoder dim: {model_config['encoder_dim']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Uncomment to start training
    # train_main(config)


def example_3_streaming_dataset():
    """Example 3: Large dataset with streaming"""
    print("\nExample 3: Streaming dataset training")
    
    config = create_training_config(
        dataset_path="mozilla-foundation/common_voice_16_1",
        batch_size=8,  # Smaller batch for streaming
        streaming=True,  # Enable streaming
        num_iterations=1000
    )
    
    # Streaming datasets need special handling
    config["training"]["save_interval"] = 200
    config["data"]["split"] = "train"
    
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Streaming: {config['data']['streaming']}")
    print(f"Split: {config['data']['split']}")
    
    # Uncomment to start training
    # train_main(config)


def example_4_config_file():
    """Example 4: Save and load configuration from JSON"""
    print("\nExample 4: Configuration file usage")
    
    config = create_training_config(
        dataset_path="vctk",
        batch_size=12,
        learning_rate=4e-4,
        num_iterations=1000
    )
    
    # Save config to file
    config_file = "example_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    
    # Load config from file
    with open(config_file, 'r') as f:
        loaded_config = json.load(f)
    
    print(f"Loaded dataset: {loaded_config['data']['dataset_path']}")
    print(f"Loaded batch size: {loaded_config['training']['batch_size']}")
    
    # Use with training script:
    # python train_snac.py --config example_config.json


def example_5_dataset_exploration():
    """Example 5: Explore dataset before training"""
    print("\nExample 5: Dataset exploration")
    
    # Create dataset without training
    dataset = AudioDataset(
        dataset_path="bad49wolf/ljspeech-simple",
        sample_rate=24000,
        segment_length=0.8,
        streaming=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample duration: {sample.shape[0] / 24000:.2f}s")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")


def example_6_preset_datasets():
    """Example 6: Using preset dataset configurations"""
    print("\nExample 6: Preset dataset configurations")
    
    print("Available presets:")
    for name, config in DATASET_CONFIGS.items():
        print(f"  {name}: {config['dataset_path']}")
        print(f"    Streaming: {config['streaming']}")
        if 'split' in config:
            print(f"    Split: {config['split']}")
        print()
    
    # Use preset configuration
    preset_name = "ljspeech"
    preset = DATASET_CONFIGS[preset_name]
    
    config = create_training_config(
        dataset_path=preset["dataset_path"],
        streaming=preset["streaming"],
        num_iterations=1000
    )
    
    print(f"Using preset '{preset_name}' configuration")
    print(f"Dataset: {config['data']['dataset_path']}")


if __name__ == "__main__":
    print("SNAC Modular Training Examples")
    print("=" * 40)
    
    # Run all examples (training calls are commented out)
    example_1_basic_training()
    example_2_custom_config()
    example_3_streaming_dataset()
    example_4_config_file()
    example_5_dataset_exploration()
    example_6_preset_datasets()
    
    print("\n" + "=" * 40)
    print("To run actual training, uncomment the train_main() calls")
    print("or use the command-line interface:")
    print("  python train_snac.py --dataset ljspeech")
    print("  python train_snac.py --config example_config.json")
