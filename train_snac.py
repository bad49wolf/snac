#!/usr/bin/env python3
"""
SNAC Training Script (Modular Version)

Simplified training script using the modular training components.
"""

import argparse
import json
from training import train_main, create_training_config
from training.config import DATASET_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Train SNAC model")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()),
                       help="Preset dataset configuration")
    parser.add_argument("--dataset_path", type=str, help="Override dataset path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--num_iterations", type=int, default=800000, help="Number of training iterations")
    parser.add_argument("--streaming", action="store_true", help="Use streaming for large datasets")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from JSON file
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    elif args.dataset:
        # Use preset dataset configuration
        preset = DATASET_CONFIGS[args.dataset]
        config = create_training_config(
            dataset_path=preset["dataset_path"],
            streaming=preset["streaming"]
        )
        if "split" in preset:
            config["data"]["split"] = preset["split"]
        print(f"Using preset configuration for: {args.dataset}")
    else:
        # Use default configuration
        config = create_training_config()
        print("Using default configuration")
    
    # Override with command line arguments
    if args.dataset_path:
        config["data"]["dataset_path"] = args.dataset_path
    if args.batch_size != 16:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate != 6e-4:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_iterations != 800000:
        config["training"]["num_iterations"] = args.num_iterations
    if args.streaming:
        config["data"]["streaming"] = True
    
    print("Training Configuration:")
    print(f"  Dataset: {config['data']['dataset_path']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Iterations: {config['training']['num_iterations']}")
    print(f"  Streaming: {config['data']['streaming']}")
    
    # Start training
    train_main(config)


if __name__ == "__main__":
    main()
