#!/usr/bin/env python3
"""
SNAC Inference Script

Simple script to load trained SNAC model and perform inference on audio files.
"""

import argparse
import torch
import torchaudio
from pathlib import Path
import json

from snac import SNAC


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load SNAC model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model configuration
    config = checkpoint.get('config', {})
    if not config:
        # Try to load config from separate file
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise ValueError("No model configuration found")
    
    # Create and load model
    model = SNAC(**config).to(device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    
    return model


def encode_decode_audio(model, audio_path: str, output_path: str, device: str = "cuda"):
    """Encode and decode audio file."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    target_sr = model.sampling_rate
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Move to device and add batch dimension
    waveform = waveform.to(device).unsqueeze(0)  # [1, 1, T]
    
    print(f"Input shape: {waveform.shape}")
    print(f"Duration: {waveform.shape[-1] / target_sr:.2f}s")
    
    # Inference
    with torch.no_grad():
        # Encode to codes
        codes = model.encode(waveform)
        print(f"Encoded to {len(codes)} code levels:")
        for i, code in enumerate(codes):
            print(f"  Level {i}: {code.shape} ({code.shape[1]} tokens)")
        
        # Decode back to audio
        reconstructed = model.decode(codes)
        
        # Alternative: single forward pass
        # reconstructed, codes = model(waveform)
    
    # Save reconstructed audio
    reconstructed = reconstructed.squeeze(0).cpu()  # Remove batch dim
    torchaudio.save(output_path, reconstructed, target_sr)
    
    print(f"Saved reconstructed audio to: {output_path}")
    
    # Calculate compression ratio
    original_size = waveform.numel() * 4  # Assuming 32-bit float
    
    # Estimate compressed size (bits)
    total_bits = 0
    for code in codes:
        bits_per_token = 12  # log2(4096) codebook entries
        total_bits += code.numel() * bits_per_token
    
    compressed_size = total_bits / 8  # Convert to bytes
    compression_ratio = original_size / compressed_size
    bitrate = total_bits / (waveform.shape[-1] / target_sr) / 1000  # kbps
    
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Bitrate: {bitrate:.2f} kbps")


def main():
    parser = argparse.ArgumentParser(description="SNAC Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    
    # Process audio
    print(f"Processing {args.input}...")
    encode_decode_audio(model, args.input, args.output, args.device)
    
    print("Done!")


if __name__ == "__main__":
    main()
