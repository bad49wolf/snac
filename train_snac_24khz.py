#!/usr/bin/env python3
"""
SNAC 24kHz Speech Model Training Script

Implementation based on the paper:
"SNAC: Multi-Scale Neural Audio Codec" by Siuzdak et al.

This script implements training for the 24kHz speech-optimized SNAC model
with multi-period and multi-scale STFT discriminators.
"""

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset

from snac import SNAC


class AudioDataset(Dataset):
    """Dataset for loading speech audio from Hugging Face datasets with 'audio' field."""
    
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 24000,
        segment_length: float = 0.8,  # seconds
        split: str = "train",
        streaming: bool = False,
        audio_column: str = "audio",
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.audio_column = audio_column
        
        print(f"Loading dataset from: {dataset_path}")
        
        # Load Hugging Face dataset
        if dataset_path.startswith('/') or dataset_path.startswith('./'):
            # Local dataset path
            self.dataset = load_dataset(
                "audiofolder", 
                data_dir=dataset_path, 
                split=split,
                streaming=streaming
            )
        else:
            # Hugging Face Hub dataset
            self.dataset = load_dataset(
                dataset_path, 
                split=split,
                streaming=streaming
            )
        
        # Convert to list if not streaming for random access
        if not streaming:
            self.dataset = list(self.dataset)
            print(f"Loaded {len(self.dataset)} audio samples")
        else:
            print("Using streaming dataset")
        
    def __len__(self):
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        else:
            # For streaming datasets, return a large number
            return 1000000
    
    def __getitem__(self, idx):
        try:
            if isinstance(self.dataset, list):
                sample = self.dataset[idx]
            else:
                # For streaming datasets, we need to iterate
                # This is not efficient for random access, but works
                for i, sample in enumerate(self.dataset):
                    if i == idx:
                        break
                else:
                    # If we can't find the index, get a random sample
                    sample = next(iter(self.dataset))
            
            # Extract audio data
            audio_data = sample[self.audio_column]
            
            # Handle different audio formats from HF datasets
            if isinstance(audio_data, dict):
                # Standard HF audio format: {"array": waveform, "sampling_rate": sr}
                waveform = torch.tensor(audio_data["array"], dtype=torch.float32)
                sr = audio_data["sampling_rate"]
            else:
                # If it's already a tensor or array
                waveform = torch.tensor(audio_data, dtype=torch.float32)
                sr = self.sample_rate  # Assume target sample rate
            
            # Ensure waveform is 2D [channels, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Random segment extraction
            if waveform.shape[1] >= self.segment_samples:
                start = random.randint(0, waveform.shape[1] - self.segment_samples)
                waveform = waveform[:, start:start + self.segment_samples]
            else:
                # Pad if too short
                pad_length = self.segment_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_length))
            
            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-8)
            
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            print(f"Error loading audio sample {idx}: {e}")
            # Return silence if loading fails
            return torch.zeros(self.segment_samples)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator from HiFi-GAN."""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, x):
        outputs = []
        feature_maps = []
        
        for discriminator in self.discriminators:
            output, features = discriminator(x)
            outputs.append(output)
            feature_maps.append(features)
        
        return outputs, feature_maps


class PeriodDiscriminator(nn.Module):
    """Single period discriminator."""
    
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))),
        ])
        
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))
    
    def forward(self, x):
        features = []
        
        # Reshape for period-wise processing
        b, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        
        x = x.view(b, 1, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT Discriminator."""
    
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(fft_size=1024, shift_size=120, win_length=600),
            STFTDiscriminator(fft_size=2048, shift_size=240, win_length=1200),
            STFTDiscriminator(fft_size=512, shift_size=50, win_length=240),
        ])
    
    def forward(self, x):
        outputs = []
        feature_maps = []
        
        for discriminator in self.discriminators:
            output, features = discriminator(x)
            outputs.append(output)
            feature_maps.append(features)
        
        return outputs, feature_maps


class STFTDiscriminator(nn.Module):
    """Single STFT-based discriminator."""
    
    def __init__(self, fft_size: int, shift_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(2, 32, (7, 7), (2, 2), (3, 3))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 5), (2, 2), (2, 2))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (3, 3), (2, 2), (1, 1))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (3, 3), (2, 2), (1, 1))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (3, 3), 1, (1, 1))),
        ])
        
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 3), 1, (1, 1)))
    
    def forward(self, x):
        features = []
        
        # Compute STFT
        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.shift_size,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(x.device),
            return_complex=True
        )
        
        # Convert to magnitude and phase
        x_mag = x_stft.abs()
        x_phase = x_stft.angle()
        
        # Stack real and imaginary parts
        x = torch.stack([x_mag, x_phase], dim=1)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, features


def feature_matching_loss(real_features: List, fake_features: List) -> torch.Tensor:
    """Compute feature matching loss between real and fake feature maps."""
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feat, fake_feat):
            loss += F.l1_loss(fake_f, real_f.detach())
    return loss


def discriminator_loss(real_outputs: List, fake_outputs: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute discriminator losses."""
    real_loss = 0
    fake_loss = 0
    
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        # Real loss
        real_loss += F.mse_loss(real_out, torch.ones_like(real_out))
        # Fake loss
        fake_loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))
    
    return real_loss, fake_loss


def generator_loss(fake_outputs: List) -> torch.Tensor:
    """Compute generator adversarial loss."""
    gen_loss = 0
    
    for fake_out in fake_outputs:
        gen_loss += F.mse_loss(fake_out, torch.ones_like(fake_out))
    
    return gen_loss


def mel_spectrogram_loss(real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
    """Compute mel-spectrogram reconstruction loss."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(real_audio.device)
    
    real_mel = mel_transform(real_audio)
    fake_mel = mel_transform(fake_audio)
    
    return F.l1_loss(fake_mel, real_mel)


def stft_loss(real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
    """Compute multi-resolution STFT loss."""
    stft_configs = [
        (512, 50, 240),
        (1024, 120, 600),
        (2048, 240, 1200),
    ]
    
    total_loss = 0
    for fft_size, hop_size, win_length in stft_configs:
        window = torch.hann_window(win_length).to(real_audio.device)
        
        real_stft = torch.stft(
            real_audio, n_fft=fft_size, hop_length=hop_size,
            win_length=win_length, window=window, return_complex=True
        )
        fake_stft = torch.stft(
            fake_audio, n_fft=fft_size, hop_length=hop_size,
            win_length=win_length, window=window, return_complex=True
        )
        
        # Magnitude loss
        real_mag = real_stft.abs()
        fake_mag = fake_stft.abs()
        total_loss += F.l1_loss(fake_mag, real_mag)
        
        # Phase loss (optional, can be commented out)
        real_phase = real_stft.angle()
        fake_phase = fake_stft.angle()
        total_loss += F.l1_loss(fake_phase, real_phase) * 0.1
    
    return total_loss


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


def main():
    # Configuration
    config = {
        "model": create_snac_24khz_config(),
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
            "dataset_path": "bad49wolf/ljspeech-simple",  # HF dataset path or local path
            "audio_column": "audio",  # Column name containing audio data
            "split": "train",  # Dataset split to use
            "streaming": False,  # Set to True for large datasets
            "segment_length": 0.8,  # seconds
        },
        "paths": {
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        }
    }
    
    # Create directories
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Models
    generator = SNAC(**config["model"]).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleSTFTDiscriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()) / 1e6:.1f}M")
    print(f"MPD parameters: {sum(p.numel() for p in mpd.parameters()) / 1e6:.1f}M")
    print(f"MSD parameters: {sum(p.numel() for p in msd.parameters()) / 1e6:.1f}M")
    
    # Optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.8, 0.99),
        weight_decay=1e-2
    )
    
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config["training"]["learning_rate"],
        betas=(0.8, 0.99),
        weight_decay=1e-2
    )
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config["training"]["lr_decay"]
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config["training"]["lr_decay"]
    )
    
    # Dataset
    dataset = AudioDataset(
        dataset_path=config["data"]["dataset_path"],
        sample_rate=config["model"]["sampling_rate"],
        segment_length=config["data"]["segment_length"],
        split=config["data"]["split"],
        streaming=config["data"]["streaming"],
        audio_column=config["data"]["audio_column"]
    )
    
    # For streaming datasets, don't use multiprocessing workers
    num_workers = 0 if config["data"]["streaming"] else 4
    shuffle = not config["data"]["streaming"]  # Can't shuffle streaming datasets
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Training loop
    iteration = 0
    generator.train()
    mpd.train()
    msd.train()
    
    # Create infinite dataloader
    def infinite_dataloader():
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader()
    
    print("Starting training...")
    
    with tqdm(total=config["training"]["num_iterations"], desc="Training") as pbar:
        while iteration < config["training"]["num_iterations"]:
            # Get batch
            real_audio = next(data_iter).to(device).unsqueeze(1)  # Add channel dim
            
            # Train Generator
            optim_g.zero_grad()
            
            # Forward pass
            fake_audio, codes = generator(real_audio)
            
            # Discriminator outputs
            mpd_real_outputs, mpd_real_features = mpd(real_audio)
            mpd_fake_outputs, mpd_fake_features = mpd(fake_audio)
            
            msd_real_outputs, msd_real_features = msd(real_audio)
            msd_fake_outputs, msd_fake_features = msd(fake_audio)
            
            # Generator losses
            gen_loss_mpd = generator_loss(mpd_fake_outputs)
            gen_loss_msd = generator_loss(msd_fake_outputs)
            
            # Feature matching losses
            fm_loss_mpd = feature_matching_loss(mpd_real_features, mpd_fake_features)
            fm_loss_msd = feature_matching_loss(msd_real_features, msd_fake_features)
            
            # Reconstruction losses
            mel_loss = mel_spectrogram_loss(real_audio.squeeze(1), fake_audio.squeeze(1))
            stft_loss_val = stft_loss(real_audio.squeeze(1), fake_audio.squeeze(1))
            
            # Total generator loss
            loss_g = (
                gen_loss_mpd + gen_loss_msd +
                2 * (fm_loss_mpd + fm_loss_msd) +
                45 * mel_loss +
                45 * stft_loss_val
            )
            
            loss_g.backward()
            optim_g.step()
            
            # Train Discriminator
            optim_d.zero_grad()
            
            # MPD losses
            mpd_real_outputs, _ = mpd(real_audio)
            mpd_fake_outputs, _ = mpd(fake_audio.detach())
            mpd_real_loss, mpd_fake_loss = discriminator_loss(mpd_real_outputs, mpd_fake_outputs)
            
            # MSD losses  
            msd_real_outputs, _ = msd(real_audio)
            msd_fake_outputs, _ = msd(fake_audio.detach())
            msd_real_loss, msd_fake_loss = discriminator_loss(msd_real_outputs, msd_fake_outputs)
            
            # Total discriminator loss
            loss_d = mpd_real_loss + mpd_fake_loss + msd_real_loss + msd_fake_loss
            
            loss_d.backward()
            optim_d.step()
            
            # Update learning rates
            scheduler_g.step()
            scheduler_d.step()
            
            iteration += 1
            
            # Logging
            if iteration % config["training"]["log_interval"] == 0:
                pbar.set_postfix({
                    'G_loss': f'{loss_g.item():.3f}',
                    'D_loss': f'{loss_d.item():.3f}',
                    'Mel': f'{mel_loss.item():.3f}',
                    'STFT': f'{stft_loss_val.item():.3f}',
                    'LR': f'{scheduler_g.get_last_lr()[0]:.2e}'
                })
            
            # Save checkpoint
            if iteration % config["training"]["save_interval"] == 0:
                checkpoint = {
                    'iteration': iteration,
                    'generator': generator.state_dict(),
                    'mpd': mpd.state_dict(),
                    'msd': msd.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'scheduler_g': scheduler_g.state_dict(),
                    'scheduler_d': scheduler_d.state_dict(),
                    'config': config
                }
                
                checkpoint_path = os.path.join(
                    config["paths"]["checkpoint_dir"], 
                    f"snac_24khz_iter_{iteration}.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Save model config
                with open(os.path.join(config["paths"]["checkpoint_dir"], "config.json"), 'w') as f:
                    json.dump(config["model"], f, indent=2)
            
            pbar.update(1)
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint = {
        'iteration': iteration,
        'generator': generator.state_dict(),
        'config': config["model"]
    }
    
    final_path = os.path.join(config["paths"]["checkpoint_dir"], "snac_24khz_final.pt")
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
