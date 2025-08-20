#!/usr/bin/env python3
"""
SNAC Loss Functions

This module contains all loss functions used in SNAC training:
- Adversarial losses for generator and discriminator
- Feature matching loss between real and fake feature maps
- Reconstruction losses (mel-spectrogram and multi-resolution STFT)
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio


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
