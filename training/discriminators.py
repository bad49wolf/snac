#!/usr/bin/env python3
"""
SNAC Discriminators

This module contains discriminator models used in SNAC training:
- Multi-Period Discriminator (MPD) from HiFi-GAN
- Multi-Scale STFT Discriminator (MSD) for spectrogram-based discrimination
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Handle both 2D and 3D input tensors
        if x.dim() == 3:
            # Input is (batch, channels, time) - squeeze channel dimension
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor with shape {x.shape}")
        
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
        
        # Handle both 2D and 3D input tensors
        if x.dim() == 3:
            # Input is (batch, channels, time) - squeeze channel dimension
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor with shape {x.shape}")
        
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
