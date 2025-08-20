#!/usr/bin/env python3
"""
SNAC Discriminators

This module contains discriminator models used in SNAC training:
- Multi-Period Discriminator (MPD) from HiFi-GAN  
- Multi-Resolution STFT Discriminator (MRD) with frequency band splitting
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def WNConv1d(*args, **kwargs):
    """Weight-normalized 1D convolution with optional activation."""
    act = kwargs.pop("act", True)
    conv = nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    """Weight-normalized 2D convolution with optional activation."""
    act = kwargs.pop("act", True)
    conv = nn.utils.weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


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
            fmap = discriminator(x)
            # Extract final output from features (last element)
            output = fmap[-1]
            outputs.append(output)
            feature_maps.append(fmap)
        
        return outputs, feature_maps


class PeriodDiscriminator(nn.Module):
    """Single period discriminator matching DAC implementation."""
    
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        
        self.conv_post = WNConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)
    
    def pad_to_period(self, x):
        """Pad input to be divisible by period."""
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x
    
    def forward(self, x):
        from einops import rearrange
        fmap = []
        
        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)
        
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        
        return fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT Discriminator with frequency band splitting."""
    
    def __init__(
        self, 
        fft_sizes: List[int] = [2048, 1024, 512],
        sample_rate: int = 16000,
        bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(fft_size, sample_rate, bands) for fft_size in fft_sizes
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
    """Single STFT-based discriminator with frequency band splitting."""
    
    def __init__(
        self, 
        window_length: int,
        sample_rate: int = 16000,
        bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]],
        hop_factor: float = 0.25
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        
        # Calculate frequency bands in terms of FFT bins
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        
        # Create convolution layers for each band
        ch = 32
        convs = lambda: nn.ModuleList([
            WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        ])
        
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)
    
    def spectrogram(self, x):
        """Compute complex STFT and split into frequency bands."""
        from einops import rearrange
        
        # Compute STFT
        window = torch.hann_window(self.window_length).to(x.device)
        stft = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=int(self.window_length * self.hop_factor),
            win_length=self.window_length,
            window=window,
            return_complex=True
        )
        
        # Convert to real and imaginary parts
        x = torch.view_as_real(stft)  # (B, F, T, 2)
        x = rearrange(x, "b f t c -> b c t f")  # (B, 2, T, F)
        
        # Split into frequency bands
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands
    
    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []
        
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        
        # Concatenate all bands
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        
        return fmap


class Discriminator(nn.Module):
    """Combined discriminator matching DAC configuration."""
    
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        fft_sizes: List[int] = [2048, 1024, 512],
        sample_rate: int = 16000,
        bands: List[List[float]] = [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]
    ):
        super().__init__()
        
        discriminators = []
        # Add multi-period discriminators
        discriminators += [PeriodDiscriminator(p) for p in periods]
        # Add multi-resolution STFT discriminators  
        discriminators += [STFTDiscriminator(f, sample_rate, bands) for f in fft_sizes]
        
        self.discriminators = nn.ModuleList(discriminators)
    
    def preprocess(self, y):
        """Preprocess audio as in DAC."""
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y
    
    def forward(self, x):
        x = self.preprocess(x)
        feature_maps = [d(x) for d in self.discriminators]
        return feature_maps


if __name__ == "__main__":
    # Test the discriminator
    disc = Discriminator()
    x = torch.randn(1, 1, 16000)  # 1 second of 16kHz audio
    
    results = disc(x)
    print(f"Number of discriminators: {len(results)}")
    
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {total_params:,}")
    
    for i, result in enumerate(results):
        print(f"Discriminator {i} feature maps: {len(result)}")
        for j, feature in enumerate(result):
            print(f"  Feature {j}: {feature.shape}")