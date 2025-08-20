#!/usr/bin/env python3
"""
SNAC Training Loop

Main training script for SNAC models with GAN-based training using 
Multi-Period and Multi-Scale STFT discriminators.
"""

import os
import json
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from snac import SNAC
from .dataset import AudioDataset
from .discriminators import MultiPeriodDiscriminator, MultiScaleSTFTDiscriminator
from .losses import (
    feature_matching_loss,
    discriminator_loss,
    generator_loss,
    mel_spectrogram_loss,
    stft_loss
)
from .config import create_training_config


def setup_models(config: Dict, device: str) -> tuple:
    """Initialize generator and discriminator models."""
    generator = SNAC(**config["model"]).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleSTFTDiscriminator().to(device)
    print(generator.sampling_rate)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()) / 1e6:.1f}M")
    print(f"MPD parameters: {sum(p.numel() for p in mpd.parameters()) / 1e6:.1f}M")
    print(f"MSD parameters: {sum(p.numel() for p in msd.parameters()) / 1e6:.1f}M")
    
    return generator, mpd, msd


def setup_optimizers(generator, mpd, msd, config: Dict) -> tuple:
    """Initialize optimizers and schedulers."""
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
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config["training"]["lr_decay"]
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config["training"]["lr_decay"]
    )
    
    return optim_g, optim_d, scheduler_g, scheduler_d


def setup_dataset(config: Dict) -> DataLoader:
    """Setup dataset and dataloader."""
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
    
    return dataloader


def train_step(
    real_audio, generator, mpd, msd, 
    optim_g, optim_d, device, iteration: int = 0
) -> Dict[str, float]:
    """Single training step."""
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
    
    # Total generator loss with scaled weights for stability
    # Original paper uses 45x for mel/stft, but we scale down during early training
    mel_weight = 5.0 if iteration < 500 else 15.0  # Gradual increase
    stft_weight = 5.0 if iteration < 500 else 15.0
    
    loss_g = (
        gen_loss_mpd + gen_loss_msd +
        2 * (fm_loss_mpd + fm_loss_msd) +
        mel_weight * mel_loss +
        stft_weight * stft_loss_val
    )
    
    loss_g.backward()
    # Gradient clipping to prevent instability
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
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
    # Gradient clipping to prevent instability
    torch.nn.utils.clip_grad_norm_(
        list(mpd.parameters()) + list(msd.parameters()), max_norm=1.0
    )
    optim_d.step()
    
    return {
        "loss_g": loss_g.item(),
        "loss_d": loss_d.item(),
        "mel_loss": mel_loss.item(),
        "stft_loss": stft_loss_val.item(),
        "gen_adv_loss": (gen_loss_mpd + gen_loss_msd).item(),
        "fm_loss": (fm_loss_mpd + fm_loss_msd).item(),
        "mel_weight": mel_weight,
        "stft_weight": stft_weight,
    }


def save_checkpoint(
    iteration, generator, mpd, msd, 
    optim_g, optim_d, scheduler_g, scheduler_d, 
    config, checkpoint_dir
):
    """Save training checkpoint."""
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
    
    checkpoint_path = os.path.join(checkpoint_dir, f"snac_24khz_iter_{iteration}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save model config
    with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
        json.dump(config["model"], f, indent=2)


def main(config: Dict = None):
    """Main training function."""
    if config is None:
        config = create_training_config()
    
    # Create directories
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup models
    generator, mpd, msd = setup_models(config, device)
    
    # Setup optimizers
    optim_g, optim_d, scheduler_g, scheduler_d = setup_optimizers(
        generator, mpd, msd, config
    )
    
    # Setup dataset
    dataloader = setup_dataset(config)
    
    # Training setup
    generator.train()
    mpd.train()
    msd.train()
    
    # Create infinite dataloader
    def infinite_dataloader():
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader()
    iteration = 0
    
    print("Starting training...")
    
    with tqdm(total=config["training"]["num_iterations"], desc="Training") as pbar:
        while iteration < config["training"]["num_iterations"]:
            # Get batch
            real_audio = next(data_iter).to(device).unsqueeze(1)  # Add channel dim
            
            # Training step
            losses = train_step(
                real_audio, generator, mpd, msd, 
                optim_g, optim_d, device, iteration
            )
            
            # Update learning rates
            scheduler_g.step()
            scheduler_d.step()
            
            iteration += 1
            
            # Logging
            if iteration % config["training"]["log_interval"] == 0:
                pbar.set_postfix({
                    'G_loss': f'{losses["loss_g"]:.1f}',
                    'D_loss': f'{losses["loss_d"]:.3f}',
                    'Mel': f'{losses["mel_loss"]:.2f}',
                    'STFT': f'{losses["stft_loss"]:.2f}',
                    'Adv': f'{losses["gen_adv_loss"]:.2f}',
                    'FM': f'{losses["fm_loss"]:.2f}',
                    'MW': f'{losses["mel_weight"]:.1f}',
                    'LR': f'{scheduler_g.get_last_lr()[0]:.2e}'
                })
            
            # Save checkpoint
            if iteration % config["training"]["save_interval"] == 0:
                save_checkpoint(
                    iteration, generator, mpd, msd,
                    optim_g, optim_d, scheduler_g, scheduler_d,
                    config, config["paths"]["checkpoint_dir"]
                )
            
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
