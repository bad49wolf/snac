#!/usr/bin/env python3
"""
SNAC Audio Dataset

Dataset module for loading and preprocessing audio data from Hugging Face datasets
with "audio" field support for SNAC training.
"""

import random
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from datasets import load_dataset


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
            
            # Debug: Check format of first sample
            if len(self.dataset) > 0:
                sample = self.dataset[0]
                audio_data = sample.get(self.audio_column)
                print(f"Audio data type in dataset: {type(audio_data)}")
                if hasattr(audio_data, '__dict__'):
                    print(f"Audio data attributes: {list(audio_data.__dict__.keys())}")
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
            elif hasattr(audio_data, 'get_all_samples') or 'AudioDecoder' in str(type(audio_data)):
                # Handle torchcodec AudioDecoder objects
                try:
                    if hasattr(audio_data, 'get_all_samples'):
                        # Use torchcodec AudioDecoder's get_all_samples method
                        audio_samples = audio_data.get_all_samples()
                        
                        # Extract tensor data from AudioSamples object
                        if hasattr(audio_samples, 'data'):
                            waveform = audio_samples.data
                            # Get sample rate from AudioSamples
                            if hasattr(audio_samples, 'sample_rate'):
                                sr = audio_samples.sample_rate
                            else:
                                sr = self.sample_rate
                        else:
                            # Fallback - try to convert the object directly
                            waveform = torch.tensor(audio_samples, dtype=torch.float32)
                            sr = self.sample_rate
                    elif hasattr(audio_data, 'decode'):
                        # Fallback to regular decode method
                        decoded_audio = audio_data.decode()
                        if isinstance(decoded_audio, dict):
                            waveform = torch.tensor(decoded_audio["array"], dtype=torch.float32)
                            sr = decoded_audio["sampling_rate"]
                        else:
                            waveform = torch.tensor(decoded_audio, dtype=torch.float32)
                            sr = self.sample_rate
                    else:
                        raise ValueError("Unknown AudioDecoder format")
                        
                except Exception as decode_error:
                    print(f"Failed to decode AudioDecoder: {decode_error}")
                    print(f"AudioDecoder type: {type(audio_data)}")
                    print(f"Available methods: {[m for m in dir(audio_data) if not m.startswith('_')]}")
                    return torch.zeros(self.segment_samples)
            elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
                # Handle objects with array and sampling_rate attributes
                waveform = torch.tensor(audio_data.array, dtype=torch.float32)
                sr = audio_data.sampling_rate
            else:
                # If it's already a tensor, array, or other format
                try:
                    if isinstance(audio_data, (list, tuple)):
                        waveform = torch.tensor(audio_data, dtype=torch.float32)
                    elif isinstance(audio_data, np.ndarray):
                        waveform = torch.from_numpy(audio_data).float()
                    else:
                        waveform = torch.tensor(audio_data, dtype=torch.float32)
                    sr = self.sample_rate  # Assume target sample rate
                except Exception as convert_error:
                    print(f"Failed to convert audio data to tensor: {convert_error}")
                    print(f"Audio data type: {type(audio_data)}")
                    return torch.zeros(self.segment_samples)
            
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
            print(f"Audio data type: {type(audio_data)}")
            print(f"Audio data attributes: {dir(audio_data) if hasattr(audio_data, '__dict__') else 'No attributes'}")
            # Return silence if loading fails
            return torch.zeros(self.segment_samples)
