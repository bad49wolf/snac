#!/usr/bin/env python3
"""
SNAC Dataset Examples

This file shows how to configure the training script for different types of 
Hugging Face datasets and local audio folders.
"""

# Example configurations for different dataset types

# 1. Local audio folder (audiofolder format)
LOCAL_FOLDER_CONFIG = {
    "data": {
        "dataset_path": "/path/to/local/audio/folder",  # Local folder with audio files
        "audio_column": "audio",
        "split": "train", 
        "streaming": False,
        "segment_length": 0.8,
    }
}

# 2. Hugging Face Hub dataset (e.g., common_voice)
HF_DATASET_CONFIG = {
    "data": {
        "dataset_path": "mozilla-foundation/common_voice_16_1",  # HF dataset name
        "audio_column": "audio",  # Column containing audio data
        "split": "train",
        "streaming": True,  # Use streaming for large datasets
        "segment_length": 0.8,
    }
}

# 3. Custom dataset with different audio column name
CUSTOM_CONFIG = {
    "data": {
        "dataset_path": "your-username/your-audio-dataset",
        "audio_column": "speech",  # Different column name
        "split": "train",
        "streaming": False,
        "segment_length": 0.8,
    }
}

# 4. LibriSpeech dataset example
LIBRISPEECH_CONFIG = {
    "data": {
        "dataset_path": "librispeech_asr",  # Official LibriSpeech
        "audio_column": "audio",
        "split": "train.clean.100",  # Specific subset
        "streaming": True,  # Large dataset, use streaming
        "segment_length": 0.8,
    }
}

# 5. VCTK dataset example  
VCTK_CONFIG = {
    "data": {
        "dataset_path": "vctk",
        "audio_column": "audio", 
        "split": "train",
        "streaming": False,
        "segment_length": 0.8,
    }
}

if __name__ == "__main__":
    print("Dataset Configuration Examples for SNAC Training")
    print("=" * 50)
    
    print("\n1. Local Audio Folder:")
    print("   - Place audio files (.wav, .flac, .mp3) in a folder")
    print("   - Set dataset_path to the folder path")
    print("   - streaming: False (for random access)")
    
    print("\n2. Hugging Face Dataset:")
    print("   - Use any HF dataset with 'audio' field")
    print("   - Set streaming: True for large datasets")
    print("   - Common datasets: common_voice, librispeech_asr, vctk")
    
    print("\n3. Custom Dataset:")
    print("   - Upload your dataset to HF Hub")
    print("   - Ensure audio files are in 'audio' column")
    print("   - Use your dataset name: 'username/dataset-name'")
    
    print("\n4. Tips:")
    print("   - Use streaming=True for datasets > 10GB")
    print("   - Use streaming=False for better shuffling")
    print("   - Ensure audio is mono and reasonably sampled")
    print("   - Check your dataset has sufficient duration per sample")
