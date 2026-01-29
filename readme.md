# Speech-Noise Separation using Semi-Supervised BSS

A complete pipeline for training a semi-supervised blind source separation (BSS) model to separate speech from noise/music using STFT magnitude spectrograms.

## Overview

This project implements:
- CNN-based architecture (~127k parameters) for source separation
- Semi-supervised learning (supports both labeled and unlabeled data)
- STFT-based processing with magnitude and phase
- Comprehensive evaluation with SIR, SAR, SDR metrics
- Audio reconstruction for listening to separated sources
- Custom audio separation for your own files

## Features

- Train on 4-second audio clips
- Separate speech from noise/music
- Process audio of any length (automatic chunking with overlap)
- Support for multiple audio formats (wav, mp3, m4a, flac, ogg)
- Real-time factor monitoring
- Complete end-to-end timing breakdown

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy
- soundfile, pydub
- mir_eval (for BSS metrics)
- tqdm, tensorboard

See `requirements.txt` for complete list.

## Installation

### 1. Clone/Download the Project

```bash
cd your_project_directory
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (for m4a/mp3 support)

Ubuntu/Debian:
```bash
sudo apt-get install ffmpeg
```

macOS:
```bash
brew install ffmpeg
```

Windows:
Download from https://ffmpeg.org/download.html

## Quick Start

### Option 1: Run Complete Pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```
### Option 2: Run Only Evaluation Part

```bash
chmod +x run_pipeline_2.sh
./run_pipeline.sh
```

This will automatically:
1. Download LibriSpeech speech data
2. Generate mixed dataset (speech + noise)
3. Convert to STFT format
4. Split into train/validation sets
5. Train the model
6. Evaluate with BSS metrics
7. Generate audio outputs

### Option 3: Step-by-Step

#### Step 1: Prepare Noise Data

Create a `noise_pool/` directory and add your noise samples (`.wav` files):

```
noise_pool/
├── noise_001.wav
├── noise_002.wav
└── ...
```

#### Step 2: Generate Dataset

```bash
python generate_dataset.py \
    --speech_dir speech_pool \
    --noise_dir noise_pool \
    --out_dir dataset \
    --num_mixes 3000 \
    --mix_seconds 4.0
```

#### Step 3: Convert to STFT

```bash
python convert_to_stft.py \
    --dataset_dir dataset \
    --output_dir dataset_stft \
    --n_fft 512 \
    --hop_length 128
```

#### Step 4: Split Dataset

```bash
python split_dataset.py \
    --meta_path dataset_stft/meta.jsonl \
    --output_dir dataset_stft \
    --train_ratio 0.8
```

#### Step 5: Train Model

```bash
# Semi-supervised (50% labeled data)
python bss_train.py \
    --train_meta dataset_stft/train_meta.jsonl \
    --val_meta dataset_stft/val_meta.jsonl \
    --output_dir checkpoints \
    --supervised_ratio 0.5 \
    --batch_size 8 \
    --num_epochs 50

# Monitor training
tensorboard --logdir checkpoints/logs
```

#### Step 6: Evaluate Model

```bash
python bss_evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --meta_path dataset_stft/val_meta.jsonl \
    --output evaluation_results.json \
    --n_samples 25
```

#### Step 7: Generate Audio Outputs

```bash
# Process validation samples
python bss_inference.py \
    --checkpoint checkpoints/best_model.pth \
    --meta_path dataset_stft/val_meta.jsonl \
    --output_dir inference_outputs \
    --n_samples 5
```

## Separate Your Own Audio

### Basic Usage

```bash
# Separate any audio file
python separate_my_audio.py your_audio.wav

# Supported formats: wav, mp3, m4a, flac, ogg, etc.
python separate_my_audio.py my_song.mp3
python separate_my_audio.py podcast.m4a
```

### Advanced Options

```bash
# Specify output directory
python separate_my_audio.py audio.wav --output_dir my_results

# Use different checkpoint
python separate_my_audio.py audio.wav --checkpoint checkpoints/checkpoint_epoch_10.pth

# Adjust chunk size (important for model trained on 4s clips)
python separate_my_audio.py long_audio.mp3 --chunk_duration 4

# Process short audio without chunking
python separate_my_audio.py short_clip.wav --chunk_duration 0
```

### Output Files

After separation, you'll get 3 files:
- `yourfile_original.wav` - Original audio
- `yourfile_speech.wav` - Separated speech
- `yourfile_noise_music.wav` - Separated noise/music/background

### Example Output

```
PROCESSING SUMMARY
============================================================
Audio Loading:           125.34 ms
Audio -> STFT:            45.67 ms
Model Inference:         936.56 ms
ISTFT -> Audio:          144.25 ms
------------------------------------------------------------
Total Processing:       1251.82 ms
Audio Duration:            5.23 s
Real-time Factor:         4.18x
============================================================
```

## Understanding the Results

### BSS Metrics

- SDR (Signal-to-Distortion Ratio): Overall separation quality
  - Higher is better (typically 5-15 dB)
  
- SIR (Signal-to-Interference Ratio): How well sources are separated
  - Higher is better (typically 10-20 dB)
  
- SAR (Signal-to-Artifacts Ratio): Artifacts introduced by separation
  - Higher is better (typically 10-20 dB)

### Timing Breakdown

- Audio -> STFT: Time to convert audio to frequency domain
- Model Inference: Time for neural network to process
- ISTFT -> Audio: Time to convert back to time domain
- Total Processing: Complete end-to-end time
- Real-time Factor: How much faster than real-time (higher is better)

### Listening to Results

Compare the separated audio files:
1. Original mix (speech + noise)
2. Separated speech (cleaner speech)
3. Separated noise/music (background removed)

## Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run_pipeline.sh               # Complete pipeline automation
│
├── split_dataset.py              # Split data into train/val
├── bss_dataset.py                # PyTorch Dataset class
├── bss_model.py                  # Model architecture (~25k params)
├── bss_train.py                  # Training script
├── bss_evaluate.py               # Evaluation with BSS metrics
├── bss_inference.py              # Generate audio outputs
├── separate_my_audio.py          # Separate custom audio files
│
├── generate_dataset.py           # Generate speech+noise mixtures
├── convert_to_stft.py            # Convert audio to STFT
│
├── dataset/                      # Generated mixed audio
├── dataset_stft/                 # STFT representations
├── checkpoints/                  # Model checkpoints
├── inference_outputs/            # Generated audio files
└── separated_output/             # Your separated audio
```

## Model Architecture

The model uses:
- Input: 2-channel STFT magnitude (257 freq bins)
- Architecture: CNN with dilated residual blocks
- Parameters: ~127,000 trainable parameters
- Output: 2 separated source magnitudes (speech + noise)

Channel progression: 2 → 34 → 40 → 42 → 40 → 34 → 32 → 4

## Training Configuration

Default settings:
- Audio length: 4 seconds
- Sample rate: 16 kHz
- FFT size: 512
- Hop length: 128
- Batch size: 8
- Epochs: 50
- Learning rate: 1e-3
- Optimizer: Adam

## Tips and Best Practices

### For Best Separation Quality

1. Use chunk size close to training length (4-5 seconds)
2. Ensure good quality training data (clean speech + diverse noise)
3. Train for sufficient epochs (30-50 epochs)
4. Use validation set to monitor overfitting

### For Processing Long Audio

1. Use 4-5 second chunks (matches training data)
2. Enable overlap (25% overlap is good)
3. Be patient with very long files (>10 minutes)

### Troubleshooting

Format Not Supported:
- Install ffmpeg
- Or convert to .wav first

## Performance Benchmarks

Typical performance (CPU):
- 4-second audio: ~1 second processing time
- 1-minute audio: ~15 seconds processing time
- Real-time factor: 3-5x on modern CPU

With GPU:
- Real-time factor: 10-20x or higher

## Citation

- PyRoomAcoustics for room simulation
- mir_eval for BSS metrics
- LibriSpeech for speech data


