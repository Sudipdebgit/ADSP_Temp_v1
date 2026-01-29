#!/bin/bash

# Complete BSS Training and Evaluation Pipeline
# This script runs all steps from data generation to inference

set -e  # Exit on error

echo "========================================="
echo "BSS Semi-Supervised Training Pipeline"
echo "========================================="

# Configuration
SPEECH_DIR="speech_pool"
NOISE_DIR="noise_pool"
DATASET_DIR="dataset"
STFT_DIR="dataset_stft"
CHECKPOINT_DIR="checkpoints"
INFERENCE_DIR="inference_outputs"

NUM_MIXES=3000
NUM_EPOCHS=5
BATCH_SIZE=4  # Very small for CPU/low memory
SUPERVISED_RATIO=0.5  # Change to 0.5 for semi-supervised (50% labeled)
N_SAMPLES_EVAL=20  # Reduced for faster evaluation
N_SAMPLES_INFERENCE=5  # Reduced for faster inference



# Step 5: Evaluate model
echo -e "\n[Step 5] Evaluating model..."
python bss_evaluate.py \
    --checkpoint "$CHECKPOINT_DIR/best_model.pth" \
    --meta_path "$STFT_DIR/val_meta.jsonl" \
    --output "evaluation_results.json" \
    --n_samples $N_SAMPLES_EVAL \
    --n_fft 512 \
    --hop_length 128

# Step 6: Generate inference outputs
echo -e "\n[Step 6] Generating audio outputs..."
python bss_inference.py \
    --checkpoint "$CHECKPOINT_DIR/best_model.pth" \
    --meta_path "$STFT_DIR/val_meta.jsonl" \
    --output_dir "$INFERENCE_DIR" \
    --n_samples $N_SAMPLES_INFERENCE \
    --n_fft 512 \
    --hop_length 128 \
    --sr 16000

echo -e "\n========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Results:"
echo "  - Checkpoints: $CHECKPOINT_DIR"
echo "  - Evaluation: evaluation_results.json"
echo "  - Audio outputs: $INFERENCE_DIR"
echo "========================================="
