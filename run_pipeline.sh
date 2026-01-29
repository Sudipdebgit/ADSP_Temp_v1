
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
BATCH_SIZE=8  # Very small for CPU/low memory
SUPERVISED_RATIO=0.5  # Change to 0.5 for semi-supervised (50% labeled)
N_SAMPLES_EVAL=25  # Reduced for faster evaluation
N_SAMPLES_INFERENCE=5  # Reduced for faster inference


# Step 0: Download speech data (optional, if not already done)
echo -e "\n[Step 0] Checking speech data..."
if [ ! -d "$SPEECH_DIR" ] || [ -z "$(ls -A $SPEECH_DIR)" ]; then
    echo "Downloading LibriSpeech data..."
    python fetch_libre_stream.py \
        --out_dir "$SPEECH_DIR" \
        --num_files 200 \
        --split test.clean
else
    echo "Speech data already exists in $SPEECH_DIR"
fi

# Step 1: Generate mixed dataset
echo -e "\n[Step 1] Generating mixed dataset..."
if [ ! -d "$DATASET_DIR" ]; then
    python generate_dataset.py \
        --speech_dir "$SPEECH_DIR" \
        --noise_dir "$NOISE_DIR" \
        --out_dir "$DATASET_DIR" \
        --num_mixes $NUM_MIXES \
        --mix_seconds 4.0 \
        --snr_db_range "(-5.0, 5.0)" \
        --seed 1234
else
    echo "Dataset already exists in $DATASET_DIR"
fi

# Step 2: Convert to STFT
echo -e "\n[Step 2] Converting to STFT format..."
if [ ! -d "$STFT_DIR" ]; then
    python convert_to_stft.py \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$STFT_DIR" \
        --n_fft 512 \
        --hop_length 128
else
    echo "STFT data already exists in $STFT_DIR"
fi

# Step 3: Split dataset
echo -e "\n[Step 3] Splitting dataset into train/val..."
python split_dataset.py \
    --meta_path "$STFT_DIR/meta.jsonl" \
    --output_dir "$STFT_DIR" \
    --train_ratio 0.8 \
    --seed 1234

# Step 4: Train model
echo -e "\n[Step 4] Training model..."
python bss_train.py \
    --train_meta "$STFT_DIR/train_meta.jsonl" \
    --val_meta "$STFT_DIR/val_meta.jsonl" \
    --output_dir "$CHECKPOINT_DIR" \
    --supervised_ratio $SUPERVISED_RATIO \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr 1e-3 \
    --lambda_unsup 0.1 \
    --num_workers 4 \
    --save_every 5


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
