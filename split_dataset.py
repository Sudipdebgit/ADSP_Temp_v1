import os
import json
import random
import argparse


def split_dataset(
    meta_path,
    output_dir,
    train_ratio=0.8,
    seed=1234,
):
    """
    Split dataset into train and validation sets.
    
    Args:
        meta_path: path to original meta.jsonl
        output_dir: directory to save split metadata files
        train_ratio: fraction of data for training
        seed: random seed
    """
    random.seed(seed)
    
    # Load all samples
    samples = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Total samples: {len(samples)}")
    
    # Shuffle
    random.shuffle(samples)
    
    # Split
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_meta.jsonl")
    val_path = os.path.join(output_dir, "val_meta.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\nSaved train metadata to: {train_path}")
    print(f"Saved validation metadata to: {val_path}")
    
    return train_path, val_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val")
    parser.add_argument("--meta_path", default="dataset_stft/meta.jsonl",
                        help="Path to original metadata file")
    parser.add_argument("--output_dir", default="dataset_stft",
                        help="Directory to save split files")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    
    args = parser.parse_args()
    
    split_dataset(
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
