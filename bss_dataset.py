import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class STFTDataset(Dataset):
    """
    Dataset for loading STFT magnitude and phase data.
    
    For supervised learning: returns (mix, s1_target, s2_target)
    For semi-supervised: can mask out some targets
    """
    
    def __init__(self, meta_path, supervised_ratio=1.0, normalize=True, max_time_frames=None):
        """
        Args:
            meta_path: path to meta.jsonl file
            supervised_ratio: fraction of data with labels (1.0 = fully supervised)
            normalize: whether to normalize magnitude spectrograms
            max_time_frames: maximum time frames for padding (None = auto-detect)
        """
        self.meta_path = meta_path
        self.supervised_ratio = supervised_ratio
        self.normalize = normalize
        
        # Load metadata
        self.samples = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples from {meta_path}")
        
        # Determine which samples are supervised
        n_supervised = int(len(self.samples) * supervised_ratio)
        self.supervised_indices = set(range(n_supervised))
        
        print(f"Supervised samples: {n_supervised}/{len(self.samples)} ({supervised_ratio*100:.1f}%)")
        
        # Determine max time frames if not specified
        if max_time_frames is None:
            max_frames = 0
            print("Determining max time frames...")
            for i, sample in enumerate(self.samples[:min(100, len(self.samples))]):
                try:
                    mag = np.load(sample['mix_magnitude_path'])
                    max_frames = max(max_frames, mag.shape[2])
                except:
                    pass
            self.max_time_frames = max_frames
            print(f"Max time frames: {self.max_time_frames}")
        else:
            self.max_time_frames = max_time_frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        meta = self.samples[idx]
        
        # Load mix (input)
        mix_mag = np.load(meta['mix_magnitude_path'])  # (2, freq, time)
        mix_phase = np.load(meta['mix_phase_path'])    # (2, freq, time)
        
        # Load targets (if supervised)
        is_supervised = idx in self.supervised_indices
        
        if is_supervised:
            s1_mag = np.load(meta['s1_magnitude_path'])   # speech
            s1_phase = np.load(meta['s1_phase_path'])
            s2_mag = np.load(meta['s2_magnitude_path'])   # noise
            s2_phase = np.load(meta['s2_phase_path'])
        else:
            # Unsupervised: no targets
            s1_mag = np.zeros_like(mix_mag)
            s1_phase = np.zeros_like(mix_phase)
            s2_mag = np.zeros_like(mix_mag)
            s2_phase = np.zeros_like(mix_phase)
        
        # Pad or crop to max_time_frames
        mix_mag = self._pad_or_crop(mix_mag, self.max_time_frames)
        mix_phase = self._pad_or_crop(mix_phase, self.max_time_frames)
        s1_mag = self._pad_or_crop(s1_mag, self.max_time_frames)
        s1_phase = self._pad_or_crop(s1_phase, self.max_time_frames)
        s2_mag = self._pad_or_crop(s2_mag, self.max_time_frames)
        s2_phase = self._pad_or_crop(s2_phase, self.max_time_frames)
        
        # Normalize magnitude (optional, helps training)
        if self.normalize:
            # Global max normalization
            max_val = np.max(mix_mag) + 1e-8
            mix_mag = mix_mag / max_val
            if is_supervised:
                s1_mag = s1_mag / max_val
                s2_mag = s2_mag / max_val
        
        # Convert to torch tensors
        mix_mag = torch.from_numpy(mix_mag).float()
        mix_phase = torch.from_numpy(mix_phase).float()
        s1_mag = torch.from_numpy(s1_mag).float()
        s1_phase = torch.from_numpy(s1_phase).float()
        s2_mag = torch.from_numpy(s2_mag).float()
        s2_phase = torch.from_numpy(s2_phase).float()
        
        return {
            'mix_mag': mix_mag,
            'mix_phase': mix_phase,
            's1_mag': s1_mag,
            's1_phase': s1_phase,
            's2_mag': s2_mag,
            's2_phase': s2_phase,
            'is_supervised': torch.tensor(is_supervised, dtype=torch.bool),
            'sample_id': meta['id'],
        }
    
    def _pad_or_crop(self, array, target_time):
        """Pad or crop array to target time frames"""
        current_time = array.shape[2]
        
        if current_time == target_time:
            return array
        elif current_time < target_time:
            # Pad
            pad_width = ((0, 0), (0, 0), (0, target_time - current_time))
            return np.pad(array, pad_width, mode='constant', constant_values=0)
        else:
            # Crop
            return array[:, :, :target_time]


def create_dataloaders(
    train_meta_path,
    val_meta_path,
    batch_size=16,
    supervised_ratio=1.0,
    num_workers=4,
):
    """
    Create train and validation dataloaders.
    """
    train_dataset = STFTDataset(
        train_meta_path,
        supervised_ratio=supervised_ratio,
        normalize=True,
    )
    
    val_dataset = STFTDataset(
        val_meta_path,
        supervised_ratio=1.0,  # Always supervised for validation
        normalize=True,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory for CPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory for CPU
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    meta_path = "dataset_stft/meta.jsonl"
    
    dataset = STFTDataset(meta_path, supervised_ratio=0.5)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print("\nSample shapes:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")
