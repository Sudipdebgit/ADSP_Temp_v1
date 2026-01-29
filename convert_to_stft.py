import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm


def compute_stft(audio, n_fft=512, hop_length=128, win_length=None):
    """
    Compute STFT and return magnitude and phase.
    
    Args:
        audio: (n_samples,) or (n_samples, n_channels)
        n_fft: FFT size
        hop_length: hop length in samples
        win_length: window length (defaults to n_fft)
    
    Returns:
        magnitude: (n_channels, freq_bins, time_frames)
        phase: (n_channels, freq_bins, time_frames)
    """
    if win_length is None:
        win_length = n_fft
    
    # Handle mono/stereo
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]  # (n_samples, 1)
    
    n_channels = audio.shape[1]
    
    # Hann window
    window = np.hanning(win_length).astype(np.float32)
    
    # Pad window if needed
    if win_length < n_fft:
        window = np.pad(window, (0, n_fft - win_length), mode='constant')
    
    magnitudes = []
    phases = []
    
    for ch in range(n_channels):
        # Compute STFT
        stft = librosa_stft(audio[:, ch], n_fft=n_fft, hop_length=hop_length, 
                           window=window, center=True)
        
        # Extract magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        magnitudes.append(magnitude)
        phases.append(phase)
    
    magnitude = np.stack(magnitudes, axis=0).astype(np.float32)  # (n_channels, freq, time)
    phase = np.stack(phases, axis=0).astype(np.float32)
    
    return magnitude, phase


def librosa_stft(y, n_fft, hop_length, window, center=True):
    """
    Simple STFT implementation similar to librosa.
    """
    if center:
        # Pad signal on both sides
        pad_len = n_fft // 2
        y = np.pad(y, (pad_len, pad_len), mode='reflect')
    
    # Number of frames
    n_frames = 1 + (len(y) - n_fft) // hop_length
    
    # Initialize output
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + n_fft] * window
        
        # Compute FFT and keep only positive frequencies
        fft = np.fft.rfft(frame, n=n_fft)
        stft_matrix[:, i] = fft
    
    return stft_matrix


def process_audio_file(wav_path, n_fft=512, hop_length=128):
    """
    Load audio file and convert to STFT magnitude and phase.
    
    Returns:
        magnitude: (n_channels, freq_bins, time_frames)
        phase: (n_channels, freq_bins, time_frames)
        sr: sample rate
    """
    audio, sr = sf.read(wav_path, always_2d=True)
    audio = audio.astype(np.float32)
    
    magnitude, phase = compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    return magnitude, phase, sr


def main(
    dataset_dir="dataset",
    output_dir="dataset_stft",
    n_fft=512,
    hop_length=128,
):
    """
    Convert all audio files in dataset to STFT representation.
    
    Saves:
        - magnitude and phase as .npy files
        - updated metadata with stft paths and parameters
    """
    print(f"Converting dataset from {dataset_dir} to STFT format...")
    print(f"STFT params: n_fft={n_fft}, hop_length={hop_length}")
    
    # Create output directories
    mix_mag_dir = os.path.join(output_dir, "mix_magnitude")
    mix_phase_dir = os.path.join(output_dir, "mix_phase")
    s1_mag_dir = os.path.join(output_dir, "s1_magnitude")
    s1_phase_dir = os.path.join(output_dir, "s1_phase")
    s2_mag_dir = os.path.join(output_dir, "s2_magnitude")
    s2_phase_dir = os.path.join(output_dir, "s2_phase")
    
    for d in [mix_mag_dir, mix_phase_dir, s1_mag_dir, s1_phase_dir, s2_mag_dir, s2_phase_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Read metadata
    meta_path = os.path.join(dataset_dir, "meta.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    # Load all metadata
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    
    print(f"Found {len(metadata)} samples to convert")
    
    # Output metadata
    output_meta_path = os.path.join(output_dir, "meta.jsonl")
    
    with open(output_meta_path, "w", encoding="utf-8") as mf:
        for i, meta in enumerate(tqdm(metadata, desc="Converting to STFT")):
            sample_id = meta["id"]
            
            # Process mix
            mix_mag, mix_phase, sr = process_audio_file(meta["mix_path"], n_fft, hop_length)
            
            # Process s1 (speech)
            s1_mag, s1_phase, _ = process_audio_file(meta["s1_path"], n_fft, hop_length)
            
            # Process s2 (noise)
            s2_mag, s2_phase, _ = process_audio_file(meta["s2_path"], n_fft, hop_length)
            
            # Save as numpy arrays
            mix_mag_path = os.path.join(mix_mag_dir, f"mix_mag_{sample_id}.npy")
            mix_phase_path = os.path.join(mix_phase_dir, f"mix_phase_{sample_id}.npy")
            s1_mag_path = os.path.join(s1_mag_dir, f"s1_mag_{sample_id}.npy")
            s1_phase_path = os.path.join(s1_phase_dir, f"s1_phase_{sample_id}.npy")
            s2_mag_path = os.path.join(s2_mag_dir, f"s2_mag_{sample_id}.npy")
            s2_phase_path = os.path.join(s2_phase_dir, f"s2_phase_{sample_id}.npy")
            
            np.save(mix_mag_path, mix_mag)
            np.save(mix_phase_path, mix_phase)
            np.save(s1_mag_path, s1_mag)
            np.save(s1_phase_path, s1_phase)
            np.save(s2_mag_path, s2_mag)
            np.save(s2_phase_path, s2_phase)
            
            # Update metadata
            meta_out = {
                **meta,  # Keep original metadata
                "stft_n_fft": n_fft,
                "stft_hop_length": hop_length,
                "stft_freq_bins": mix_mag.shape[1],
                "stft_time_frames": mix_mag.shape[2],
                "mix_magnitude_path": mix_mag_path.replace("\\", "/"),
                "mix_phase_path": mix_phase_path.replace("\\", "/"),
                "s1_magnitude_path": s1_mag_path.replace("\\", "/"),
                "s1_phase_path": s1_phase_path.replace("\\", "/"),
                "s2_magnitude_path": s2_mag_path.replace("\\", "/"),
                "s2_phase_path": s2_phase_path.replace("\\", "/"),
            }
            
            mf.write(json.dumps(meta_out) + "\n")
    
    print(f"\nDone!")
    print(f"STFT data saved to: {output_dir}")
    print(f"Metadata: {output_meta_path}")
    print(f"\nSTFT shapes (example from first sample):")
    print(f"  Frequency bins: {meta_out['stft_freq_bins']}")
    print(f"  Time frames: {meta_out['stft_time_frames']}")
    print(f"  Mix magnitude shape: {mix_mag.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert audio dataset to STFT format")
    parser.add_argument("--dataset_dir", default="dataset", help="Input dataset directory")
    parser.add_argument("--output_dir", default="dataset_stft", help="Output STFT directory")
    parser.add_argument("--n_fft", type=int, default=512, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=128, help="Hop length")
    
    args = parser.parse_args()
    
    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
