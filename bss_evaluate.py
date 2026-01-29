import os
import json
import time
import numpy as np
import torch
from tqdm import tqdm
import mir_eval

from bss_dataset import STFTDataset
from bss_model import ImprovedBSS


def istft(magnitude, phase, n_fft=512, hop_length=128):
    """
    Inverse STFT to reconstruct time-domain signal.
    
    Args:
        magnitude: (n_channels, freq_bins, time_frames)
        phase: (n_channels, freq_bins, time_frames)
        n_fft: FFT size
        hop_length: hop length
    
    Returns:
        audio: (n_samples, n_channels)
    """
    if isinstance(magnitude, torch.Tensor):
        magnitude = magnitude.cpu().numpy()
    if isinstance(phase, torch.Tensor):
        phase = phase.cpu().numpy()
    
    n_channels = magnitude.shape[0]
    
    # Reconstruct complex STFT
    stft_complex = magnitude * np.exp(1j * phase)
    
    # ISTFT for each channel
    audio_channels = []
    window = np.hanning(n_fft).astype(np.float32)
    
    for ch in range(n_channels):
        audio = istft_single_channel(
            stft_complex[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window
        )
        audio_channels.append(audio)
    
    # Stack channels
    audio = np.stack(audio_channels, axis=1)  # (n_samples, n_channels)
    
    return audio


def istft_single_channel(stft_matrix, n_fft, hop_length, window):
    """ISTFT for single channel"""
    freq_bins, n_frames = stft_matrix.shape
    
    # Length of output signal
    length = n_fft + hop_length * (n_frames - 1)
    
    # Initialize output
    y = np.zeros(length, dtype=np.float32)
    norm = np.zeros(length, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        
        # IFFT (use full spectrum by mirroring)
        if freq_bins == n_fft // 2 + 1:
            # Reconstruct full spectrum
            full_spectrum = np.concatenate([
                stft_matrix[:, i],
                np.conj(stft_matrix[-2:0:-1, i])
            ])
        else:
            full_spectrum = stft_matrix[:, i]
        
        # Inverse FFT
        frame = np.fft.ifft(full_spectrum, n=n_fft).real
        
        # Apply window and overlap-add
        y[start:start + n_fft] += frame * window
        norm[start:start + n_fft] += window ** 2
    
    # Normalize
    norm[norm < 1e-8] = 1.0
    y = y / norm
    
    # Trim to original length (remove padding)
    trim = n_fft // 2
    y = y[trim:-trim]
    
    return y.astype(np.float32)


def compute_bss_metrics(reference, estimated, sr=16000):
    """
    Compute BSS metrics: SDR, SIR, SAR
    
    Args:
        reference: (n_samples, n_sources) ground truth sources
        estimated: (n_samples, n_sources) estimated sources
        sr: sample rate
    
    Returns:
        dict with SDR, SIR, SAR for each source
    """
    # mir_eval expects (nsrc, nsamples)
    reference = reference.T  # (n_sources, n_samples)
    estimated = estimated.T
    
    # Compute metrics
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference,
        estimated,
        compute_permutation=True
    )
    
    return {
        'sdr': sdr,  # Signal-to-Distortion Ratio
        'sir': sir,  # Signal-to-Interference Ratio
        'sar': sar,  # Signal-to-Artifacts Ratio
        'permutation': perm,
    }


def stft_from_audio(audio, n_fft=512, hop_length=128):
    """
    Compute STFT from time-domain audio.
    
    Args:
        audio: (n_samples, n_channels) time-domain audio
        n_fft: FFT size
        hop_length: hop length
    
    Returns:
        magnitude: (n_channels, freq_bins, time_frames)
        phase: (n_channels, freq_bins, time_frames)
    """
    n_channels = audio.shape[1] if audio.ndim > 1 else 1
    window = np.hanning(n_fft).astype(np.float32)
    
    magnitudes = []
    phases = []
    
    for ch in range(n_channels):
        audio_ch = audio[:, ch] if audio.ndim > 1 else audio
        
        # Compute number of frames
        n_frames = 1 + (len(audio_ch) - n_fft) // hop_length
        
        # Pad audio
        pad_len = n_fft // 2
        audio_padded = np.pad(audio_ch, (pad_len, pad_len), mode='reflect')
        
        # Initialize STFT matrix
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
        
        # Compute STFT
        for i in range(n_frames):
            start = i * hop_length
            frame = audio_padded[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame, n=n_fft)
            stft_matrix[:, i] = spectrum
        
        magnitudes.append(np.abs(stft_matrix))
        phases.append(np.angle(stft_matrix))
    
    magnitude = np.stack(magnitudes, axis=0)  # (n_channels, freq, time)
    phase = np.stack(phases, axis=0)
    
    return magnitude.astype(np.float32), phase.astype(np.float32)


def evaluate_model(
    model,
    dataset,
    device,
    n_samples=None,
    n_fft=512,
    hop_length=128,
):
    """
    Evaluate model on dataset and compute BSS metrics.
    Also measures end-to-end processing time: audio -> STFT -> model -> ISTFT -> audio
    
    Args:
        model: trained model
        dataset: STFTDataset
        device: torch device
        n_samples: number of samples to evaluate (None = all)
        n_fft, hop_length: STFT parameters
    
    Returns:
        metrics: dict with average SDR, SIR, SAR, and timing info
        results: list of per-sample results
    """
    model.eval()
    
    if n_samples is None:
        n_samples = len(dataset)
    
    all_sdr = []
    all_sir = []
    all_sar = []
    all_times = []
    all_stft_times = []
    all_model_times = []
    all_istft_times = []
    all_total_times = []
    results = []
    
    print(f"\nEvaluating on {n_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples)):
            sample = dataset[i]
            
            # Get ground truth STFT for comparison
            mix_phase = sample['mix_phase'].to(device)
            s1_mag_true = sample['s1_mag'].to(device)
            s1_phase_true = sample['s1_phase'].to(device)
            s2_mag_true = sample['s2_mag'].to(device)
            s2_phase_true = sample['s2_phase'].to(device)
            
            # === FULL PIPELINE TIMING ===
            # Start with ground truth audio
            s1_reference = istft(s1_mag_true, s1_phase_true, n_fft, hop_length)
            s2_reference = istft(s2_mag_true, s2_phase_true, n_fft, hop_length)
            mix_audio = s1_reference + s2_reference
            
            # Ensure stereo
            if mix_audio.ndim == 1:
                mix_audio = np.stack([mix_audio, mix_audio], axis=1)
            
            # === 1. Audio to STFT ===
            stft_start = time.time()
            mix_mag_from_audio, mix_phase_from_audio = stft_from_audio(mix_audio, n_fft, hop_length)
            stft_time = time.time() - stft_start
            
            # Normalize (same as dataset)
            max_val = np.max(mix_mag_from_audio) + 1e-8
            mix_mag_normalized = mix_mag_from_audio / max_val
            
            # === 2. Model Inference ===
            mix_mag_batch = torch.from_numpy(mix_mag_normalized).unsqueeze(0).float().to(device)
            
            model_start = time.time()
            pred_s1_mag, pred_s2_mag, _ = model(mix_mag_batch)
            model_time = time.time() - model_start
            
            # Remove batch dimension and denormalize
            pred_s1_mag = pred_s1_mag.squeeze(0).cpu().numpy() * max_val
            pred_s2_mag = pred_s2_mag.squeeze(0).cpu().numpy() * max_val
            
            # === 3. ISTFT to Audio ===
            istft_start = time.time()
            s1_estimated = istft(pred_s1_mag, mix_phase_from_audio, n_fft, hop_length)
            s2_estimated = istft(pred_s2_mag, mix_phase_from_audio, n_fft, hop_length)
            istft_time = time.time() - istft_start
            
            # Total end-to-end time
            total_time = stft_time + model_time + istft_time
            
            # Ensure same length
            min_len = min(len(s1_estimated), len(s1_reference))
            s1_estimated = s1_estimated[:min_len]
            s2_estimated = s2_estimated[:min_len]
            s1_reference = s1_reference[:min_len]
            s2_reference = s2_reference[:min_len]
            
            # Average over channels if stereo
            if s1_estimated.ndim > 1:
                s1_estimated = s1_estimated.mean(axis=1)
                s2_estimated = s2_estimated.mean(axis=1)
                s1_reference = s1_reference.mean(axis=1)
                s2_reference = s2_reference.mean(axis=1)
            
            # Stack sources
            reference = np.stack([s1_reference, s2_reference], axis=1)  # (n_samples, 2)
            estimated = np.stack([s1_estimated, s2_estimated], axis=1)
            
            # Compute BSS metrics
            try:
                metrics = compute_bss_metrics(reference, estimated)
                
                # Store results
                sdr_s1, sdr_s2 = metrics['sdr']
                sir_s1, sir_s2 = metrics['sir']
                sar_s1, sar_s2 = metrics['sar']
                
                all_sdr.extend([sdr_s1, sdr_s2])
                all_sir.extend([sir_s1, sir_s2])
                all_sar.extend([sar_s1, sar_s2])
                all_stft_times.append(stft_time)
                all_model_times.append(model_time)
                all_istft_times.append(istft_time)
                all_total_times.append(total_time)
                
                results.append({
                    'sample_id': sample['sample_id'],
                    'sdr_s1': float(sdr_s1),
                    'sdr_s2': float(sdr_s2),
                    'sir_s1': float(sir_s1),
                    'sir_s2': float(sir_s2),
                    'sar_s1': float(sar_s1),
                    'sar_s2': float(sar_s2),
                    'stft_time_ms': stft_time * 1000,
                    'model_time_ms': model_time * 1000,
                    'istft_time_ms': istft_time * 1000,
                    'total_time_ms': total_time * 1000,
                })
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
    
    # Compute average metrics
    avg_metrics = {
        'sdr_mean': float(np.mean(all_sdr)),
        'sdr_std': float(np.std(all_sdr)),
        'sir_mean': float(np.mean(all_sir)),
        'sir_std': float(np.std(all_sir)),
        'sar_mean': float(np.mean(all_sar)),
        'sar_std': float(np.std(all_sar)),
        'avg_stft_time_ms': float(np.mean(all_stft_times) * 1000),
        'avg_model_time_ms': float(np.mean(all_model_times) * 1000),
        'avg_istft_time_ms': float(np.mean(all_istft_times) * 1000),
        'avg_total_time_ms': float(np.mean(all_total_times) * 1000),
        'std_total_time_ms': float(np.std(all_total_times) * 1000),
    }
    
    return avg_metrics, results


def main(
    checkpoint_path="checkpoints/best_model.pth",
    meta_path="dataset_stft/meta.jsonl",
    output_path="evaluation_results.json",
    n_samples=100,
    n_fft=512,
    hop_length=128,
):
    """Main evaluation function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset
    print(f"\nLoading dataset from {meta_path}...")
    dataset = STFTDataset(meta_path, supervised_ratio=1.0, normalize=True)
    
    # Evaluate
    avg_metrics, results = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        n_samples=n_samples,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nSDR: {avg_metrics['sdr_mean']:.2f} ± {avg_metrics['sdr_std']:.2f} dB")
    print(f"SIR: {avg_metrics['sir_mean']:.2f} ± {avg_metrics['sir_std']:.2f} dB")
    print(f"SAR: {avg_metrics['sar_mean']:.2f} ± {avg_metrics['sar_std']:.2f} dB")
    print(f"\n{'TIMING BREAKDOWN (End-to-End Pipeline)':^60}")
    print("-" * 60)
    print(f"Audio -> STFT:        {avg_metrics['avg_stft_time_ms']:>8.2f} ms")
    print(f"Model Inference:      {avg_metrics['avg_model_time_ms']:>8.2f} ms")
    print(f"ISTFT -> Audio:       {avg_metrics['avg_istft_time_ms']:>8.2f} ms")
    print("-" * 60)
    print(f"Total Processing:     {avg_metrics['avg_total_time_ms']:>8.2f} ± {avg_metrics['std_total_time_ms']:.2f} ms")
    print("="*60)
    
    # Save results
    output_data = {
        'average_metrics': avg_metrics,
        'per_sample_results': results,
        'config': {
            'checkpoint_path': checkpoint_path,
            'meta_path': meta_path,
            'n_samples': n_samples,
            'n_fft': n_fft,
            'hop_length': hop_length,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BSS model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--meta_path", default="dataset_stft/meta.jsonl")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta_path,
        output_path=args.output,
        n_samples=args.n_samples,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
