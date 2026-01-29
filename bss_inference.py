import os
import json
import time
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from bss_dataset import STFTDataset
from bss_model import ImprovedBSS
from bss_evaluate import istft


def separate_audio(
    model,
    mix_mag,
    mix_phase,
    device,
    n_fft=512,
    hop_length=128,
):
    """
    Separate a single audio mixture.
    
    Args:
        model: trained model
        mix_mag: (n_channels, freq, time) mixture magnitude
        mix_phase: (n_channels, freq, time) mixture phase
        device: torch device
        n_fft, hop_length: STFT parameters
    
    Returns:
        s1_audio: (n_samples, n_channels) separated source 1
        s2_audio: (n_samples, n_channels) separated source 2
        inference_time: processing time in seconds
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension and move to device
        mix_mag_batch = torch.from_numpy(mix_mag).unsqueeze(0).float().to(device)
        
        # Inference
        start_time = time.time()
        pred_s1_mag, pred_s2_mag, _ = model(mix_mag_batch)
        inference_time = time.time() - start_time
        
        # Remove batch dimension
        pred_s1_mag = pred_s1_mag.squeeze(0).cpu().numpy()
        pred_s2_mag = pred_s2_mag.squeeze(0).cpu().numpy()
    
    # Reconstruct audio using ISTFT
    # Using mix phase as approximation (common practice in magnitude-only separation)
    s1_audio = istft(pred_s1_mag, mix_phase, n_fft, hop_length)
    s2_audio = istft(pred_s2_mag, mix_phase, n_fft, hop_length)
    
    return s1_audio, s2_audio, inference_time


def process_dataset_samples(
    model,
    dataset,
    output_dir,
    device,
    n_samples=10,
    n_fft=512,
    hop_length=128,
    sr=16000,
):
    """
    Process samples from dataset and save separated audio.
    
    Args:
        model: trained model
        dataset: STFTDataset
        output_dir: directory to save outputs
        device: torch device
        n_samples: number of samples to process
        n_fft, hop_length: STFT parameters
        sr: sample rate
    """
    # Create output directories
    mix_dir = os.path.join(output_dir, "mix")
    s1_pred_dir = os.path.join(output_dir, "s1_predicted")
    s2_pred_dir = os.path.join(output_dir, "s2_predicted")
    s1_true_dir = os.path.join(output_dir, "s1_true")
    s2_true_dir = os.path.join(output_dir, "s2_true")
    
    for d in [mix_dir, s1_pred_dir, s2_pred_dir, s1_true_dir, s2_true_dir]:
        os.makedirs(d, exist_ok=True)
    
    results = []
    
    print(f"\nProcessing {n_samples} samples...")
    
    for i in tqdm(range(min(n_samples, len(dataset)))):
        sample = dataset[i]
        sample_id = sample['sample_id']
        
        # Get STFT data
        mix_mag = sample['mix_mag'].numpy()
        mix_phase = sample['mix_phase'].numpy()
        s1_mag_true = sample['s1_mag'].numpy()
        s1_phase_true = sample['s1_phase'].numpy()
        s2_mag_true = sample['s2_mag'].numpy()
        s2_phase_true = sample['s2_phase'].numpy()
        
        # Separate audio
        s1_pred, s2_pred, inference_time = separate_audio(
            model, mix_mag, mix_phase, device, n_fft, hop_length
        )
        
        # Reconstruct ground truth
        s1_true = istft(s1_mag_true, s1_phase_true, n_fft, hop_length)
        s2_true = istft(s2_mag_true, s2_phase_true, n_fft, hop_length)
        mix_audio = istft(mix_mag, mix_phase, n_fft, hop_length)
        
        # Ensure same length
        min_len = min(len(mix_audio), len(s1_pred), len(s2_pred), len(s1_true), len(s2_true))
        mix_audio = mix_audio[:min_len]
        s1_pred = s1_pred[:min_len]
        s2_pred = s2_pred[:min_len]
        s1_true = s1_true[:min_len]
        s2_true = s2_true[:min_len]
        
        # Save audio files
        mix_path = os.path.join(mix_dir, f"mix_{sample_id}.wav")
        s1_pred_path = os.path.join(s1_pred_dir, f"s1_pred_{sample_id}.wav")
        s2_pred_path = os.path.join(s2_pred_dir, f"s2_pred_{sample_id}.wav")
        s1_true_path = os.path.join(s1_true_dir, f"s1_true_{sample_id}.wav")
        s2_true_path = os.path.join(s2_true_dir, f"s2_true_{sample_id}.wav")
        
        sf.write(mix_path, mix_audio, sr, subtype='FLOAT')
        sf.write(s1_pred_path, s1_pred, sr, subtype='FLOAT')
        sf.write(s2_pred_path, s2_pred, sr, subtype='FLOAT')
        sf.write(s1_true_path, s1_true, sr, subtype='FLOAT')
        sf.write(s2_true_path, s2_true, sr, subtype='FLOAT')
        
        results.append({
            'sample_id': sample_id,
            'mix_path': mix_path.replace("\\", "/"),
            's1_predicted_path': s1_pred_path.replace("\\", "/"),
            's2_predicted_path': s2_pred_path.replace("\\", "/"),
            's1_true_path': s1_true_path.replace("\\", "/"),
            's2_true_path': s2_true_path.replace("\\", "/"),
            'inference_time_ms': inference_time * 1000,
        })
    
    # Save metadata
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAudio files saved to: {output_dir}")
    print(f"Results metadata: {results_path}")
    
    # Print summary
    avg_time = np.mean([r['inference_time_ms'] for r in results])
    print(f"\nAverage inference time: {avg_time:.2f} ms")
    
    return results


def separate_custom_audio(
    model,
    audio_path,
    output_dir,
    device,
    n_fft=512,
    hop_length=128,
    target_sr=16000,
):
    """
    Separate a custom audio file.
    
    Args:
        model: trained model
        audio_path: path to input audio file
        output_dir: directory to save outputs
        device: torch device
        n_fft, hop_length: STFT parameters
        target_sr: target sample rate
    """
    from input_sound import load_audio_mono
    from convert_to_stft import compute_stft
    
    print(f"\nLoading audio: {audio_path}")
    
    # Load audio
    audio, sr = load_audio_mono(audio_path, target_sr)
    
    # Convert to stereo if mono
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    
    # Compute STFT
    print("Computing STFT...")
    mix_mag, mix_phase = compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    print(f"STFT magnitude shape: {mix_mag.shape}")
    
    # Separate
    print("Separating sources...")
    s1_audio, s2_audio, inference_time = separate_audio(
        model, mix_mag, mix_phase, device, n_fft, hop_length
    )
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    s1_path = os.path.join(output_dir, f"{base_name}_speech.wav")
    s2_path = os.path.join(output_dir, f"{base_name}_noise.wav")
    mix_path = os.path.join(output_dir, f"{base_name}_mix.wav")
    
    sf.write(mix_path, audio, sr, subtype='FLOAT')
    sf.write(s1_path, s1_audio, sr, subtype='FLOAT')
    sf.write(s2_path, s2_audio, sr, subtype='FLOAT')
    
    print(f"\nSeparation complete!")
    print(f"Processing time: {inference_time*1000:.2f} ms")
    print(f"\nOutput files:")
    print(f"  Mix: {mix_path}")
    print(f"  Speech (S1): {s1_path}")
    print(f"  Noise (S2): {s2_path}")
    
    return s1_path, s2_path


def main(
    checkpoint_path="checkpoints/best_model.pth",
    meta_path="dataset_stft/meta.jsonl",
    output_dir="inference_outputs",
    n_samples=10,
    custom_audio=None,
    n_fft=512,
    hop_length=128,
    sr=16000,
):
    """Main inference function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    if custom_audio:
        # Process custom audio file
        separate_custom_audio(
            model=model,
            audio_path=custom_audio,
            output_dir=output_dir,
            device=device,
            n_fft=n_fft,
            hop_length=hop_length,
            target_sr=sr,
        )
    else:
        # Process dataset samples
        print(f"\nLoading dataset from {meta_path}...")
        dataset = STFTDataset(meta_path, supervised_ratio=1.0, normalize=True)
        
        process_dataset_samples(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            device=device,
            n_samples=n_samples,
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sr,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference for BSS model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--meta_path", default="dataset_stft/meta.jsonl")
    parser.add_argument("--output_dir", default="inference_outputs")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of dataset samples to process")
    parser.add_argument("--custom_audio", type=str, default=None,
                        help="Path to custom audio file to separate")
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--sr", type=int, default=16000)
    
    args = parser.parse_args()
    
    main(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        custom_audio=args.custom_audio,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sr=args.sr,
    )
