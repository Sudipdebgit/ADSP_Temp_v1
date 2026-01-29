import os
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import mir_eval
from scipy.ndimage import gaussian_filter, median_filter

from bss_dataset import STFTDataset
from bss_model import ImprovedBSS
from bss_evaluate import istft, stft_from_audio

def apply_full_postprocessing(
    magnitude, 
    phase, 
    gaussian_sigma, 
    noise_floor_db,
    spectral_gate_db,
    enable_wiener=True
):
    """
    Replicates the exact pipeline from bss_evaluate.py for the grid search.
    """
    processed_mag = magnitude.copy()
    
    # 1. Wiener-like filtering (Ratio Masking) - Kept as base 
    # (This provides the initial separation quality)
    if enable_wiener:
        smooth_mag = processed_mag.copy()
        for ch in range(magnitude.shape[0]):
            # Minimal smoothing for the noise estimate
            smooth_mag[ch] = gaussian_filter(magnitude[ch], sigma=0.5)
        
        noise_estimate = np.percentile(smooth_mag, 10, axis=(1, 2), keepdims=True)
        gain = (smooth_mag ** 2) / ((smooth_mag ** 2) + (noise_estimate ** 2) + 1e-8)
        processed_mag = processed_mag * gain
    
    # 2. Gaussian Smoothing (User Parameter 1)
    # Reduces high-freq artifacts -> Increases SAR
    if gaussian_sigma > 0:
        for ch in range(magnitude.shape[0]):
            processed_mag[ch] = gaussian_filter(processed_mag[ch], sigma=gaussian_sigma)
    
    # 3. Noise Floor (User Parameter 2)
    # Aggressive noise suppression
    if noise_floor_db is not None:
        mag_db = 20 * np.log10(processed_mag + 1e-8)
        mag_db = np.maximum(mag_db, noise_floor_db)
        processed_mag = 10 ** (mag_db / 20)
    
    # 4. Spectral Gating (User Parameter 3)
    # Suppresses quiet components relative to peak
    if spectral_gate_db is not None:
        max_mag = np.max(processed_mag, axis=1, keepdims=True)
        max_mag_db = 20 * np.log10(max_mag + 1e-8)
        mag_db = 20 * np.log10(processed_mag + 1e-8)
        
        # Soft transition gate
        gate_mask = mag_db > (max_mag_db + spectral_gate_db)
        transition_db = 5.0
        gate_db_diff = mag_db - (max_mag_db + spectral_gate_db)
        smooth_gate = np.clip((gate_db_diff + transition_db) / (2 * transition_db), 0, 1)
        
        processed_mag = processed_mag * smooth_gate

    return processed_mag

def precompute_model_outputs(model, dataset, device, n_samples):
    """Run model once to get raw predictions."""
    print(f"\nPre-computing raw model outputs for {n_samples} samples...")
    cache = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(min(n_samples, len(dataset)))):
            sample = dataset[i]
            
            # Ground Truth (Audio)
            s1_ref = istft(sample['s1_mag'], sample['s1_phase'])
            s2_ref = istft(sample['s2_mag'], sample['s2_phase'])
            mix_audio = s1_ref + s2_ref
            
            # Stereo check
            if mix_audio.ndim == 1:
                mix_audio = np.stack([mix_audio, mix_audio], axis=1)
                
            # Compute STFT for input
            mix_mag, mix_phase = stft_from_audio(mix_audio)
            
            # Model Inference
            max_val = np.max(mix_mag) + 1e-8
            mix_norm = torch.from_numpy(mix_mag / max_val).unsqueeze(0).float().to(device)
            p1, p2, _ = model(mix_norm)
            
            # Store Denormalized Raw Predictions
            cache.append({
                'p1_raw': p1.squeeze(0).cpu().numpy() * max_val,
                'p2_raw': p2.squeeze(0).cpu().numpy() * max_val,
                'mix_phase': mix_phase,
                'ref_s1': s1_ref,
                'ref_s2': s2_ref
            })
    return cache

def run_grid_search(checkpoint_path, meta_path, n_samples=10):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = STFTDataset(meta_path, supervised_ratio=1.0, normalize=True)
    
    # Precompute
    cache = precompute_model_outputs(model, dataset, device, n_samples)
    
    # --- SWEEP RANGES (From User Request) ---
    sigma_vals = [1.5, 2.0, 2.5, 3.0]      # Smoothing
    floor_vals = [-40, -35, -30]           # Noise Floor dB
    gate_vals  = [-30, -25, -20]           # Spectral Gate dB
    # ----------------------------------------
    
    results = []
    print("\nStarting Grid Search...")
    print(f"{'Sigma':<6} | {'Floor':<6} | {'Gate':<6} | {'SIR':<8} | {'SAR':<8}")
    print("-" * 50)
    
    for sigma in sigma_vals:
        for floor in floor_vals:
            for gate in gate_vals:
                
                curr_sir, curr_sar = [], []
                
                for item in cache:
                    # Apply Pipeline to BOTH sources
                    est_s1_mag = apply_full_postprocessing(
                        item['p1_raw'], item['mix_phase'], 
                        sigma, floor, gate
                    )
                    est_s2_mag = apply_full_postprocessing(
                        item['p2_raw'], item['mix_phase'], 
                        sigma, floor, gate
                    )
                    
                    # ISTFT
                    audio_s1 = istft(est_s1_mag, item['mix_phase'])
                    audio_s2 = istft(est_s2_mag, item['mix_phase'])
                    
                    # Evaluation Prep
                    min_len = min(len(audio_s1), len(item['ref_s1']))
                    est = np.stack([audio_s1[:min_len].mean(1), audio_s2[:min_len].mean(1)])
                    ref = np.stack([item['ref_s1'][:min_len].mean(1), item['ref_s2'][:min_len].mean(1)])
                    
                    try:
                        _, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                            ref, est, compute_permutation=False
                        )
                        curr_sir.append(np.mean(sir))
                        curr_sar.append(np.mean(sar))
                    except:
                        continue

                # Aggregate
                m_sir, m_sar = np.mean(curr_sir), np.mean(curr_sar)
                results.append({
                    'Sigma': sigma, 'Floor': floor, 'Gate': gate,
                    'SIR': m_sir, 'SAR': m_sar
                })
                print(f"{sigma:<6.1f} | {floor:<6} | {gate:<6} | {m_sir:>8.2f} | {m_sar:>8.2f}")

    # Analysis
    df = pd.DataFrame(results)
    
    # Find best SAR > 30 while keeping reasonable SIR
    # Or simply sort by Distance to (SIR 9, SAR 33)
    target_sir, target_sar = 9.0, 33.0
    df['Dist'] = np.sqrt((df['SIR']-target_sir)**2 + (df['SAR']-target_sar)**2)
    best = df.sort_values('Dist').iloc[0]
    
    print("\n" + "="*50)
    print(f"BEST CONFIGURATION (Closest to SAR {target_sar}dB)")
    print("="*50)
    print(f"Sigma: {best['Sigma']}")
    print(f"Noise Floor: {best['Floor']} dB")
    print(f"Spectral Gate: {best['Gate']} dB")
    print(f"Result -> SIR: {best['SIR']:.2f} | SAR: {best['SAR']:.2f}")
    
    df.to_csv("grid_search_results_v2.csv", index=False)
    print("\nSaved to grid_search_results_v2.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--meta_path", default="dataset_stft/meta.jsonl")
    parser.add_argument("--n_samples", type=int, default=15)
    args = parser.parse_args()
    
    run_grid_search(args.checkpoint, args.meta_path, args.n_samples)
