import os
import time
import numpy as np
import torch
import soundfile as sf
import argparse

from bss_model import ImprovedBSS


def load_audio(audio_path, target_sr=16000):
    """
    Load audio file and resample if needed.
    Supports multiple formats: wav, mp3, m4a, flac, ogg, etc.
    
    Args:
        audio_path: path to audio file
        target_sr: target sample rate
    
    Returns:
        audio: (n_samples, n_channels) audio array
        sr: sample rate
    """
    print(f"Loading audio: {audio_path}")
    
    # Try using pydub for better format support
    try:
        from pydub import AudioSegment
        
        # Load audio with pydub
        audio_segment = AudioSegment.from_file(audio_path)
        
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        elif audio_segment.sample_width == 1:  # 8-bit
            samples = (samples - 128) / 128.0
        
        # Get sample rate and channels
        sr = audio_segment.frame_rate
        n_channels = audio_segment.channels
        
        # Reshape for stereo
        if n_channels == 2:
            audio = samples.reshape((-1, 2))
        else:
            # Mono to stereo
            audio = np.stack([samples, samples], axis=1)
        
        print(f"Loaded with pydub: {n_channels} channels, {sr} Hz, {audio_segment.sample_width*8}-bit")
        
    except ImportError:
        print("pydub not found, trying soundfile (limited format support)")
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # Convert mono to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=1)
            
            # If already stereo but wrong shape, transpose
            if audio.shape[0] == 2 and audio.shape[1] > audio.shape[0]:
                audio = audio.T
        except Exception as e:
            print(f"\nERROR: Could not load audio file.")
            print(f"File: {audio_path}")
            print(f"Error: {e}")
            print("\nTo support .m4a, .mp3, and other formats, install:")
            print("  pip install pydub")
            print("  sudo apt-get install ffmpeg  (on Ubuntu/Debian)")
            raise
    
    # Resample if needed
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples, axis=0)
            sr = target_sr
        except ImportError:
            print("WARNING: scipy not found, cannot resample. Using original sample rate.")
    
    print(f"Final audio shape: {audio.shape}, Sample rate: {sr} Hz")
    return audio.astype(np.float32), sr


def compute_stft(audio, n_fft=512, hop_length=128):
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
    n_channels = audio.shape[1]
    window = np.hanning(n_fft).astype(np.float32)
    
    magnitudes = []
    phases = []
    
    for ch in range(n_channels):
        audio_ch = audio[:, ch]
        
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


def compute_istft(magnitude, phase, n_fft=512, hop_length=128):
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
    n_channels = magnitude.shape[0]
    freq_bins, n_frames = magnitude.shape[1], magnitude.shape[2]
    
    # Length of output signal
    length = n_fft + hop_length * (n_frames - 1)
    window = np.hanning(n_fft).astype(np.float32)
    
    audio_channels = []
    
    for ch in range(n_channels):
        # Reconstruct complex STFT
        stft_complex = magnitude[ch] * np.exp(1j * phase[ch])
        
        # Initialize output
        y = np.zeros(length, dtype=np.float32)
        norm = np.zeros(length, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop_length
            
            # Reconstruct full spectrum
            if freq_bins == n_fft // 2 + 1:
                full_spectrum = np.concatenate([
                    stft_complex[:, i],
                    np.conj(stft_complex[-2:0:-1, i])
                ])
            else:
                full_spectrum = stft_complex[:, i]
            
            # Inverse FFT
            frame = np.fft.ifft(full_spectrum, n=n_fft).real
            
            # Apply window and overlap-add
            y[start:start + n_fft] += frame * window
            norm[start:start + n_fft] += window ** 2
        
        # Normalize
        norm[norm < 1e-8] = 1.0
        y = y / norm
        
        # Trim padding
        trim = n_fft // 2
        y = y[trim:-trim]
        
        audio_channels.append(y)
    
    # Stack channels
    audio = np.stack(audio_channels, axis=1)  # (n_samples, n_channels)
    
    return audio.astype(np.float32)


def process_in_chunks(audio, model, device, sr, n_fft, hop_length, chunk_duration, overlap_ratio=0.25):
    """
    Process long audio in overlapping chunks to avoid memory issues and smooth transitions.
    
    Args:
        audio: (n_samples, n_channels) audio
        model: trained model
        device: torch device
        sr: sample rate
        n_fft: FFT size
        hop_length: hop length
        chunk_duration: chunk duration in seconds
        overlap_ratio: fraction of chunk to overlap (0.25 = 25% overlap)
    
    Returns:
        speech_audio: separated speech
        noise_audio: separated noise
        times: dict with timing info
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap_ratio))
    n_samples = len(audio)
    
    # Calculate number of chunks with overlap
    n_chunks = int(np.ceil((n_samples - chunk_samples) / hop_samples)) + 1
    
    print(f"Processing {n_chunks} overlapping chunks of {chunk_duration}s each...")
    print(f"Overlap: {overlap_ratio*100:.0f}% ({overlap_ratio*chunk_duration:.1f}s)")
    
    # Initialize output arrays
    speech_output = np.zeros_like(audio)
    noise_output = np.zeros_like(audio)
    weight_sum = np.zeros(n_samples)
    
    total_stft_time = 0
    total_model_time = 0
    total_istft_time = 0
    
    for i in range(n_chunks):
        start_idx = i * hop_samples
        end_idx = min(start_idx + chunk_samples, n_samples)
        
        # Handle last chunk
        if end_idx - start_idx < chunk_samples:
            start_idx = max(0, n_samples - chunk_samples)
            end_idx = n_samples
        
        chunk = audio[start_idx:end_idx]
        chunk_len = len(chunk)
        
        print(f"  Chunk {i+1}/{n_chunks} [{start_idx/sr:.1f}s - {end_idx/sr:.1f}s]... ", end='', flush=True)
        
        # STFT
        stft_start = time.time()
        mix_mag, mix_phase = compute_stft(chunk, n_fft=n_fft, hop_length=hop_length)
        total_stft_time += time.time() - stft_start
        
        # Normalize
        max_val = np.max(mix_mag) + 1e-8
        mix_mag_normalized = mix_mag / max_val
        
        # Model inference
        with torch.no_grad():
            mix_mag_batch = torch.from_numpy(mix_mag_normalized).unsqueeze(0).float().to(device)
            
            model_start = time.time()
            pred_s1_mag, pred_s2_mag, _ = model(mix_mag_batch)
            total_model_time += time.time() - model_start
            
            pred_s1_mag = pred_s1_mag.squeeze(0).cpu().numpy() * max_val
            pred_s2_mag = pred_s2_mag.squeeze(0).cpu().numpy() * max_val
        
        # ISTFT
        istft_start = time.time()
        speech_chunk = compute_istft(pred_s1_mag, mix_phase, n_fft=n_fft, hop_length=hop_length)
        noise_chunk = compute_istft(pred_s2_mag, mix_phase, n_fft=n_fft, hop_length=hop_length)
        total_istft_time += time.time() - istft_start
        
        # Trim to actual chunk length (ISTFT may produce slightly different length)
        actual_len = min(chunk_len, len(speech_chunk), len(noise_chunk))
        speech_chunk = speech_chunk[:actual_len]
        noise_chunk = noise_chunk[:actual_len]
        
        # Create window for smooth blending (Hann window)
        window = np.hanning(actual_len)[:, np.newaxis]  # (actual_len, 1)
        
        # Add to output with windowing
        speech_output[start_idx:start_idx+actual_len] += speech_chunk * window
        noise_output[start_idx:start_idx+actual_len] += noise_chunk * window
        weight_sum[start_idx:start_idx+actual_len] += window.squeeze()
        
        print("Done")
    
    # Normalize by window sum
    weight_sum[weight_sum < 1e-8] = 1.0
    speech_audio = speech_output / weight_sum[:, np.newaxis]
    noise_audio = noise_output / weight_sum[:, np.newaxis]
    
    times = {
        'stft': total_stft_time,
        'model': total_model_time,
        'istft': total_istft_time,
    }
    
    return speech_audio, noise_audio, times


def separate_audio(
    audio_path,
    checkpoint_path,
    output_dir="separated_output",
    n_fft=512,
    hop_length=128,
    target_sr=16000,
    chunk_duration=5.0,  # Default to 5s chunks (close to 4s training)
):
    """
    Separate audio file into speech and noise/music.
    For very long audio files, processes in chunks to avoid memory issues.
    
    Args:
        audio_path: path to input audio file
        checkpoint_path: path to trained model checkpoint
        output_dir: directory to save separated audio
        n_fft: FFT size
        hop_length: hop length for STFT
        target_sr: target sample rate
        chunk_duration: duration of each chunk in seconds (0 = process all at once)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = ImprovedBSS(n_channels=2, n_freq=n_fft//2 + 1, n_sources=2).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # === STEP 1: Load Audio ===
    print("\n" + "="*60)
    print("STEP 1: Loading audio file")
    print("="*60)
    load_start = time.time()
    audio, sr = load_audio(audio_path, target_sr)
    load_time = time.time() - load_start
    print(f"Audio loaded in {load_time*1000:.2f} ms")
    
    # === STEP 2: Convert to STFT ===
    print("\n" + "="*60)
    print("STEP 2: Converting audio to STFT (magnitude + phase)")
    print("="*60)
    
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Check if we need to process in chunks
    if chunk_duration > 0 and audio_duration > chunk_duration:
        print(f"Processing in {chunk_duration}s chunks to manage memory...")
        speech_audio, noise_audio, total_times = process_in_chunks(
            audio, model, device, sr, n_fft, hop_length, chunk_duration
        )
        stft_time = total_times['stft']
        model_time = total_times['model']
        istft_time = total_times['istft']
    else:
        print("Processing entire audio at once...")
        stft_start = time.time()
        mix_mag, mix_phase = compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_time = time.time() - stft_start
        print(f"STFT shape: {mix_mag.shape}")
        print(f"STFT computed in {stft_time*1000:.2f} ms")
        
        # Normalize magnitude
        max_val = np.max(mix_mag) + 1e-8
        mix_mag_normalized = mix_mag / max_val
        
        # === STEP 3: Model Inference (Separation) ===
        print("\n" + "="*60)
        print("STEP 3: Separating sources using model")
        print("="*60)
        
        with torch.no_grad():
            # Prepare input
            mix_mag_batch = torch.from_numpy(mix_mag_normalized).unsqueeze(0).float().to(device)
            
            # Run model
            model_start = time.time()
            pred_s1_mag, pred_s2_mag, masks = model(mix_mag_batch)
            model_time = time.time() - model_start
            
            # Get outputs
            pred_s1_mag = pred_s1_mag.squeeze(0).cpu().numpy() * max_val
            pred_s2_mag = pred_s2_mag.squeeze(0).cpu().numpy() * max_val
        
        print(f"Separation completed in {model_time*1000:.2f} ms")
        
        # === STEP 4: Convert back to audio (ISTFT) ===
        print("\n" + "="*60)
        print("STEP 4: Converting STFT back to audio (ISTFT)")
        print("="*60)
        istft_start = time.time()
        
        speech_audio = compute_istft(pred_s1_mag, mix_phase, n_fft=n_fft, hop_length=hop_length)
        noise_audio = compute_istft(pred_s2_mag, mix_phase, n_fft=n_fft, hop_length=hop_length)
        
        istft_time = time.time() - istft_start
        print(f"ISTFT completed in {istft_time*1000:.2f} ms")
    
    # === STEP 5: Save outputs ===
    print("\n" + "="*60)
    print("STEP 5: Saving separated audio files")
    print("="*60)
    
    # Ensure same length
    min_len = min(len(audio), len(speech_audio), len(noise_audio))
    audio = audio[:min_len]
    speech_audio = speech_audio[:min_len]
    noise_audio = noise_audio[:min_len]
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    mix_path = os.path.join(output_dir, f"{base_name}_original.wav")
    speech_path = os.path.join(output_dir, f"{base_name}_speech.wav")
    noise_path = os.path.join(output_dir, f"{base_name}_noise_music.wav")
    
    # Save files
    sf.write(mix_path, audio, sr, subtype='FLOAT')
    sf.write(speech_path, speech_audio, sr, subtype='FLOAT')
    sf.write(noise_path, noise_audio, sr, subtype='FLOAT')
    
    print(f"\nOriginal:      {mix_path}")
    print(f"Speech:        {speech_path}")
    print(f"Noise/Music:   {noise_path}")
    
    # === Summary ===
    total_time = load_time + stft_time + model_time + istft_time
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Audio Loading:        {load_time*1000:>8.2f} ms")
    print(f"Audio -> STFT:        {stft_time*1000:>8.2f} ms")
    print(f"Model Inference:      {model_time*1000:>8.2f} ms")
    print(f"ISTFT -> Audio:       {istft_time*1000:>8.2f} ms")
    print("-" * 60)
    print(f"Total Processing:     {total_time*1000:>8.2f} ms")
    print(f"Audio Duration:       {len(audio)/sr:>8.2f} s")
    print(f"Real-time Factor:     {(len(audio)/sr)/(total_time):>8.2f}x")
    print("="*60)
    
    return speech_path, noise_path


def main():
    parser = argparse.ArgumentParser(
        description="Separate speech from noise/music in any audio file"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to input audio file (wav, mp3, m4a, flac, etc.)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="separated_output",
        help="Directory to save separated audio files"
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="FFT size for STFT"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Hop length for STFT"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sample rate (Hz)"
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=5.0,
        help="Process audio in chunks of this duration (seconds). Use 0 to process all at once. Default: 5s (recommended for model trained on 4s clips)"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first or provide correct checkpoint path.")
        return
    
    # Run separation
    separate_audio(
        audio_path=args.audio_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_sr=args.sr,
        chunk_duration=args.chunk_duration,
    )


if __name__ == "__main__":
    main()
