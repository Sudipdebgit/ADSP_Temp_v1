import os
import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


TARGET_SR = 16000


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # average channels -> mono
    return np.mean(x, axis=1)


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load audio as mono float32 at target sample rate.
    If resampling is needed, requires librosa.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    x, sr = sf.read(path, always_2d=False)
    x = _to_mono(x).astype(np.float32)

    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                f"File {path} has sr={sr}, need {target_sr} but librosa is not installed."
            )
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr

    # Normalize safely (avoid clipping, keep relative dynamics)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 1.0:
        x = x / peak

    return x, sr


def crop_or_tile(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Make signal exactly 'length' samples.
    If too long -> random crop. If too short -> tile and crop.
    """
    if len(x) == length:
        return x

    if len(x) > length:
        start = int(rng.integers(0, len(x) - length + 1))
        return x[start:start + length]

    # too short
    reps = int(np.ceil(length / len(x)))
    xt = np.tile(x, reps)
    return xt[:length]


if __name__ == "__main__":
    # Edit these paths to your files
    male_path = "male.wav"
    female_path = "female.wav"

    male, sr1 = load_audio_mono(male_path)
    female, sr2 = load_audio_mono(female_path)

    print("Loaded:")
    print(" male:", male.shape, sr1)
    print(" female:", female.shape, sr2)
