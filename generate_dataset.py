import os
import json
import glob
import numpy as np
import soundfile as sf

from input_sound import load_audio_mono, crop_or_tile, TARGET_SR
from custom_room_mix import generate_stereo_bss_mix


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def apply_snr(s1: np.ndarray, s2: np.ndarray, snr_db: float):
    """
    Scale s2 so that RMS(s1) / RMS(s2_scaled) = 10^(snr_db/20)
    Positive snr_db means s1 louder than s2.
    """
    r1 = rms(s1)
    r2 = rms(s2)
    target_ratio = 10.0 ** (snr_db / 20.0)
    s2_scaled = s2 * (r1 / (r2 * target_ratio + 1e-12))
    return s2_scaled.astype(np.float32)


def load_audio_pool(pool_dir: str, pool_name: str):
    """
    Reads pool_dir/meta.jsonl if exists; otherwise just lists .wav files.
    Returns list of dicts: {"path":..., "id":...}
    """
    meta_path = os.path.join(pool_dir, "meta.jsonl")
    items = []

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                # Use speaker_id for speech, or a generic id for noise
                item_id = str(j.get("speaker_id", j.get("id", "unknown")))
                items.append({"path": j["path"], "id": item_id})
    else:
        wavs = sorted(glob.glob(os.path.join(pool_dir, "*.wav")))
        if not wavs:
            raise RuntimeError(f"No wavs found in {pool_dir}")
        # fallback: id unknown for all
        items = [{"path": p.replace("\\", "/"), "id": "unknown"} for p in wavs]

    if len(items) < 1:
        raise RuntimeError(f"Need at least 1 audio file in {pool_name}.")
    return items


def build_speaker_index(items):
    """Build index of items by speaker/id"""
    id2items = {}
    for it in items:
        item_id = it["id"]
        id2items.setdefault(item_id, []).append(it)
    # keep ids with at least 1 file
    ids = [i for i in id2items.keys() if len(id2items[i]) >= 1]
    return id2items, ids


def main(
    speech_dir="speech_pool",
    noise_dir="noise_pool",
    out_dir="dataset",
    num_mixes=3000,
    mix_seconds=4.0,
    snr_db_range=(-5.0, 5.0),
    seed=1234,
):
    rng = np.random.default_rng(seed)

    # Load speech pool
    speech_items = load_audio_pool(speech_dir, "speech_pool")
    spk2items, speakers = build_speaker_index(speech_items)

    if len(speakers) < 1:
        raise RuntimeError(
            f"Need at least 1 speaker in {speech_dir}. Found speakers={len(speakers)}."
        )

    # Load noise pool
    noise_items = load_audio_pool(noise_dir, "noise_pool")
    
    if len(noise_items) < 1:
        raise RuntimeError(f"Need at least 1 noise file in {noise_dir}.")

    print(f"Loaded {len(speech_items)} speech files from {len(speakers)} speakers")
    print(f"Loaded {len(noise_items)} noise files")

    # Output dirs
    mix_dir = os.path.join(out_dir, "mix")
    s1_dir = os.path.join(out_dir, "s1")  # speech (clean)
    s2_dir = os.path.join(out_dir, "s2")  # noise
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.jsonl")

    n_samples = int(mix_seconds * TARGET_SR)

    with open(meta_path, "w", encoding="utf-8") as mf:
        for i in range(num_mixes):
            # Pick one speech file
            spk = speakers[int(rng.integers(0, len(speakers)))]
            it1 = spk2items[spk][int(rng.integers(0, len(spk2items[spk])))]
            
            # Pick one noise file
            it2 = noise_items[int(rng.integers(0, len(noise_items)))]
            
            p1 = it1["path"]  # speech
            p2 = it2["path"]  # noise

            s1, _ = load_audio_mono(p1, TARGET_SR)  # speech
            s2, _ = load_audio_mono(p2, TARGET_SR)  # noise

            s1_seg = crop_or_tile(s1, n_samples, rng)
            s2_seg = crop_or_tile(s2, n_samples, rng)

            # random SNR (speech over noise)
            snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
            s2_seg = apply_snr(s1_seg, s2_seg, snr_db)

            # stereo room mix
            mix, s1_img, s2_img, meta = generate_stereo_bss_mix(
                s1_seg,
                s2_seg,
                sr=TARGET_SR,
                seed=int(rng.integers(0, 2**31 - 1)),
            )

            mix_path = os.path.join(mix_dir, f"mix_{i:05d}.wav")
            s1_path = os.path.join(s1_dir, f"s1_{i:05d}.wav")
            s2_path = os.path.join(s2_dir, f"s2_{i:05d}.wav")

            sf.write(mix_path, mix, TARGET_SR, subtype="FLOAT")
            sf.write(s1_path, s1_img, TARGET_SR, subtype="FLOAT")
            sf.write(s2_path, s2_img, TARGET_SR, subtype="FLOAT")

            meta_out = {
                "id": f"{i:05d}",
                "mix_path": mix_path.replace("\\", "/"),
                "s1_path": s1_path.replace("\\", "/"),
                "s2_path": s2_path.replace("\\", "/"),
                "mix_seconds": mix_seconds,
                "snr_db_speech_over_noise": snr_db,
                "speech_file": p1.replace("\\", "/"),
                "noise_file": p2.replace("\\", "/"),
                "speaker_id": str(spk),
                "noise_id": str(it2["id"]),
                **meta,
            }
            mf.write(json.dumps(meta_out) + "\n")

            if (i + 1) % 50 == 0:
                print(f"Generated {i+1}/{num_mixes}")

    print(f"\nDone. Dataset written to: {out_dir}")
    print(f"Mixes: {mix_dir}")
    print(f"Sources: {s1_dir} (speech), {s2_dir} (noise)")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main(
        speech_dir="speech_pool",
        noise_dir="noise_pool",
        out_dir="dataset",
        num_mixes=3000,
        mix_seconds=4.0,
        snr_db_range=(-5.0, 5.0),  # SNR of speech over noise
        seed=1234,
    )