import os
import random
import argparse
import numpy as np
import soundfile as sf
import io

from datasets import load_dataset, Audio


def main(out_dir="speech_pool", num_files=200, seed=1234, split="test.clean"):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Streaming dataset
    ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)

    # IMPORTANT: do NOT let datasets decode audio (avoids torchcodec/ffmpeg)
    ds = ds.cast_column("audio", Audio(decode=False))

    meta_path = os.path.join(out_dir, "meta.jsonl")
    written = 0

    with open(meta_path, "w", encoding="utf-8") as mf:
        for ex in ds:
            audio = ex["audio"]
            audio_bytes = audio.get("bytes", None)
            if audio_bytes is None:
                # In rare cases streaming may not provide bytes. Skip.
                continue

            # Decode using soundfile directly from bytes (libsndfile supports FLAC)
            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)

            # resample to 16k if needed
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
                sr = 16000

            peak = float(np.max(np.abs(wav)) + 1e-12)
            if peak > 0.99:
                wav = 0.99 * wav / peak

            utt_id = ex.get("id", f"{written}")
            spk_id = ex.get("speaker_id", "unknown")

            out_path = os.path.join(out_dir, f"ls_{written:05d}_spk{spk_id}_{utt_id}.wav")
            sf.write(out_path, wav, sr, subtype="FLOAT")

            mf.write(
                '{"path":"%s","speaker_id":"%s","utt_id":"%s","split":"%s"}\n'
                % (out_path.replace("\\", "/"), str(spk_id), str(utt_id), split)
            )

            written += 1
            if written % 25 == 0:
                print(f"Saved {written}/{num_files}")

            if written >= num_files:
                break

    print(f"\nDone. Saved {written} wav files into {out_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="speech_pool")
    ap.add_argument("--num_files", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--split", default="test.clean")
    args = ap.parse_args()
    main(args.out_dir, args.num_files, args.seed, args.split)
