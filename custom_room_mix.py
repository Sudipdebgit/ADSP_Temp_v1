import numpy as np
import pyroomacoustics as pra
import warnings


def _rand_pos_in_room(room_dim: np.ndarray, buffer_m: float, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(buffer_m, d - buffer_m) for d in room_dim], dtype=np.float32)


def _try_make_room(
    sr: int,
    room_dim_range,
    rt60_range,
    rng: np.random.Generator,
    max_attempts: int = 200,
):
    """
    Try sampling (room_dim, rt60) until pra.inverse_sabine succeeds and returns valid absorption.
    Returns: (room, room_dim, rt60, e_abs)
    """
    min_dim = np.array(room_dim_range[0], dtype=np.float32)
    max_dim = np.array(room_dim_range[1], dtype=np.float32)

    for _ in range(max_attempts):
        room_dim = rng.uniform(min_dim, max_dim).astype(np.float32)
        rt60 = float(rng.uniform(rt60_range[0], rt60_range[1]))

        try:
            e_abs_raw = pra.inverse_sabine(rt60, room_dim)
            e_abs = e_abs_raw[0] if isinstance(e_abs_raw, tuple) else e_abs_raw
            e_abs = float(e_abs)

            # must be in (0,1)
            if not (0.0 < e_abs < 1.0):
                continue

            room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_abs))
            return room, room_dim, rt60, e_abs

        except ValueError:
            # This is the exact error you saw. Just retry.
            continue

    raise RuntimeError(
        f"Failed to create valid room after {max_attempts} attempts. "
        f"Try narrowing rt60_range or room_dim_range."
    )


def _convolve(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return np.convolve(signal, rir).astype(np.float32)


def generate_stereo_bss_mix(
    s1_dry: np.ndarray,
    s2_dry: np.ndarray,
    sr: int = 16000,
    room_dim_range=((5.0, 5.0, 2.5), (10.0, 10.0, 3.5)),
    rt60_range=(0.1, 0.6),
    mic_spacing_range=(0.08, 0.30),
    wall_buffer=0.5,
    min_src_dist=1.0,
    target_snr_db_range=(-5.0, 10.0),
    seed: int | None = None,
):
    """
    Returns:
      mix_stereo: (N, 2)
      s1_stereo:  (N, 2)  source-1 image at both mics
      s2_stereo:  (N, 2)  source-2 image at both mics
      meta: dict with room+placement+snr
    """
    rng = np.random.default_rng(seed)

    # 1) Robust room creation (retries until inverse_sabine is valid)
    room, room_dim, rt60, e_abs = _try_make_room(
        sr=sr,
        room_dim_range=room_dim_range,
        rt60_range=rt60_range,
        rng=rng,
        max_attempts=300,
    )

    # 2) Mic positions (2-mic linear array)
    mic_center = _rand_pos_in_room(room_dim, wall_buffer, rng)
    spacing = float(rng.uniform(mic_spacing_range[0], mic_spacing_range[1]))

    mic1 = mic_center.copy()
    mic2 = mic_center.copy()
    mic2[0] = np.clip(mic1[0] + spacing, wall_buffer, room_dim[0] - wall_buffer)

    room.add_microphone_array(pra.MicrophoneArray(np.c_[mic1, mic2], fs=sr))

    # 3) Source positions (keep distance)
    src1 = _rand_pos_in_room(room_dim, wall_buffer, rng)
    src2 = _rand_pos_in_room(room_dim, wall_buffer, rng)
    tries = 0
    while np.linalg.norm(src1 - src2) < min_src_dist and tries < 80:
        src2 = _rand_pos_in_room(room_dim, wall_buffer, rng)
        tries += 1
    if tries >= 80:
        warnings.warn("Could not enforce min_src_dist; continuing anyway.")

    # Add sources only to compute RIRs
    room.add_source(src1, signal=None)
    room.add_source(src2, signal=None)
    room.compute_rir()

    rir_m1_s1 = np.array(room.rir[0][0], dtype=np.float32)
    rir_m2_s1 = np.array(room.rir[1][0], dtype=np.float32)
    rir_m1_s2 = np.array(room.rir[0][1], dtype=np.float32)
    rir_m2_s2 = np.array(room.rir[1][1], dtype=np.float32)

    # 4) Convolve dry sources -> source images at mics
    s1_m1 = _convolve(s1_dry, rir_m1_s1)
    s1_m2 = _convolve(s1_dry, rir_m2_s1)
    s2_m1 = _convolve(s2_dry, rir_m1_s2)
    s2_m2 = _convolve(s2_dry, rir_m2_s2)

    # 5) Match length
    N = min(len(s1_m1), len(s1_m2), len(s2_m1), len(s2_m2))
    s1_m1, s1_m2, s2_m1, s2_m2 = s1_m1[:N], s1_m2[:N], s2_m1[:N], s2_m2[:N]

    # 6) Enforce target SNR at mic1
    target_snr_db = float(rng.uniform(target_snr_db_range[0], target_snr_db_range[1]))

    p1 = float(np.mean(s1_m1**2) + 1e-12)
    p2 = float(np.mean(s2_m1**2) + 1e-12)
    gain = np.sqrt(p1 / (p2 * (10.0 ** (target_snr_db / 10.0))))

    s2_m1 *= gain
    s2_m2 *= gain

    # 7) Mix
    mix_m1 = s1_m1 + s2_m1
    mix_m2 = s1_m2 + s2_m2

    mix_stereo = np.stack([mix_m1, mix_m2], axis=1).astype(np.float32)
    s1_stereo = np.stack([s1_m1, s1_m2], axis=1).astype(np.float32)
    s2_stereo = np.stack([s2_m1, s2_m2], axis=1).astype(np.float32)

    # 8) Normalize to avoid clipping
    peak = float(np.max(np.abs(mix_stereo)) + 1e-12)
    if peak > 0.99:
        scale = 0.99 / peak
        mix_stereo *= scale
        s1_stereo *= scale
        s2_stereo *= scale

    meta = {
        "sr": sr,
        "room_dim": room_dim.tolist(),
        "rt60": rt60,
        "absorption": e_abs,
        "mic1": mic1.tolist(),
        "mic2": mic2.tolist(),
        "src1": src1.tolist(),
        "src2": src2.tolist(),
        "mic_spacing": spacing,
        "target_snr_db_at_mic1": target_snr_db,
        "gain_applied_to_s2": float(gain),
    }

    return mix_stereo, s1_stereo, s2_stereo, meta
