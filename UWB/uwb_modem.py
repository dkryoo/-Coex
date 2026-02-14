from __future__ import annotations

import numpy as np

from Channel.Rician import apply_distance_rician_channel_with_thermal_noise
from UWB.mms_uwb_packet_mode import get_rsf_base_sequence


def db_to_w(db: float) -> float:
    return 10.0 ** (db / 10.0)


def scale_waveform_to_power_dbw(wf: np.ndarray, tx_power_dbw: float) -> np.ndarray:
    p_target_w = db_to_w(tx_power_dbw)
    p_now = float(np.mean(np.abs(wf) ** 2)) + 1e-30
    return (wf * np.sqrt(p_target_w / p_now)).astype(np.complex128)


def _build_preamble_bits(preamble_symbols: int) -> np.ndarray:
    if preamble_symbols <= 0:
        raise ValueError("preamble_symbols must be > 0")
    rng = np.random.default_rng(2026)
    return rng.integers(0, 2, preamble_symbols, dtype=int)


def build_spreading_code(code_index: int = 15) -> np.ndarray:
    code = get_rsf_base_sequence(code_index=code_index).astype(np.float64)
    code = code / (np.sqrt(np.sum(code**2)) + 1e-30)
    return code


def modulate_uwb_bits(
    bits: np.ndarray,
    code_index: int = 15,
    preamble_symbols: int = 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Simple symbolic UWB-like spread-BPSK modulator.
    Returns (waveform, spreading_code, n_preamble_bits).
    """
    bits = np.asarray(bits, dtype=int).flatten()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be binary (0/1)")
    code = build_spreading_code(code_index=code_index)
    n_chip = len(code)

    pre_bits = _build_preamble_bits(preamble_symbols)
    frame_bits = np.concatenate([pre_bits, bits])
    symbols = 2.0 * frame_bits.astype(np.float64) - 1.0

    chips = np.concatenate([sym * code for sym in symbols])
    return chips.astype(np.complex128), code, int(len(pre_bits))


def _sync_and_estimate_gain(rx: np.ndarray, ref: np.ndarray, search_limit: int | None = None) -> tuple[int, complex]:
    corr = np.convolve(rx, np.conj(ref[::-1]), mode="valid")
    if search_limit is not None:
        lim = max(1, min(int(search_limit), len(corr)))
        idx = int(np.argmax(np.abs(corr[:lim])))
    else:
        idx = int(np.argmax(np.abs(corr)))
    gain = corr[idx] / (np.vdot(ref, ref) + 1e-30)
    return idx, gain


def _estimate_rake_fingers(
    x: np.ndarray,
    start: int,
    code: np.ndarray,
    preamble_symbols: int,
    pre_syms: np.ndarray,
    rake_span_chips: int,
    max_fingers: int,
) -> list[tuple[int, complex]]:
    n_chip = len(code)
    # Estimate one complex weight per delay finger using known preamble (+1 symbols).
    weights: list[tuple[int, complex]] = []
    for d in range(rake_span_chips + 1):
        acc = 0.0 + 0.0j
        used = 0
        for p in range(preamble_symbols):
            s0 = start + p * n_chip + d
            s1 = s0 + n_chip
            if s1 > len(x):
                break
            acc += pre_syms[p] * np.vdot(code, x[s0:s1])
            used += 1
        if used > 0:
            weights.append((d, acc / used))

    if not weights:
        return [(0, 1.0 + 0.0j)]

    weights.sort(key=lambda t: float(np.abs(t[1])), reverse=True)
    return weights[: max(1, int(max_fingers))]


def _refine_start_with_preamble(
    x: np.ndarray,
    coarse_start: int,
    code: np.ndarray,
    pre_syms: np.ndarray,
    search_radius: int,
) -> int:
    n_chip = len(code)
    best_s = max(0, int(coarse_start))
    best_score_abs = -1e30
    s_min = max(0, int(coarse_start) - int(search_radius))
    s_max = max(s_min, int(coarse_start) + int(search_radius))
    for s in range(s_min, s_max + 1):
        score = 0.0
        ok = True
        for p in range(len(pre_syms)):
            s0 = s + p * n_chip
            s1 = s0 + n_chip
            if s1 > len(x):
                ok = False
                break
            z = np.vdot(code, x[s0:s1])
            score += float(pre_syms[p] * np.real(z))
        score_abs = abs(score)
        if ok and score_abs > best_score_abs:
            best_score_abs = score_abs
            best_s = s
    return best_s


def demodulate_uwb_bits(
    rx_wf: np.ndarray,
    n_payload_bits: int,
    code: np.ndarray,
    preamble_symbols: int = 16,
    rake_span_chips: int = 12,
    max_fingers: int = 4,
    channel_taps: np.ndarray | None = None,
) -> np.ndarray:
    rx = rx_wf.astype(np.complex128)
    if channel_taps is not None and len(channel_taps) > 1:
        # RAKE-like front-end: matched filter against channel taps.
        rx = np.convolve(rx, np.conj(np.asarray(channel_taps, dtype=np.complex128)[::-1]), mode="same")

    n_chip = len(code)
    pre_bits = _build_preamble_bits(preamble_symbols)
    pre_syms = 2.0 * pre_bits.astype(np.float64) - 1.0
    pre_ref = np.concatenate([s * code for s in pre_syms]).astype(np.complex128)
    # Expected frame start is near head of waveform (lead zeros + ToA). Restrict search
    # to avoid false locks on random payload correlation peaks.
    sync_search_limit = (preamble_symbols + 12) * n_chip + 1024
    start, gain = _sync_and_estimate_gain(rx, pre_ref, search_limit=sync_search_limit)
    start = _refine_start_with_preamble(
        x=rx,
        coarse_start=start,
        code=code,
        pre_syms=pre_syms,
        search_radius=n_chip,
    )
    # Re-estimate gain at refined start.
    pre_seg = rx[start : start + len(pre_ref)]
    if len(pre_seg) == len(pre_ref):
        gain = np.vdot(pre_ref, pre_seg) / (np.vdot(pre_ref, pre_ref) + 1e-30)
    x = rx / (gain + 1e-30)

    n_total_bits = preamble_symbols + n_payload_bits
    need = n_total_bits * n_chip
    if len(x) < start + need:
        raise RuntimeError("Not enough samples after sync for payload demodulation")

    fingers = _estimate_rake_fingers(
        x=x,
        start=start,
        code=code,
        preamble_symbols=preamble_symbols,
        pre_syms=pre_syms,
        rake_span_chips=rake_span_chips,
        max_fingers=max_fingers,
    )

    # Use known preamble signs to resolve residual polarity ambiguity.
    pre_metrics = []
    for p in range(preamble_symbols):
        base = start + p * n_chip
        metric_c = 0.0 + 0.0j
        for d, w in fingers:
            s0 = base + d
            s1 = s0 + n_chip
            if s1 > len(x):
                continue
            z = np.vdot(code, x[s0:s1])
            metric_c += np.conj(w) * z
        pre_metrics.append(float(pre_syms[p] * np.real(metric_c)))
    polarity = 1.0 if (np.mean(pre_metrics) >= 0.0) else -1.0

    out = np.zeros(n_payload_bits, dtype=int)
    for i in range(n_payload_bits):
        base = start + (preamble_symbols + i) * n_chip
        metric_c = 0.0 + 0.0j
        for d, w in fingers:
            s0 = base + d
            s1 = s0 + n_chip
            if s1 > len(x):
                continue
            z = np.vdot(code, x[s0:s1])
            # MRC-like combining with preamble-estimated channel weights.
            metric_c += np.conj(w) * z
        metric = polarity * np.real(metric_c)
        out[i] = 1 if metric >= 0 else 0
    return out


def run_uwb_modem_ber_test(
    n_frames: int = 100,
    bits_per_frame: int = 128,
    code_index: int = 15,
    fc_hz: float = 6.5e9,
    distance_m: float = 20.0,
    tx_eirp_dbw: float = -20.0,
    nf_db: float = 6.0,
    seed: int = 12345,
    use_channel: bool = True,
) -> dict:
    fs_hz = 499.2e6
    rng = np.random.default_rng(seed)

    total_err = 0
    total_bits = n_frames * bits_per_frame
    frame_ok = 0

    for k in range(n_frames):
        tx_bits = rng.integers(0, 2, bits_per_frame, dtype=int)
        tx_wf, code, pre_n = modulate_uwb_bits(tx_bits, code_index=code_index, preamble_symbols=2)
        tx_wf = scale_waveform_to_power_dbw(tx_wf, tx_power_dbw=tx_eirp_dbw)

        if use_channel:
            rx_wf, ch_info = apply_distance_rician_channel_with_thermal_noise(
                tx_wf=tx_wf,
                fs_hz=fs_hz,
                fc_hz=fc_hz,
                distance_m=distance_m,
                tx_eirp_db=tx_eirp_dbw,
                pathloss_exp=2.0,
                delays_s=(0.0, 6e-9, 12e-9),
                powers_db=(0.0, -6.0, -10.0),
                k_factor_db=8.0,
                nf_db=nf_db,
                temperature_k=290.0,
                noise_ref_bw_hz=20e6,
                rx_lead_zeros=32,
                channel_seed=seed + 1000 + k,
                noise_seed=seed + 2000 + k,
            )
            ch_taps = ch_info.get("h", None)
        else:
            rx_wf = tx_wf.copy()
            ch_taps = None

        try:
            rx_bits = demodulate_uwb_bits(
                rx_wf=rx_wf,
                n_payload_bits=bits_per_frame,
                code=code,
                preamble_symbols=pre_n,
                channel_taps=ch_taps,
            )
        except Exception:
            total_err += bits_per_frame
            continue

        bit_err = int(np.sum(rx_bits != tx_bits))
        total_err += bit_err
        if bit_err == 0:
            frame_ok += 1

    ber = total_err / max(total_bits, 1)
    fer = 1.0 - (frame_ok / max(n_frames, 1))
    return {
        "n_frames": int(n_frames),
        "bits_per_frame": int(bits_per_frame),
        "distance_m": float(distance_m),
        "use_channel": bool(use_channel),
        "ber": float(ber),
        "fer": float(fer),
        "frame_ok_pct": 100.0 * frame_ok / max(n_frames, 1),
    }
