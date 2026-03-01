from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

from Channel.Interference import (
    apply_fir,
    alias_frequency,
    apply_soft_clipping,
    freq_shift,
    resample_complex_linear,
    welch_psd_db_per_hz,
    save_psd_triplet,
)
from Channel.Selectivity import apply_fir_same, design_lowpass_fir, gain_at_offset_db
from Channel.Rician import apply_distance_rician_channel
from Channel.Thermal_noise import thermal_noise_power_w
from simulation.utils.power import dbw_to_dbm, fmt_dbw_dbm, mean_power_w
from UWB.mms_uwb_packet_mode import C_MPS, MmsUwbConfig, RSTU_S, build_mms_uwb_fragments, get_rsf_base_sequence


def _load_wifi_tx_class():
    wifi_path = Path(__file__).resolve().parents[2] / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("wifi_tx_module_mms_perf", wifi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Wi-Fi TX module from {wifi_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WiFiOFDMTx


def _db_to_w(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _w_to_db(w: float) -> float:
    return 10.0 * np.log10(max(w, 1e-30))


def dbw_to_w(p_dbw: float) -> float:
    return _db_to_w(p_dbw)


def w_to_dbw(p_w: float) -> float:
    return _w_to_db(p_w)


def power_w(x: np.ndarray) -> float:
    return mean_power_w(x)


def power_dbw(x: np.ndarray) -> float:
    return w_to_dbw(power_w(x) + 1e-30)


def scale_to_power_dbw(x: np.ndarray, p_dbw: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    p_now = power_w(x) + 1e-30
    p_tgt = dbw_to_w(float(p_dbw))
    return (x * np.sqrt(p_tgt / p_now)).astype(np.complex128)


def crc16_802154(data: bytes) -> int:
    """IEEE 802.15.4 reflected CRC-16 (poly 0x8408, init 0x0000)."""
    crc = 0x0000
    for b in data:
        crc ^= int(b) & 0xFF
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc & 0xFFFF


def _append_fcs(payload: bytes) -> bytes:
    fcs = crc16_802154(payload)
    return payload + bytes([fcs & 0xFF, (fcs >> 8) & 0xFF])


def _check_fcs(frame: bytes) -> bool:
    if len(frame) < 2:
        return False
    got = frame[-2] | (frame[-1] << 8)
    calc = crc16_802154(frame[:-2])
    return got == calc


def _normalize_rx_selectivity(rx_selectivity: dict | None) -> dict:
    sel = {
        "pass_bw_hz": 120e6,
        "transition_hz": 60e6,
        "stop_atten_db": 25.0,
        "taps": 257,
    }
    if rx_selectivity is not None:
        sel.update(rx_selectivity)
    # Backward compatibility with older key names.
    if "passband_hz" in sel:
        sel["pass_bw_hz"] = float(sel["passband_hz"])
    if "stopband_hz" in sel:
        sel["transition_hz"] = float(sel["stopband_hz"]) - float(sel["pass_bw_hz"])
    if "stopband_atten_db" in sel:
        sel["stop_atten_db"] = float(sel["stopband_atten_db"])
    return sel


def _design_rx_selectivity_fir(fs_hz: float, rx_selectivity: dict | None) -> np.ndarray:
    sel = _normalize_rx_selectivity(rx_selectivity)
    pass_bw = float(sel["pass_bw_hz"])
    transition = float(sel["transition_hz"])
    stop_atten = float(sel["stop_atten_db"])
    taps = int(sel["taps"])
    passband_hz = max(1.0, min(pass_bw, fs_hz / 2.0 - 2.0))
    stopband_hz = min(fs_hz / 2.0 - 1.0, passband_hz + max(1.0, transition))
    if stopband_hz <= passband_hz:
        stopband_hz = min(fs_hz / 2.0 - 1.0, passband_hz + 1.0)
    h, _ = design_lowpass_fir(
        fs_hz=fs_hz,
        pass_hz=passband_hz,
        stop_hz=stopband_hz,
        stop_atten_db=stop_atten,
        num_taps=taps,
    )
    return h


def apply_rx_frontend_filter(
    wf: np.ndarray,
    fs: float,
    passband_hz: float,
    stopband_hz: float,
    stopband_atten_db: float = 25.0,
    taps: int = 257,
) -> np.ndarray:
    """
    Apply UWB RX front-end selectivity filter (finite stopband attenuation).

    The filter is Kaiser-windowed FIR low-pass, used to model non-ideal
    adjacent-channel rejection.
    """
    h, _ = design_lowpass_fir(
        fs_hz=fs,
        pass_hz=passband_hz,
        stop_hz=stopband_hz,
        stop_atten_db=stopband_atten_db,
        num_taps=taps,
    )
    return apply_fir_same(wf, h)


def _resample_with_antialias(x: np.ndarray, fs_in_hz: float, fs_out_hz: float) -> np.ndarray:
    if abs(fs_in_hz - fs_out_hz) <= 1e-9:
        return np.asarray(x, dtype=np.complex128).flatten()

    x = np.asarray(x, dtype=np.complex128).flatten()
    if fs_out_hz < fs_in_hz:
        cutoff_hz = 0.45 * fs_out_hz
        trans_hz = max(1.0, 0.1 * cutoff_hz)
        h_aa, _ = design_lowpass_fir(
            fs_hz=fs_in_hz,
            pass_hz=cutoff_hz,
            stop_hz=min(fs_in_hz / 2.0 - 1.0, cutoff_hz + trans_hz),
            stop_atten_db=50.0,
            num_taps=257,
        )
        x = apply_fir_same(x, h_aa)

    try:
        from scipy import signal  # type: ignore

        gcd = int(np.gcd(int(round(fs_in_hz)), int(round(fs_out_hz))))
        up = int(round(fs_out_hz / gcd))
        down = int(round(fs_in_hz / gcd))
        if up > 0 and down > 0:
            y = signal.resample_poly(x, up=up, down=down)
            return np.asarray(y, dtype=np.complex128).flatten()
    except Exception:
        pass

    return resample_complex_linear(x, fs_in=fs_in_hz, fs_out=fs_out_hz)


def agc_normalize(x: np.ndarray, target_rms_dbfs: float = -12.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).flatten()
    rms_now = np.sqrt(float(np.mean(np.abs(x) ** 2)) + 1e-30)
    rms_tgt = 10.0 ** (float(target_rms_dbfs) / 20.0)
    return (x * (rms_tgt / (rms_now + 1e-30))).astype(np.complex128)


def adc_quantize_iq(x: np.ndarray, nbits: int = 10) -> np.ndarray:
    if nbits < 2:
        return x.astype(np.complex128)
    x = np.asarray(x, dtype=np.complex128).flatten()
    q_levels = 2**nbits
    step = 2.0 / (q_levels - 1)  # full scale [-1,1]
    re = np.clip(np.real(x), -1.0, 1.0)
    im = np.clip(np.imag(x), -1.0, 1.0)
    re_q = np.round((re + 1.0) / step) * step - 1.0
    im_q = np.round((im + 1.0) / step) * step - 1.0
    return (re_q + 1j * im_q).astype(np.complex128)


def _bandlimit_complex_noise(n: int, fs_hz: float, pass_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    Z = np.fft.fft(z)
    f = np.fft.fftfreq(n, d=1.0 / fs_hz)
    mask = np.abs(f) <= float(pass_hz)
    Z *= mask.astype(float)
    return np.fft.ifft(Z).astype(np.complex128)


def _complex_awgn_with_power(n: int, p_w: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    p0 = power_w(z) + 1e-30
    return (z * np.sqrt(max(float(p_w), 0.0) / p0)).astype(np.complex128)


def _ds_twr_tof_rstu(
    i_tx1: float,
    i_rx1: float,
    r_rx1: float,
    r_tx1: float,
    i_tx2: float,
    i_rx2: float,
    r_rx2: float,
    r_tx2: float,
) -> float:
    ra = i_rx1 - i_tx1
    rb = i_rx2 - i_tx2
    da = r_tx1 - r_rx1
    db = r_tx2 - r_rx2
    denom = ra + rb + da + db
    if denom <= 1e-12:
        return 0.0
    return max(0.0, (ra * rb - da * db) / denom)


def _ber_bpsk_awgn(snr_db: float) -> float:
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(max(snr_lin, 1e-18)))


def _build_rsf_template(cfg: MmsUwbConfig) -> np.ndarray:
    base = get_rsf_base_sequence(code_index=cfg.phy_uwb_mms_rsf_code_index).astype(np.float64)
    if len(base) == 0:
        raise ValueError("Empty RSF base sequence")
    n_chips = max(1, cfg.phy_uwb_mms_rsf_length_units512 * 512)
    reps = int(np.ceil(n_chips / len(base)))
    seq = np.tile(base, reps)[:n_chips]
    seq = np.tile(seq, max(1, cfg.phy_uwb_mms_rsf_reps))
    seq = seq / (np.sqrt(np.mean(np.abs(seq) ** 2)) + 1e-30)
    return seq.astype(np.complex128)


def _first_path_legacy(corr: np.ndarray, rx_lead_zeros: int, fp_alpha: float) -> tuple[float, float]:
    peak = int(np.argmax(corr))
    peak_val = float(corr[peak])
    n_noise = max(32, min(256, len(corr) // 8))
    noise_floor = float(np.median(corr[:n_noise]))
    search_back = min(2048, peak + 1)
    left = max(rx_lead_zeros, peak - search_back + 1)
    local = corr[left : peak + 1]
    thr = noise_floor + fp_alpha * max(peak_val - noise_floor, 1e-12)
    idx_rel = np.where(local >= thr)[0]
    fp = int(left + idx_rel[0]) if len(idx_rel) > 0 else peak

    fp_f = float(fp)
    if 1 <= fp < (len(corr) - 1):
        y1 = corr[fp - 1]
        y2 = corr[fp]
        y3 = corr[fp + 1]
        den = (y1 - 2.0 * y2 + y3)
        if abs(den) > 1e-18:
            delta = 0.5 * (y1 - y3) / den
            fp_f = fp_f + float(np.clip(delta, -0.5, 0.5))
    return fp_f, float(peak_val)


def _first_path_cfar_leading_edge(
    corr: np.ndarray,
    rx_lead_zeros: int,
    fp_alpha: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
) -> tuple[float, float]:
    peak = int(np.argmax(corr))
    peak_val = float(corr[peak])
    n_noise = max(32, min(256, len(corr) // 8))
    noise_floor = float(np.median(corr[:n_noise]))

    search_back = min(2048, peak + 1)
    left = max(rx_lead_zeros, peak - search_back + 1)
    right = peak
    det_idx = None
    g = max(1, int(cfar_guard))
    t = max(4, int(cfar_train))
    mult = max(1.0, float(cfar_mult))
    for i in range(left + t + g, right - t - g + 1):
        l0 = i - g - t
        l1 = i - g
        r0 = i + g + 1
        r1 = i + g + 1 + t
        noise_l = float(np.mean(corr[l0:l1]))
        noise_r = float(np.mean(corr[r0:r1]))
        noise_hat = 0.5 * (noise_l + noise_r)
        thr_i = max(noise_floor * (1.0 + 0.5 * fp_alpha), mult * noise_hat)
        if corr[i] >= thr_i and corr[i] >= corr[i - 1]:
            det_idx = i
            break
    fp = int(det_idx) if det_idx is not None else peak

    fp_f = float(fp)
    if fp < (len(corr) - 4):
        win = corr[fp : min(fp + 24, len(corr))]
        local_peak = float(np.max(win)) if len(win) > 0 else peak_val
        if local_peak > 1e-18:
            yn = win / local_peak
            xs = np.arange(len(win), dtype=float)
            mask = (yn >= 0.10) & (yn <= 0.60)
            if int(np.sum(mask)) >= 3:
                a, b = np.polyfit(xs[mask], yn[mask], 1)
                if abs(a) > 1e-12:
                    x_cross = float(np.clip((0.5 - b) / a, 0.0, len(win) - 1))
                    fp_f = float(fp) + x_cross
    return fp_f, float(peak_val)


def _first_path_simple_leading_edge(corr: np.ndarray, alpha: float = 0.25) -> tuple[float, float]:
    i_peak = int(np.argmax(corr))
    peak_val = float(corr[i_peak])
    n0 = max(8, i_peak // 2)
    if n0 <= 0:
        return float(i_peak), peak_val
    noise_floor = float(np.median(corr[:n0]))
    thr = noise_floor + float(alpha) * max(peak_val - noise_floor, 1e-18)
    # First-path detector: earliest local max above threshold.
    i_fp = i_peak
    for i in range(1, i_peak):
        if (corr[i] > thr) and (corr[i] >= corr[i - 1]) and (corr[i] >= corr[i + 1]):
            i_fp = i
            break
    fp_f = float(i_fp)
    if 1 <= i_fp < (len(corr) - 1):
        y1 = corr[i_fp - 1]
        y2 = corr[i_fp]
        y3 = corr[i_fp + 1]
        den = (y1 - 2.0 * y2 + y3)
        if abs(den) > 1e-18:
            delta = 0.5 * (y1 - y3) / den
            fp_f = fp_f + float(np.clip(delta, -0.5, 0.5))
    return fp_f, peak_val


def parabolic_interp_peak(y: np.ndarray, k: int) -> float:
    """
    Parabolic interpolation around peak index k.
    Returns fractional offset in [-0.5, 0.5].
    """
    if k <= 0 or k >= (len(y) - 1):
        return 0.0
    y_m1 = float(y[k - 1])
    y_0 = float(y[k])
    y_p1 = float(y[k + 1])
    den = (y_m1 - 2.0 * y_0 + y_p1)
    if abs(den) <= 1e-18:
        return 0.0
    delta = 0.5 * (y_m1 - y_p1) / den
    return float(np.clip(delta, -0.5, 0.5))


def _fft_zero_pad_interp_complex(x: np.ndarray, up: int, validate: bool = False) -> np.ndarray:
    """
    Complex FFT zero-padding interpolation with exact grid consistency:
    x_up[::up] == x (within numerical tolerance).
    """
    xv = np.asarray(x, dtype=np.complex128).flatten()
    n = len(xv)
    if up <= 1 or n == 0:
        return xv.copy()
    m = int(n * up)
    X = np.fft.fft(xv)
    Y = np.zeros(m, dtype=np.complex128)
    if n % 2 == 1:
        kpos = (n + 1) // 2
        Y[:kpos] = X[:kpos]
        nneg = (n - 1) // 2
        Y[m - nneg :] = X[n - nneg :]
    else:
        kpos = n // 2
        Y[:kpos] = X[:kpos]
        Y[m - kpos :] = X[n - kpos :]
    x_up = np.fft.ifft(Y) * float(up)
    if validate:
        denom = max(float(np.max(np.abs(xv))), 1e-12)
        err = float(np.max(np.abs(x_up[::up] - xv)))
        if err > (1e-6 * denom):
            raise RuntimeError(f"FFT upsample grid check failed: err={err:.3e}, ref={denom:.3e}")
    return x_up


def _refine_peak_fft_upsample_complex(
    corr_cplx: np.ndarray,
    k_center: int,
    half_win: int,
    up: int,
    search_radius: int | None = None,
    validate: bool = False,
) -> float:
    if up <= 1 or half_win <= 1:
        return float(k_center)
    x = np.asarray(corr_cplx, dtype=np.complex128).flatten()
    if x.size < 3:
        return float(k_center)
    x_up = _fft_zero_pad_interp_complex(x, up=up, validate=validate)
    mag_up = np.abs(x_up) ** 2
    c_up = int(k_center * up)
    rad = int(max(2, half_win) if search_radius is None else max(1, int(search_radius)))
    rad_up = int(rad * up)
    l = max(0, c_up - rad_up)
    r = min(len(mag_up), c_up + rad_up + 1)
    seg = mag_up[l:r]
    if seg.size == 0:
        return float(k_center)
    idx_up = int(l + int(np.argmax(seg)))
    delta_up = parabolic_interp_peak(mag_up, idx_up)
    return (float(idx_up) + float(delta_up)) / float(up)


def _first_path_index(corr: np.ndarray, peak_idx: int, k_sigma: float = 6.0) -> int:
    n0 = max(16, min(peak_idx // 2, len(corr) // 4))
    if n0 < 8:
        return int(peak_idx)
    noise = corr[:n0]
    mu = float(np.mean(noise))
    sig = float(np.std(noise) + 1e-18)
    thr = mu + float(k_sigma) * sig
    crossing = None
    for i in range(2, max(3, peak_idx)):
        if (corr[i] > thr) and (corr[i] >= corr[i - 1]):
            crossing = i
            break
    if crossing is None:
        return int(peak_idx)
    return int(crossing)


def _save_corr_debug_csv(
    path: str,
    mag2_up: np.ndarray,
    up: int,
    k_max_up: int,
    k_fp_up: int,
    noise_start_up: int,
    noise_end_up: int,
    thr: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    win = max(16, 256 * max(1, int(up)))
    s = max(0, int(k_max_up) - win)
    e = min(len(mag2_up), int(k_max_up) + win + 1)
    idx = np.arange(s, e, dtype=float)
    k_base = idx / max(1, int(up))
    is_kmax = (idx == int(k_max_up)).astype(int)
    is_kfp = (idx == int(k_fp_up)).astype(int)
    in_noise_win = ((idx >= int(noise_start_up)) & (idx < int(noise_end_up))).astype(int)
    thr_col = np.full_like(k_base, float(thr))
    mat = np.column_stack([k_base, mag2_up[s:e], is_kmax, is_kfp, in_noise_win, thr_col])
    np.savetxt(
        path,
        mat,
        delimiter=",",
        header="k_base,mag2,is_kmax,is_kfp,in_noise_win,thr",
        comments="",
    )


def _estimate_delay_samples_from_rx(
    rx_wf: np.ndarray,
    ref: np.ndarray,
    rx_lead_zeros: int,
    detector_mode: str,
    fp_alpha: float,
    nlos_alpha: float,
    nlos_cap_ratio: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
    quality_min_db: float,
    toa_refine_method: str,
    corr_upsample: int,
    corr_win: int,
    first_path: bool,
    first_path_thr_db: float = 13.0,
    first_path_peak_frac: float | None = 0.08,
    fp_use_adaptive_thr: bool = False,
    fp_snr_switch_db: float = 14.0,
    fp_thr_noise_cap_mult: float = 2.5,
    fp_thr_min_floor_mult: float = 3.0,
    first_path_search_back: int = 8,
    first_path_persist: int = 3,
    first_path_local_win: int = 8,
    corr_debug_csv_path: str | None = None,
) -> dict[str, float | bool | str]:
    corr_cplx = np.convolve(rx_wf, np.conj(ref[::-1]), mode="valid").astype(np.complex128)
    corr = (np.abs(corr_cplx) ** 2).astype(np.float64)
    if len(corr_cplx) == 0:
        return {
            "delay_samples": 0.0,
            "delay_samples_int": 0.0,
            "delay_samples_frac": 0.0,
            "peak": 0.0,
            "noise_floor": 0.0,
            "detect_ok": False,
            "reason": "empty_correlation",
        }
    n_noise = max(32, min(256, len(corr) // 8))
    noise_floor = float(np.median(corr[:n_noise]))
    peak_idx = int(np.argmax(corr))
    peak_pos_f = float(peak_idx)
    peak_mag = float(corr[peak_idx])

    # Build refined grid first when requested.
    up = max(1, int(corr_upsample))
    if toa_refine_method == "fft_upsample" and up > 1:
        corr_up = _fft_zero_pad_interp_complex(corr_cplx, up=up, validate=True)
        env_up = np.abs(corr_up).astype(np.float64)
    else:
        up = 1
        env_up = np.abs(corr_cplx).astype(np.float64)
    mag2_up = env_up ** 2

    k_max_up = int(np.argmax(env_up))
    if 1 <= k_max_up < (len(mag2_up) - 1):
        k_max_up_f = float(k_max_up) + parabolic_interp_peak(mag2_up, k_max_up)
    else:
        k_max_up_f = float(k_max_up)
    k_max_f = float(k_max_up) / float(up)
    k_max_refined_f = float(k_max_up_f) / float(up)
    k_fp_up = k_max_up
    fp_fallback = False
    thr = 0.0
    thr_noise = 0.0
    thr_peak = 0.0
    snr_corr_db = 0.0
    noise_floor_up = 0.0
    thr_mode = "n/a"

    fp_cross_up_f = None
    fp_cross_refine_dbg = "none"
    fp_cross_slope_dbg = 0.0
    fp_cross_resid_dbg = float("nan")
    noise_start_up = 0
    noise_end_up = max(1, min(len(mag2_up), int(rx_lead_zeros * up)))
    if first_path:
        # First-path from leading-edge threshold crossing on upsampled CIR.
        # Noise is estimated from a guaranteed pre-arrival window.
        pre_end = max(16, min(len(env_up) // 3, int(max(1, rx_lead_zeros - 1) * up)))
        pre_start = max(0, pre_end - max(16, 128 * up))
        noise_start_up = pre_start
        noise_end_up = pre_end
        peak_frac = float(first_path_peak_frac) if first_path_peak_frac is not None else 0.0
        if bool(fp_use_adaptive_thr):
            # Adaptive leading-edge on correlation power metric.
            det_sig = mag2_up
            noise_seg = det_sig[pre_start:pre_end]
            noise_floor_up = float(np.median(noise_seg)) if noise_seg.size > 0 else float(np.median(det_sig))
            thr_noise = noise_floor_up * (10.0 ** (float(first_path_thr_db) / 10.0))
            peak_val = float(np.max(det_sig))
            if peak_frac > 0.0:
                thr_peak = peak_frac * peak_val
            snr_corr_db = 10.0 * np.log10((peak_val + 1e-18) / (noise_floor_up + 1e-18))
            if snr_corr_db >= float(fp_snr_switch_db):
                thr = min(thr_noise, thr_peak) if thr_peak > 0.0 else thr_noise
                thr_mode = "min(noise,peak)"
            else:
                thr = thr_noise
                thr_mode = "noise"
            # Keep backward-compatible safety caps/floors.
            if thr_peak > 0.0 and float(fp_thr_noise_cap_mult) > 0.0:
                thr = min(thr, float(fp_thr_noise_cap_mult) * thr_peak)
            thr = max(thr, float(fp_thr_min_floor_mult) * noise_floor_up)
        else:
            # Legacy-compatible mode on power metric.
            det_sig = mag2_up
            noise_seg = det_sig[pre_start:pre_end]
            noise_floor_up = float(np.median(noise_seg)) if noise_seg.size > 0 else float(np.median(det_sig))
            thr_noise = noise_floor_up * (10.0 ** (float(first_path_thr_db) / 10.0))
            if peak_frac > 0.0:
                thr_peak = peak_frac * float(np.max(det_sig))
            thr = max(thr_noise, thr_peak)
            snr_corr_db = 10.0 * np.log10((float(np.max(det_sig)) + 1e-30) / (noise_floor_up + 1e-30))
            thr_mode = "legacy_max(noise,peak)"

        start = max(int(max(0, rx_lead_zeros - 2) * up), k_max_up - int(first_path_search_back * up), 1)
        stop = max(start + 1, k_max_up + 1)
        pers = max(1, int(first_path_persist))
        loc_win = max(1, int(first_path_local_win))
        found = None

        # Option A: earliest threshold crossing with persistence.
        for i in range(start, min(stop, len(det_sig) - pers)):
            if np.all(det_sig[i : i + pers] >= thr):
                i0 = max(start, i - 1)
                y0 = float(det_sig[i0])
                y1 = float(det_sig[i])
                if y1 > y0 and y1 >= thr:
                    frac = (thr - y0) / max(y1 - y0, 1e-30)
                    frac = float(np.clip(frac, 0.0, 1.0))
                    k_cross_seed = float(i0) + frac
                else:
                    k_cross_seed = float(i)

                fp_cross_up_f = k_cross_seed

                # Refine threshold crossing by local line-fit y = a*k + b.
                win_r = 3
                fit_l = max(start, int(np.floor(k_cross_seed)) - win_r)
                fit_r = min(len(det_sig), int(np.floor(k_cross_seed)) + win_r + 1)
                if (fit_r - fit_l) >= 4:
                    k_fit = np.arange(fit_l, fit_r, dtype=float)
                    y_fit = det_sig[fit_l:fit_r].astype(float)
                    k0 = float(np.mean(k_fit))
                    y0f = float(np.mean(y_fit))
                    kk = k_fit - k0
                    den = float(np.sum(kk * kk))
                    if den > 1e-18:
                        a = float(np.sum(kk * (y_fit - y0f)) / den)
                        b = float(y0f - a * k0)
                        fp_cross_slope_dbg = a
                        y_hat = a * k_fit + b
                        fp_cross_resid_dbg = float(np.mean((y_fit - y_hat) ** 2))
                        if a > 0.0:
                            k_fit_cross = (float(thr) - b) / a
                            if np.isfinite(k_fit_cross):
                                fp_cross_up_f = float(np.clip(k_fit_cross, float(fit_l), float(fit_r - 1)))
                                fp_cross_refine_dbg = "linefit"

                r = min(stop, i + loc_win + 1)
                found = int(i + int(np.argmax(det_sig[i:r]))) if r > i else i
                break

        # Option B: earliest local peak above threshold.
        if found is None:
            for i in range(max(start, 1), min(stop - 1, len(det_sig) - 1)):
                if det_sig[i] <= thr:
                    continue
                if det_sig[i] >= det_sig[i - 1] and det_sig[i] >= det_sig[i + 1]:
                    found = i
                    break

        if found is not None:
            if fp_cross_up_f is not None:
                k_fp_up = int(np.clip(int(round(fp_cross_up_f)), 0, len(mag2_up) - 1))
            else:
                k_fp_up = found
        else:
            fp_fallback = True
            k_fp_up = k_max_up
    elif detector_mode == "cfar":
        fp_f, peak_mag = _first_path_cfar_leading_edge(
            corr=corr,
            rx_lead_zeros=rx_lead_zeros,
            fp_alpha=fp_alpha,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
        )
        k_fp_up = int(max(0, min(len(mag2_up) - 1, round(fp_f * up))))
    elif detector_mode == "first_path":
        # Keep backward compatibility, but when `first_path` flag is False,
        # default to peak-based coarse detector.
        fp_f = float(peak_idx)
        peak_mag = float(corr[peak_idx])
        k_fp_up = int(max(0, min(len(mag2_up) - 1, round(fp_f * up))))
    else:
        fp_f = float(peak_idx)
        peak_mag = float(corr[peak_idx])
        k_fp_up = int(max(0, min(len(mag2_up) - 1, round(fp_f * up))))

    # Select k_fp in first-path mode; strongest otherwise.
    if first_path:
        sel_idx = int(np.clip(int(np.floor(float(k_fp_up) / float(up))), 0, len(corr) - 1))
    else:
        sel_idx = int(np.clip(int(np.floor(float(k_max_up) / float(up))), 0, len(corr) - 1))

    if toa_refine_method == "parabolic":
        fp_f = float(sel_idx) + parabolic_interp_peak(corr, sel_idx)
    elif toa_refine_method == "fft_upsample":
        # In first-path mode prefer threshold-crossing time directly.
        if first_path and (fp_cross_up_f is not None):
            fp_f = float(fp_cross_up_f) / float(up)
        else:
            seed_up = int(k_fp_up if first_path else k_max_up)
            idx_up = seed_up
            if 1 <= idx_up < (len(mag2_up) - 1):
                idx_up_f = float(idx_up) + parabolic_interp_peak(mag2_up, idx_up)
            else:
                idx_up_f = float(idx_up)
            fp_f = idx_up_f / float(up)
    else:
        fp_f = float(sel_idx)

    if corr_debug_csv_path is not None:
        _save_corr_debug_csv(
            corr_debug_csv_path,
            mag2_up=mag2_up,
            up=up,
            k_max_up=k_max_up,
            k_fp_up=k_fp_up,
            noise_start_up=noise_start_up,
            noise_end_up=noise_end_up,
            thr=float(thr),
        )

    sel_delta = float(fp_f - float(sel_idx))

    excess_samples = max(0.0, peak_pos_f - fp_f)
    fp_f = max(fp_f, float(rx_lead_zeros))
    raw_delay_samples = max(0.0, fp_f - float(rx_lead_zeros))
    raw_delay_int = max(0.0, math.floor(fp_f) - float(rx_lead_zeros))
    nlos_bias_samples = min(nlos_alpha * excess_samples, nlos_cap_ratio * raw_delay_samples)
    delay_f = max(0.0, raw_delay_samples - nlos_bias_samples)
    delay_i = max(0.0, raw_delay_int - nlos_bias_samples)
    ratio = (peak_mag + 1e-30) / (noise_floor + 1e-30)
    quality_db = 20.0 * np.log10(ratio)
    detect_ok = bool(quality_db >= float(quality_min_db))
    reason = "ok" if detect_ok else "low_correlation_quality"
    return {
        "delay_samples": float(delay_f),
        "delay_samples_int": float(delay_i),
        "delay_samples_frac": float(delay_f - delay_i),
        "peak": float(peak_mag),
        "noise_floor": float(noise_floor),
        "quality_db": float(quality_db),
        "detect_ok": detect_ok,
        "reason": reason,
        "peak_idx": float(peak_idx),
        "peak_pos_f": float(peak_pos_f),
        "sel_idx": float(sel_idx),
        "sel_delta": float(sel_delta),
        "k_max": float(k_max_f),
        "k_max_refined": float(k_max_refined_f),
        "k_fp": float(float(k_fp_up) / float(up)),
        "fp_fallback": float(1.0 if fp_fallback else 0.0),
        "fp_thr_db": float(first_path_thr_db),
        "fp_peak_frac": float(first_path_peak_frac if first_path_peak_frac is not None else 0.0),
        "fp_use_adaptive_thr": float(1.0 if fp_use_adaptive_thr else 0.0),
        "fp_snr_switch_db": float(fp_snr_switch_db),
        "fp_thr_noise_cap_mult": float(fp_thr_noise_cap_mult),
        "fp_thr_min_floor_mult": float(fp_thr_min_floor_mult),
        "fp_snr_corr_db": float(snr_corr_db),
        "fp_noise_floor": float(noise_floor_up),
        "fp_thr_noise_abs": float(thr_noise),
        "fp_thr_peak_abs": float(thr_peak),
        "fp_thr_mode": str(thr_mode),
        "fp_peak_ratio_db": float(10.0 * np.log10((mag2_up[k_fp_up] + 1e-30) / (mag2_up[k_max_up] + 1e-30))),
        "fp_cross_refine": str(fp_cross_refine_dbg),
        "fp_cross_slope": float(fp_cross_slope_dbg),
        "fp_cross_resid": float(fp_cross_resid_dbg),
        "noise_win_start": float(noise_start_up) / float(up),
        "noise_win_end": float(noise_end_up) / float(up),
        "fp_thr_abs": float(thr),
        "toa_peak_raw": float(max(0.0, k_max_refined_f - float(rx_lead_zeros))),
        "toa_edge_raw": float(max(0.0, fp_f - float(rx_lead_zeros))),
    }


def _build_wifi_interference_at_uwb_rx(
    length_out: int,
    uwb_fs_hz: float,
    uwb_fc_hz: float,
    wifi_tx,
    wifi_params: dict,
    selectivity_h: np.ndarray,
    leg_seed: int,
) -> np.ndarray:
    wf, _ = make_wifi_interference_at_victim_baseband(
        wifi_tx=wifi_tx,
        victim_fs_hz=uwb_fs_hz,
        victim_fc_hz=uwb_fc_hz,
        wifi_fc_hz=float(wifi_params.get("fc_hz", 6.49e9)),
        wifi_bw_hz=float(wifi_params.get("bw_hz", 160e6)),
        duration_s=max(length_out / uwb_fs_hz * 1.2, 50e-6),
        wifi_tx_power_dbw=float(wifi_params.get("tx_power_dbw", -20.0)),
        distance_m=float(wifi_params.get("distance_m", 2.0)),
        pathloss_exp=float(wifi_params.get("pathloss_exp", 2.0)),
        delays_s=tuple(wifi_params.get("delays_s", (0.0, 30e-9, 80e-9))),
        powers_db=tuple(wifi_params.get("powers_db", (0.0, -6.0, -10.0))),
        k_factor_db=float(wifi_params.get("k_factor_db", 6.0)),
        aclr_db=float(wifi_params.get("aclr_db", 35.0)),
        rx_selectivity_h=selectivity_h,
        rx_stop_db=float(wifi_params.get("rx_stop_db", 25.0)),
        alias_guard_ratio=float(wifi_params.get("alias_guard_ratio", 0.45)),
        coupling_db=float(wifi_params.get("coupling_db", 0.0)),
        wifi_oob_atten_db=wifi_params.get("wifi_oob_atten_db"),
        seed=leg_seed,
    )
    if len(wf) < length_out:
        wf = np.pad(wf, (0, length_out - len(wf)))
    return wf[:length_out].astype(np.complex128)


def make_wifi_interference_at_victim_baseband(
    wifi_tx,
    victim_fs_hz: float,
    victim_fc_hz: float,
    wifi_fc_hz: float,
    wifi_bw_hz: float,
    duration_s: float,
    wifi_tx_power_dbw: float,
    distance_m: float,
    pathloss_exp: float,
    delays_s: tuple[float, ...],
    powers_db: tuple[float, ...],
    k_factor_db: float,
    aclr_db: float,
    rx_selectivity_h: np.ndarray,
    rx_stop_db: float,
    alias_guard_ratio: float,
    coupling_db: float,
    wifi_oob_atten_db,
    seed: int,
) -> tuple[np.ndarray, dict]:
    dist_wifi = float(distance_m)
    tx_power_dbw = float(wifi_tx_power_dbw)
    target_tp = 800.0
    std = "wifi7"
    ch_bw_mhz = int(round(wifi_bw_hz / 1e6))

    wf_tx, _ = wifi_tx.generate_for_target_rx_throughput(
        target_rx_throughput_mbps=target_tp,
        duration_s=float(duration_s),
        channel_bw_mhz=ch_bw_mhz,
        standard=std,
        tx_power_dbw=tx_power_dbw,
        center_freq_hz=wifi_fc_hz,
    )
    wf_tx = wifi_tx.apply_tx_emission_mask(
        wf_tx,
        fs_hz=float(wifi_bw_hz),
        channel_bw_hz=float(wifi_bw_hz),
        aclr_db=float(aclr_db),
        seed=seed + 17,
    )

    wf_ch, _, _, _, _ = apply_distance_rician_channel(
        wf=wf_tx,
        fs_hz=wifi_bw_hz,
        fc_hz=wifi_fc_hz,
        distance_m=dist_wifi,
        pathloss_exp=float(pathloss_exp),
        delays_s=delays_s,
        powers_db=powers_db,
        k_factor_db=float(k_factor_db),
        include_toa=True,
        seed=seed + 1000,
    )
    wf_rs = _resample_with_antialias(wf_ch, fs_in_hz=wifi_bw_hz, fs_out_hz=victim_fs_hz)
    f_off = float(wifi_fc_hz) - float(victim_fc_hz)
    f_alias = alias_frequency(f_off, victim_fs_hz)
    p_wifi_rx = power_w(wf_rs) + 1e-30
    extra_oob_db = 0.0
    if callable(wifi_oob_atten_db):
        extra_oob_db = float(wifi_oob_atten_db(abs(f_off)))
    elif wifi_oob_atten_db is not None:
        extra_oob_db = float(wifi_oob_atten_db)
    if extra_oob_db > 0.0:
        wf_rs = wf_rs * (10.0 ** (-extra_oob_db / 20.0))
        p_wifi_rx = power_w(wf_rs) + 1e-30

    alias_safe = bool((abs(f_off) + 0.5 * float(wifi_bw_hz)) < 0.5 * victim_fs_hz)
    # Always use representable sampled offset; for large |f_off| this is the aliased tone.
    wf_bb = freq_shift(wf_rs, fs=victim_fs_hz, f_off_hz=f_alias).astype(np.complex128)
    # RF preselector attenuation is based on true RF offset, not aliased frequency.
    if abs(f_off) > 0.5 * victim_fs_hz:
        wf_bb = wf_bb * (10.0 ** (-float(rx_stop_db) / 20.0))
    p_leak = power_w(wf_bb) + 1e-30
    # Strong blocker power at RF front-end (before preselector).
    p_block = p_wifi_rx

    if float(coupling_db) != 0.0:
        wf_bb = wf_bb * (10.0 ** (float(coupling_db) / 20.0))
        p_block *= dbw_to_w(float(coupling_db))
    wf_sel = apply_fir_same(wf_bb, rx_selectivity_h)
    p_sel = power_w(wf_sel) + 1e-30

    # Sanity assertions: leakage path must not increase power.
    assert w_to_dbw(max(p_leak, 1e-30)) <= w_to_dbw(p_wifi_rx) + 0.5
    assert w_to_dbw(p_sel) <= w_to_dbw(max(p_leak, 1e-30)) + 0.5

    dbg = {
        "P_wifi_rx_dbw": w_to_dbw(p_wifi_rx),
        "P_leak_dbw": w_to_dbw(max(p_leak, 1e-30)),
        "P_block_dbw": w_to_dbw(max(p_block, 1e-30)),
        "P_int_sel_dbw": w_to_dbw(p_sel),
        "f_off_hz": float(f_off),
        "f_alias_hz": float(f_alias),
        "used_alias_safe_mode": bool(not alias_safe),
    }
    return wf_sel.astype(np.complex128), dbg


def _estimate_toa_calibration_samples(
    ref_template: np.ndarray,
    rx_selectivity_h: np.ndarray,
    detector_mode: str,
    fp_alpha: float,
    nlos_alpha: float,
    nlos_cap_ratio: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
    quality_min_db: float,
    toa_refine_method: str,
    corr_upsample: int,
    corr_win: int,
    first_path: bool,
    first_path_thr_db: float,
    first_path_peak_frac: float | None,
    fp_use_adaptive_thr: bool,
    fp_snr_switch_db: float,
    fp_thr_noise_cap_mult: float,
    fp_thr_min_floor_mult: float,
    first_path_search_back: int,
    first_path_persist: int,
    first_path_local_win: int,
) -> float:
    rx_lead_zeros = 32
    desired = np.concatenate([np.zeros(rx_lead_zeros, dtype=np.complex128), ref_template.astype(np.complex128)])
    desired_rx = apply_fir(desired, rx_selectivity_h)
    ref_rx = apply_fir(ref_template.astype(np.complex128), rx_selectivity_h)
    info = _estimate_delay_samples_from_rx(
        rx_wf=desired_rx,
        ref=ref_rx,
        rx_lead_zeros=rx_lead_zeros,
        detector_mode=detector_mode,
        fp_alpha=fp_alpha,
        nlos_alpha=nlos_alpha,
        nlos_cap_ratio=nlos_cap_ratio,
        cfar_guard=cfar_guard,
        cfar_train=cfar_train,
        cfar_mult=cfar_mult,
        quality_min_db=quality_min_db,
        toa_refine_method=toa_refine_method,
        corr_upsample=corr_upsample,
        corr_win=corr_win,
        first_path=first_path,
        first_path_thr_db=first_path_thr_db,
        first_path_peak_frac=first_path_peak_frac,
        fp_use_adaptive_thr=fp_use_adaptive_thr,
        fp_snr_switch_db=fp_snr_switch_db,
        fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
        fp_thr_min_floor_mult=fp_thr_min_floor_mult,
        first_path_search_back=first_path_search_back,
        first_path_persist=first_path_persist,
        first_path_local_win=first_path_local_win,
    )
    # Return correction term so caller can use: delay_used = delay_est + correction.
    return -float(info["delay_samples"])


def auto_calibrate_toa_offset(
    tx_template: np.ndarray,
    ref_template: np.ndarray,
    rx_selectivity_h: np.ndarray,
    fs_hz: float,
    fc_hz: float,
    detector_mode: str,
    fp_alpha: float,
    nlos_alpha: float,
    nlos_cap_ratio: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
    quality_min_db: float,
    toa_refine_method: str,
    corr_upsample: int,
    corr_win: int,
    first_path: bool,
    first_path_thr_db: float,
    first_path_peak_frac: float | None,
    fp_use_adaptive_thr: bool,
    fp_snr_switch_db: float,
    fp_thr_noise_cap_mult: float,
    fp_thr_min_floor_mult: float,
    first_path_search_back: int,
    first_path_persist: int,
    first_path_local_win: int,
    calibration_distance_m: float = 10.0,
    calibration_trials: int = 32,
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    noise_bw_hz: float | None = None,
) -> float:
    """
    Estimate constant ToA bias (in samples) from a noiseless single-path known-distance setup.
    """
    rx_lead_zeros = 32
    ntr = max(1, int(calibration_trials))
    if noise_bw_hz is None:
        noise_bw_hz = float(fs_hz)
    # Calibration should estimate deterministic detector bias, not random SNR bias.
    p_noise = 0.0
    errs = []
    for i in range(ntr):
        wf_ch, _, _, toa_s, _ = apply_distance_rician_channel(
            wf=tx_template.astype(np.complex128),
            fs_hz=fs_hz,
            fc_hz=fc_hz,
            distance_m=float(calibration_distance_m),
            pathloss_exp=2.0,
            delays_s=(0.0,),
            powers_db=(0.0,),
            k_factor_db=40.0,
            include_toa=True,
            seed=99991 + i,
        )
        rx = np.concatenate([np.zeros(rx_lead_zeros, dtype=np.complex128), wf_ch.astype(np.complex128)])
        if p_noise > 0.0:
            rx = rx + _complex_awgn_with_power(len(rx), p_noise, seed=190001 + i)
        rx = apply_fir_same(rx, rx_selectivity_h)
        ref_rx = apply_fir_same(ref_template.astype(np.complex128), rx_selectivity_h)
        info = _estimate_delay_samples_from_rx(
            rx_wf=rx,
            ref=ref_rx,
            rx_lead_zeros=rx_lead_zeros,
            detector_mode=detector_mode,
            fp_alpha=fp_alpha,
            nlos_alpha=nlos_alpha,
            nlos_cap_ratio=nlos_cap_ratio,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
            quality_min_db=quality_min_db,
            toa_refine_method=toa_refine_method,
            corr_upsample=corr_upsample,
            corr_win=corr_win,
            first_path=first_path,
            first_path_thr_db=first_path_thr_db,
            first_path_peak_frac=first_path_peak_frac,
            fp_use_adaptive_thr=fp_use_adaptive_thr,
            fp_snr_switch_db=fp_snr_switch_db,
            fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=fp_thr_min_floor_mult,
            first_path_search_back=first_path_search_back,
            first_path_persist=first_path_persist,
            first_path_local_win=first_path_local_win,
        )
        true_samp = float(toa_s * fs_hz)
        errs.append(float(info["delay_samples"]) - true_samp)
    # Return correction term so caller can use: delay_used = delay_est + correction.
    return -float(np.mean(errs))


def _simulate_leg(
    tx_template: np.ndarray,
    ref_template: np.ndarray,
    uwb_fs_hz: float,
    uwb_fc_hz: float,
    distance_m: float,
    tx_eirp_dbw: float,
    nf_db: float,
    temperature_k: float,
    noise_bw_hz: float,
    pathloss_exp: float,
    delays_s: tuple[float, ...],
    powers_db: tuple[float, ...],
    k_factor_db: float,
    channel_seed: int,
    noise_seed: int,
    detector_mode: str,
    fp_alpha: float,
    nlos_alpha: float,
    nlos_cap_ratio: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
    quality_min_db: float,
    toa_refine_method: str,
    corr_upsample: int,
    corr_win: int,
    first_path: bool,
    first_path_thr_db: float,
    first_path_peak_frac: float | None,
    fp_use_adaptive_thr: bool,
    fp_snr_switch_db: float,
    fp_thr_noise_cap_mult: float,
    fp_thr_min_floor_mult: float,
    first_path_search_back: int,
    first_path_persist: int,
    first_path_local_win: int,
    rx_selectivity_h: np.ndarray,
    rx_stop_atten_db: float,
    interference_wf: np.ndarray | None,
    interference_fs_hz: float | None,
    interference_fc_hz: float | None,
    interference_bw_hz: float | None,
    interference_aclr_db: float,
    interference_block_dbw: float | None,
    use_leakage_equivalent_for_alias: bool,
    wifi_interference_on: bool,
    wifi_tx,
    wifi_params: dict,
    agc_stage: str,
    enable_agc: bool,
    agc_target_dbfs: float,
    enable_adc_clipping: bool,
    clip_dbfs: float,
    quant_bits: int | None,
    agc_min_gain_db: float,
    agc_max_gain_db: float,
    lna_p1db_dbm: float | None,
    lna_max_gain_db: float,
    debug: bool,
    psd_prefix: str | None,
    psd_unit: str,
    psd_sanity_check: bool,
    toa_calibration_samples: float = 0.0,
    corr_debug_csv_path: str | None = None,
) -> tuple[float, float, bool, dict[str, float]]:
    rx_lead_zeros = 32
    tx_target_w = _db_to_w(tx_eirp_dbw)
    tx_now_w = float(np.mean(np.abs(tx_template) ** 2)) + 1e-30
    tx_scaled = tx_template * np.sqrt(tx_target_w / tx_now_w)
    desired_ch, _, _, _, _ = apply_distance_rician_channel(
        wf=tx_scaled,
        fs_hz=uwb_fs_hz,
        fc_hz=uwb_fc_hz,
        distance_m=distance_m,
        pathloss_exp=pathloss_exp,
        delays_s=delays_s,
        powers_db=powers_db,
        k_factor_db=k_factor_db,
        include_toa=True,
        seed=channel_seed,
    )
    sig_ant = np.concatenate([np.zeros(rx_lead_zeros, dtype=np.complex128), desired_ch])

    intf_components: list[np.ndarray] = []
    used_alias_safe_mode = False
    interferer_preselected = False
    p_block_w = 0.0
    p_int_ant_wide_w = 0.0

    if interference_wf is not None:
        if interference_fs_hz is None:
            raise ValueError('interference_fs_hz must be provided when interference_wf is given')
        intf = _resample_with_antialias(interference_wf, fs_in_hz=float(interference_fs_hz), fs_out_hz=uwb_fs_hz)
        if intf.size < sig_ant.size:
            reps = int(np.ceil(sig_ant.size / max(intf.size, 1)))
            intf = np.tile(intf, reps)
        intf = intf[: sig_ant.size]

        p_int_wide_this = power_w(intf) + 1e-30
        p_int_ant_wide_w += p_int_wide_this
        if interference_fc_hz is not None:
            f_off = float(interference_fc_hz) - float(uwb_fc_hz)
            f_alias = alias_frequency(f_off, uwb_fs_hz)
            bw_int_hz = float(interference_bw_hz) if interference_bw_hz is not None else float(interference_fs_hz)
            alias_safe = bool((abs(f_off) + 0.5 * bw_int_hz) < 0.5 * uwb_fs_hz)
            if not alias_safe:
                # Apply RF-domain rejection using true RF offset first, then map to sampled alias.
                aclr_extra_db = float(interference_aclr_db)
                oob_prof = wifi_params.get("wifi_oob_atten_db")
                if callable(oob_prof):
                    aclr_extra_db = max(aclr_extra_db, float(oob_prof(abs(f_off))))
                atten_db = max(0.0, float(rx_stop_atten_db) + aclr_extra_db)
                p_block_w = max(p_block_w, p_int_wide_this)
                if use_leakage_equivalent_for_alias:
                    p_leak_w = p_int_wide_this * dbw_to_w(-atten_db)
                    eq = _bandlimit_complex_noise(
                        n=intf.size,
                        fs_hz=uwb_fs_hz,
                        pass_hz=min(0.45 * uwb_fs_hz, 120e6),
                        seed=noise_seed + 919,
                    )
                    intf = scale_to_power_dbw(eq, w_to_dbw(p_leak_w))
                    interferer_preselected = True
                else:
                    intf = intf * (10.0 ** (-atten_db / 20.0))
                    intf = freq_shift(intf, fs=uwb_fs_hz, f_off_hz=f_alias)
                used_alias_safe_mode = True
            else:
                intf = freq_shift(intf, fs=uwb_fs_hz, f_off_hz=f_alias)
        intf_components.append(intf.astype(np.complex128))

    if wifi_interference_on and wifi_tx is not None:
        c_wifi = _build_wifi_interference_at_uwb_rx(
            length_out=len(sig_ant),
            uwb_fs_hz=uwb_fs_hz,
            uwb_fc_hz=uwb_fc_hz,
            wifi_tx=wifi_tx,
            wifi_params=wifi_params,
            selectivity_h=np.array([1.0]),
            leg_seed=channel_seed,
        )
        p_int_ant_wide_w += power_w(c_wifi) + 1e-30
        intf_components.append(c_wifi)

    has_interferer = len(intf_components) > 0
    int_ant = np.zeros_like(sig_ant)
    for c in intf_components:
        int_ant = int_ant + c[: sig_ant.size]
    if interference_block_dbw is not None:
        p_blk_ext = dbw_to_w(float(interference_block_dbw))
        p_block_w += p_blk_ext
        p_int_ant_wide_w += p_blk_ext

    p_noise = thermal_noise_power_w(fs_hz=noise_bw_hz, nf_db=nf_db, temperature_k=temperature_k)
    noise_ant = _complex_awgn_with_power(len(sig_ant), p_noise, seed=noise_seed + 33)
    rx_ant = sig_ant + int_ant + noise_ant
    p_sig_ant_pre = power_w(sig_ant) + 1e-30
    p_int_ant_pre = power_w(int_ant) + 1e-30
    p_noise_ant_pre = power_w(noise_ant) + 1e-30
    p_total_ant_pre = power_w(rx_ant) + 1e-30

    # Optional front-end compression model driven by wideband antenna RSSI.
    lna_gain_lin = 1.0
    lna_over_db = 0.0
    if lna_p1db_dbm is not None:
        p_ant_wide_w = max(power_w(rx_ant) + p_block_w, 1e-30)
        p_ant_wide_dbm = dbw_to_dbm(w_to_dbw(p_ant_wide_w))
        lna_gain_db_req = float(lna_max_gain_db)
        over_db = p_ant_wide_dbm - float(lna_p1db_dbm)
        lna_over_db = float(max(0.0, over_db))
        if lna_over_db > 0.0:
            # Soft compression: reduce effective gain when input exceeds P1dB.
            lna_gain_db_req = float(lna_max_gain_db) - 0.8 * lna_over_db
        lna_gain_lin = 10.0 ** (lna_gain_db_req / 20.0)
    sig_ant = sig_ant * lna_gain_lin
    int_ant = int_ant * lna_gain_lin
    noise_ant = noise_ant * lna_gain_lin
    rx_ant = sig_ant + int_ant + noise_ant

    if agc_stage not in ('pre_selectivity', 'post_selectivity'):
        raise ValueError("agc_stage must be 'pre_selectivity' or 'post_selectivity'")

    sig_sel_preagc = apply_fir_same(sig_ant, rx_selectivity_h)
    noise_sel_preagc = apply_fir_same(noise_ant, rx_selectivity_h)
    int_sel_preagc = int_ant if interferer_preselected else apply_fir_same(int_ant, rx_selectivity_h)
    rx_sel_preagc = sig_sel_preagc + int_sel_preagc + noise_sel_preagc
    if lna_over_db > 0.0:
        # Blocker-driven nonlinear residue folded into selected band.
        p_sel_now = max(power_w(rx_sel_preagc), 1e-30)
        im3_ratio = min(10.0 ** (lna_over_db / 10.0), 1e3)
        p_imd = p_sel_now * 0.15 * im3_ratio
        imd_noise = _complex_awgn_with_power(len(rx_sel_preagc), p_imd, seed=noise_seed + 97)
        noise_sel_preagc = noise_sel_preagc + imd_noise
        rx_sel_preagc = sig_sel_preagc + int_sel_preagc + noise_sel_preagc

    rms_tgt = 10.0 ** (float(agc_target_dbfs) / 20.0)
    g_min = 10.0 ** (float(agc_min_gain_db) / 20.0)
    g_max = 10.0 ** (float(agc_max_gain_db) / 20.0)
    if agc_stage == 'pre_selectivity':
        p_for_agc = power_w(rx_ant) + p_block_w + 1e-30
    else:
        p_for_agc = power_w(rx_sel_preagc) + 1e-30
    g_req = rms_tgt / (np.sqrt(p_for_agc) + 1e-30)
    g_app = float(np.clip(g_req, g_min, g_max)) if enable_agc else 1.0

    def _adc_path(x: np.ndarray) -> np.ndarray:
        y = x * g_app
        if enable_adc_clipping:
            y = apply_soft_clipping(y, clip_dbfs=clip_dbfs)
        if quant_bits is not None:
            y = adc_quantize_iq(y, nbits=int(quant_bits))
        return y.astype(np.complex128)

    y_total = _adc_path(rx_sel_preagc)
    y_sig = _adc_path(sig_sel_preagc)
    y_int = _adc_path(int_sel_preagc)
    y_noise = _adc_path(noise_sel_preagc)
    y_sig_noise = _adc_path(sig_sel_preagc + noise_sel_preagc)
    # Interference-effective residual after removing the sig+noise reference path.
    int_eff_wave = y_total - y_sig_noise

    rx_adc = y_total
    mixed_pre_clip = rx_sel_preagc * g_app
    clip_level = 10.0 ** (float(clip_dbfs) / 20.0)
    clip_frac = float(np.mean(np.abs(mixed_pre_clip) > clip_level)) if enable_adc_clipping else 0.0
    rx_det = rx_adc

    ref_rx = apply_fir(ref_template, rx_selectivity_h)
    delay_info = _estimate_delay_samples_from_rx(
        rx_wf=rx_det,
        ref=ref_rx,
        rx_lead_zeros=rx_lead_zeros,
        detector_mode=detector_mode,
        fp_alpha=fp_alpha,
        nlos_alpha=nlos_alpha,
        nlos_cap_ratio=nlos_cap_ratio,
        cfar_guard=cfar_guard,
        cfar_train=cfar_train,
        cfar_mult=cfar_mult,
        quality_min_db=quality_min_db,
        toa_refine_method=toa_refine_method,
        corr_upsample=corr_upsample,
        corr_win=corr_win,
        first_path=first_path,
        first_path_thr_db=first_path_thr_db,
        first_path_peak_frac=first_path_peak_frac,
        fp_use_adaptive_thr=fp_use_adaptive_thr,
        fp_snr_switch_db=fp_snr_switch_db,
        fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
        fp_thr_min_floor_mult=fp_thr_min_floor_mult,
        first_path_search_back=first_path_search_back,
        first_path_persist=first_path_persist,
        first_path_local_win=first_path_local_win,
        corr_debug_csv_path=corr_debug_csv_path if debug else None,
    )

    p_sig_ant = power_w(sig_ant) + 1e-30
    p_int_ant = power_w(int_ant) + 1e-30
    p_int_block_ant = max(p_int_ant_wide_w, p_int_ant) + 1e-30
    p_noise_ant = power_w(noise_ant) + 1e-30
    p_total_ant = power_w(rx_ant) + 1e-30

    p_sig_sel = power_w(sig_sel_preagc) + 1e-30
    p_int_sel = max(power_w(int_sel_preagc) + 1e-30, (1e-24 if has_interferer else 1e-30))
    p_noise_sel = power_w(noise_sel_preagc) + 1e-30
    p_total_sel = power_w(rx_sel_preagc) + 1e-30
    p_sig_adc = power_w(y_sig) + 1e-30
    p_int_adc = max(power_w(y_int) + 1e-30, (1e-24 if has_interferer else 1e-30))
    p_noise_adc = power_w(y_noise) + 1e-30
    p_total_adc = power_w(rx_adc) + 1e-30
    p_int_eff = max(power_w(int_eff_wave) + 1e-30, (1e-24 if has_interferer else 1e-30))

    p_sig_ant_pre_dbw = _w_to_db(p_sig_ant_pre)
    p_int_ant_pre_dbw = _w_to_db(p_int_ant_pre)
    p_noise_ant_pre_dbw = _w_to_db(p_noise_ant_pre)
    p_total_ant_pre_dbw = _w_to_db(p_total_ant_pre)

    p_sig_ant_dbw = _w_to_db(p_sig_ant)
    p_int_ant_dbw = _w_to_db(p_int_ant)
    p_noise_dbw = _w_to_db(p_noise_ant)
    p_total_ant_dbw = _w_to_db(p_total_ant)

    p_sig_dbw = _w_to_db(p_sig_sel)
    p_int_dbw = _w_to_db(p_int_sel)
    p_noise_sel_dbw = _w_to_db(p_noise_sel)
    p_total_sel_dbw = _w_to_db(p_total_sel)
    p_sig_adc_dbw = _w_to_db(p_sig_adc)
    p_int_adc_dbw = _w_to_db(p_int_adc)
    p_noise_adc_dbw = _w_to_db(p_noise_adc)
    p_total_adc_dbw = _w_to_db(p_total_adc)
    p_int_eff_dbw = _w_to_db(p_int_eff)

    snr_db = p_sig_dbw - p_noise_sel_dbw
    sir_db = p_sig_dbw - p_int_dbw
    # Detector-effective SINR at ADC/detector domain.
    sinr_db = p_sig_adc_dbw - _w_to_db(p_int_eff + p_noise_adc)
    sir_ant_db = p_sig_ant_dbw - p_int_ant_dbw
    sinr_ant_db = p_sig_ant_dbw - _w_to_db(p_int_ant + p_noise_ant)

    if debug:
        delta_ant_pre_db = p_total_ant_pre_dbw - _w_to_db(p_sig_ant_pre + p_int_ant_pre + p_noise_ant_pre)
        delta_ant_db = p_total_ant_dbw - _w_to_db(p_sig_ant + p_int_ant + p_noise_ant)
        delta_sel_db = p_total_sel_dbw - _w_to_db(p_sig_sel + p_int_sel + p_noise_sel)
        p_total_adc_dbfs = 10.0 * np.log10(p_total_adc + 1e-30)
        p_pre_clip_dbfs = 10.0 * np.log10(power_w(mixed_pre_clip) + 1e-30)
        p_post_clip_dbfs = 10.0 * np.log10(p_total_adc + 1e-30)
        corr_dbg = np.convolve(rx_det, np.conj(ref_rx[::-1]), mode='valid')
        corr_metric_db = 20.0 * np.log10(float(np.max(np.abs(corr_dbg)) + 1e-30))

        print(
            '[MMS stage rx_ant_pre_lna] '
            f'P_sig={fmt_dbw_dbm(p_sig_ant_pre_dbw)}, P_int={fmt_dbw_dbm(p_int_ant_pre_dbw)}, '
            f'Noise={fmt_dbw_dbm(p_noise_ant_pre_dbw)}, RSSI_total={fmt_dbw_dbm(p_total_ant_pre_dbw)}, '
            f'SIR={p_sig_ant_pre_dbw - p_int_ant_pre_dbw:.2f} dB, '
            f'SINR={p_sig_ant_pre_dbw - _w_to_db(p_int_ant_pre + p_noise_ant_pre):.2f} dB, '
            f'delta_total={delta_ant_pre_db:.2f} dB'
        )
        print(
            '[MMS stage rx_ant_post_lna] '
            f'P_sig={fmt_dbw_dbm(p_sig_ant_dbw)}, P_int={fmt_dbw_dbm(p_int_ant_dbw)}, '
            f'Noise={fmt_dbw_dbm(p_noise_dbw)}, RSSI_total={fmt_dbw_dbm(p_total_ant_dbw)}, '
            f'SIR={sir_ant_db:.2f} dB, SINR={sinr_ant_db:.2f} dB, delta_total={delta_ant_db:.2f} dB'
        )
        if lna_p1db_dbm is not None:
            print(
                "[MMS stage lna] "
                f"lna_gain_db={20.0*np.log10(max(abs(lna_gain_lin), 1e-30)):.2f}, "
                f"p1db_dbm={float(lna_p1db_dbm):.2f}, over_db={lna_over_db:.2f}"
            )
        if p_int_block_ant > p_int_ant * 1.2:
            print(f"[MMS stage rx_ant] blocker_rf_wide={fmt_dbw_dbm(_w_to_db(p_int_block_ant))}")
        print(
            '[MMS stage rf_selectivity] '
            f'P_sig={fmt_dbw_dbm(p_sig_dbw)}, P_int={fmt_dbw_dbm(p_int_dbw)}, '
            f'Noise={fmt_dbw_dbm(p_noise_sel_dbw)}, RSSI_total={fmt_dbw_dbm(p_total_sel_dbw)}, '
            f'SNR={snr_db:.2f} dB, SIR={sir_db:.2f} dB, SINR={sinr_db:.2f} dB, '
            f'alias_safe={not used_alias_safe_mode}, delta_total={delta_sel_db:.2f} dB'
        )
        print(
            '[MMS stage rx_adc] '
            f'agc_req_gain_db={20*np.log10(max(abs(g_req),1e-30)):.2f}, '
            f'agc_applied_gain_db={20*np.log10(max(abs(g_app),1e-30)):.2f}, clip_frac={clip_frac:.4f}, '
            f'P_preclip={p_pre_clip_dbfs:.2f} dBFS, P_postclip={p_post_clip_dbfs:.2f} dBFS, '
            f'RSSI_total={fmt_dbw_dbm(p_total_adc_dbw)}, '
            f'P_sig={fmt_dbw_dbm(p_sig_adc_dbw)}, P_int={fmt_dbw_dbm(p_int_adc_dbw)}, '
            f'Noise={fmt_dbw_dbm(p_noise_adc_dbw)}'
        )
        print(f'[MMS stage rx_det] corr_metric_db={corr_metric_db:.2f} dB (arb)')
        print(
            "[MMS stage detect] "
            f"quality_db={float(delay_info.get('quality_db', 0.0)):.2f}, "
            f"quality_min_db={quality_min_db:.2f}, "
            f"detect_ok={bool(delay_info.get('detect_ok', False))}, "
            f"reason={delay_info.get('reason', 'unknown')}"
        )
        print(
            "[MMS fp debug] "
            f"k_max={float(delay_info.get('k_max', 0.0)):.6f}, "
            f"k_max_refined={float(delay_info.get('k_max_refined', 0.0)):.6f}, "
            f"k_fp={float(delay_info.get('k_fp', 0.0)):.6f}, "
            f"fallback={bool(delay_info.get('fp_fallback', 0.0) > 0.5)}, "
            f"thr_db={float(delay_info.get('fp_thr_db', 0.0)):.2f}, "
            f"peak_frac={float(delay_info.get('fp_peak_frac', 0.0)):.3f}, "
            f"snr_corr_db={float(delay_info.get('fp_snr_corr_db', 0.0)):.2f}, "
            f"noise_floor={float(delay_info.get('fp_noise_floor', 0.0)):.3e}, "
            f"thr_noise_abs={float(delay_info.get('fp_thr_noise_abs', 0.0)):.3e}, "
            f"thr_peak_abs={float(delay_info.get('fp_thr_peak_abs', 0.0)):.3e}, "
            f"thr_abs={float(delay_info.get('fp_thr_abs', 0.0)):.3e}, "
            f"thr_mode={delay_info.get('fp_thr_mode', 'n/a')}, "
            f"fp_cross_refine={delay_info.get('fp_cross_refine', 'none')}, "
            f"slope_a={float(delay_info.get('fp_cross_slope', 0.0)):.3e}, "
            f"fit_resid={float(delay_info.get('fp_cross_resid', float('nan'))):.3e}, "
            f"fp_vs_max_db={float(delay_info.get('fp_peak_ratio_db', 0.0)):.2f}, "
            f"noise_win=[{float(delay_info.get('noise_win_start', 0.0)):.2f},"
            f"{float(delay_info.get('noise_win_end', 0.0)):.2f})"
        )
        print(
            "[MMS toa compare] "
            f"toa_peak_raw={float(delay_info.get('toa_peak_raw', 0.0)):.6f}, "
            f"toa_edge_raw={float(delay_info.get('toa_edge_raw', 0.0)):.6f}"
        )
        if used_alias_safe_mode:
            mode_txt = "leakage-equivalent" if use_leakage_equivalent_for_alias else "RF-attenuated waveform alias-mapped"
            print(f'[MMS alias][WARN] Interferer not alias-safe at victim fs; {mode_txt} path is used.')
        if ((not enable_adc_clipping and quant_bits is None) or clip_frac == 0.0) and has_interferer:
            diff_db = abs(p_int_eff_dbw - p_int_adc_dbw)
            print(f'[MMS sanity] |P_int_eff-P_int_adc|={diff_db:.2f} dB')
            if diff_db > 1.0:
                print('[MMS sanity][WARN] Non-clipping case but effective interference deviates >1 dB')
        if p_int_eff_dbw > (p_total_adc_dbw + 1.0):
            print('[MMS sanity][WARN] P_int_eff exceeds total stage power by >1 dB')
        if psd_prefix is not None:
            out = save_psd_triplet(
                desired=y_sig,
                interferer=y_int,
                mixed=rx_adc,
                fs=uwb_fs_hz,
                out_prefix=psd_prefix,
                y_label_unit=psd_unit,
                p_td_dbw=p_total_adc_dbw,
                noise_dbw_ref=p_noise_dbw,
                nf_db=nf_db,
            )
            print(f'[MMS leg debug] PSD saved: {out}')
            if psd_sanity_check:
                f_hz, psd_dbu = welch_psd_db_per_hz(rx_adc, fs=uwb_fs_hz)
                psd_w_hz = 10.0 ** (psd_dbu / 10.0)
                p_psd = float(np.trapz(psd_w_hz, f_hz))
                p_td = float(np.mean(np.abs(rx_adc) ** 2))
                err_db = 10.0 * np.log10((p_psd + 1e-30) / (p_td + 1e-30))
                print(
                    '[MMS leg debug] PSD sanity: '
                    f'P_psd={fmt_dbw_dbm(_w_to_db(p_psd))}, '
                    f'P_td={fmt_dbw_dbm(_w_to_db(p_td))}, delta={err_db:.2f} dB'
                )

    # No rounding: use k_peak + delta + calibration-correction.
    k_peak_delay = float(delay_info['delay_samples_int'])
    delta_delay = float(delay_info['delay_samples_frac'])
    toa_raw = k_peak_delay + delta_delay
    toa_samples_used = toa_raw + float(toa_calibration_samples)
    delay_corr = max(0.0, toa_samples_used)
    delay_int_corr = max(0.0, math.floor(delay_corr))
    delay_frac_corr = float(delay_corr - delay_int_corr)
    arr_rstu = float(delay_corr / (uwb_fs_hz * RSTU_S))
    if debug:
        k_peak = int(k_peak_delay)
        delta = float(delta_delay)
        print(
            "[MMS toa debug] "
            f"k_peak={k_peak}, delta={delta:.6f}, "
            f"toa_raw={toa_raw:.6f}, "
            f"toa_cal={float(toa_calibration_samples):.6f}, "
            f"k_fine={toa_raw:.6f}, "
            f"toa_samples_used={toa_samples_used:.6f}"
        )
    return arr_rstu, sinr_db, bool(delay_info['detect_ok']), {
        'p_sig_dbw': p_sig_dbw,
        'p_sig_ant_dbw': p_sig_ant_dbw,
        'p_int_ant_dbw': p_int_ant_dbw,
        'p_total_ant_dbw': p_total_ant_dbw,
        'p_int_dbw': p_int_dbw,
        'p_total_sel_dbw': p_total_sel_dbw,
        'p_int_eff_dbw': p_int_eff_dbw,
        'p_noise_dbw': p_noise_dbw,
        'p_noise_sel_dbw': p_noise_sel_dbw,
        'p_sig_adc_dbw': p_sig_adc_dbw,
        'p_int_adc_dbw': p_int_adc_dbw,
        'p_noise_adc_dbw': p_noise_adc_dbw,
        'rssi_sig_dbm_ant': dbw_to_dbm(p_sig_ant_dbw),
        'rssi_int_dbm_ant': dbw_to_dbm(p_int_ant_dbw),
        'rssi_total_dbm_ant': dbw_to_dbm(p_total_ant_dbw),
        'rssi_sig_dbm_sel': dbw_to_dbm(p_sig_dbw),
        'rssi_int_dbm_sel': dbw_to_dbm(p_int_dbw),
        'rssi_total_dbm_sel': dbw_to_dbm(p_total_sel_dbw),
        'rssi_total_dbm_adc': dbw_to_dbm(p_total_adc_dbw),
        'noise_dbm_ref_bw': dbw_to_dbm(p_noise_dbw),
        'snr_db': snr_db,
        'sir_db': sir_db,
        'sinr_db': sinr_db,
        'alias_safe_mode': float(1.0 if used_alias_safe_mode else 0.0),
        'agc_gain_db': 20.0 * np.log10(max(abs(g_app), 1e-30)),
        'clip_frac': clip_frac,
        'fail_reason': str(delay_info['reason']),
        'delay_samples_int': float(delay_int_corr),
        'delay_samples_frac': delay_frac_corr,
        'delay_samples': float(delay_corr),
        'peak_idx': float(delay_info.get('peak_idx', 0.0)),
        'peak_pos_f': float(delay_info.get('peak_pos_f', 0.0)),
        'toa_refine_fft_upsample': float(corr_upsample if toa_refine_method == "fft_upsample" else 1.0),
        'lna_gain_db': 20.0 * np.log10(max(abs(lna_gain_lin), 1e-30)),
    }


def simulate_mms_performance(
    distances_m: list[float] | tuple[float, ...] = (5.0, 10.0, 20.0, 40.0),
    n_trials: int = 200,
    rif_payload_bits: int = 256,
    fc_hz: float = 6.5e9,
    tx_eirp_dbw: float = -20.0,
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    seed: int = 1234,
    fp_alpha: float = 0.70,
    toa_refine_method: str = "fft_upsample",
    corr_upsample: int = 8,
    corr_win: int = 64,
    first_path: bool = True,
    first_path_thr_db: float = 13.0,
    first_path_peak_frac: float | None = 0.08,
    fp_use_adaptive_thr: bool = False,
    fp_snr_switch_db: float = 14.0,
    fp_thr_noise_cap_mult: float = 2.5,
    fp_thr_min_floor_mult: float = 3.0,
    first_path_search_back: int = 8,
    first_path_persist: int = 3,
    first_path_local_win: int = 8,
    nlos_alpha: float = 0.0,
    nlos_cap_ratio: float = 0.2,
    detector_mode: str = "first_path",
    cfar_guard: int = 8,
    cfar_train: int = 32,
    cfar_mult: float = 3.7,
    quality_min_db: float = 8.0,
    external_interference_penalty_db: float = 0.0,
    interference_wf: np.ndarray | None = None,
    interference_fs_hz: float | None = None,
    interference_fc_hz: float | None = None,
    interference_bw_hz: float | None = None,
    interference_block_dbw: float | None = None,
    interference_tx_eirp_dbw: float | None = None,
    interference_aclr_db: float = 35.0,
    use_leakage_equivalent_for_alias: bool = False,
    wifi_interference_on: bool = False,
    wifi_params: dict | None = None,
    uwb_fs_hz: float = 499.2e6,
    uwb_fc_hz: float | None = None,
    uwb_noise_bw_hz: float | None = None,
    rx_selectivity: dict | None = None,
    uwb_rx_selectivity: dict | None = None,
    adc_clip_db: float | None = None,
    enable_adc_clipping: bool = False,
    clip_dbfs: float = 0.0,
    enable_agc: bool = False,
    agc_stage: str = "post_selectivity",
    agc_target_dbfs: float = -12.0,
    agc_min_gain_db: float = -60.0,
    agc_max_gain_db: float = 60.0,
    lna_p1db_dbm: float | None = None,
    lna_max_gain_db: float = 0.0,
    quant_bits: int | None = None,
    debug_first_trial: bool = True,
    save_psd: bool = True,
    psd_unit: str = "dBm/MHz",
    psd_sanity_check: bool = False,
    psd_prefix_base: str = os.path.join("simulation", "mms", "psd_trial"),
    corr_debug_prefix_base: str | None = os.path.join("simulation", "mms", "corr_trial"),
    channel_pathloss_exp: float = 2.0,
    channel_delays_s: tuple[float, ...] = (0.0, 6e-9, 12e-9),
    channel_powers_db: tuple[float, ...] = (0.0, -6.0, -10.0),
    channel_k_factor_db: float = 8.0,
    clock_ppm_std: float = 0.0,
    baseline_sanity_mode: bool = False,
    enable_toa_calibration: bool = True,
    auto_calibrate: bool | None = None,
    toa_calibration_samples_override: float | None = None,
    range_bias_correction_m: float = 0.0,
    toa_calibration_distance_m: float | None = None,
    toa_calibration_trials: int = 32,
    enable_crc: bool = True,
    raw_mode: bool = False,
) -> list[dict]:
    """
    Simulate MMS UWB ranging/BER performance.

    Interference can be modeled by directly injecting a complex baseband waveform
    (`interference_wf`) and mapping it into UWB baseband via resampling + frequency shift.
    RX selectivity filtering and optional ADC clipping are applied before detection.
    """
    if uwb_fc_hz is None:
        uwb_fc_hz = float(fc_hz)
    if uwb_noise_bw_hz is None:
        uwb_noise_bw_hz = float(uwb_fs_hz)
    if baseline_sanity_mode:
        channel_delays_s = (0.0,)
        channel_powers_db = (0.0,)
        channel_k_factor_db = 30.0
        clock_ppm_std = 0.0
        nf_db = -200.0
        if detector_mode == "legacy":
            detector_mode = "first_path"
        quality_min_db = min(float(quality_min_db), 3.0)
    if detector_mode == "legacy":
        detector_mode = "first_path"
    if toa_refine_method not in ("parabolic", "fft_upsample"):
        raise ValueError("toa_refine_method must be 'parabolic' or 'fft_upsample'")
    if int(corr_upsample) < 1:
        raise ValueError("corr_upsample must be >= 1")
    if int(corr_win) < 4:
        raise ValueError("corr_win must be >= 4")
    if auto_calibrate is None:
        auto_calibrate = bool(enable_toa_calibration)

    cfg = MmsUwbConfig(phy_uwb_mms_rsf_number_frags=2, phy_uwb_mms_rif_number_frags=1)
    tx_template = _build_rsf_template(cfg)
    ref_template = tx_template.astype(np.complex128)

    wifi_params = {} if wifi_params is None else dict(wifi_params)
    sel_cfg = {} if rx_selectivity is None else dict(rx_selectivity)
    if uwb_rx_selectivity is not None:
        sel_cfg.update(uwb_rx_selectivity)
    sel_norm = _normalize_rx_selectivity(sel_cfg)
    h_sel = _design_rx_selectivity_fir(fs_hz=uwb_fs_hz, rx_selectivity=sel_norm)
    if adc_clip_db is not None:
        enable_adc_clipping = True
        clip_dbfs = float(adc_clip_db)
    if agc_stage not in ("pre_selectivity", "post_selectivity"):
        raise ValueError("agc_stage must be 'pre_selectivity' or 'post_selectivity'")

    if (interference_wf is not None) and (interference_tx_eirp_dbw is not None):
        p_now = float(np.mean(np.abs(interference_wf) ** 2)) + 1e-30
        p_tgt = _db_to_w(float(interference_tx_eirp_dbw))
        interference_wf = np.asarray(interference_wf, dtype=np.complex128) * np.sqrt(p_tgt / p_now)

    toa_cal_firstpath = 0.0
    toa_cal_strongest = 0.0
    toa_calibration_samples = 0.0
    cal_dist_m = float(toa_calibration_distance_m) if toa_calibration_distance_m is not None else float(distances_m[0])

    if toa_calibration_samples_override is not None:
        toa_calibration_samples = float(toa_calibration_samples_override)
        if first_path:
            toa_cal_firstpath = toa_calibration_samples
        else:
            toa_cal_strongest = toa_calibration_samples
    elif bool(auto_calibrate):
        toa_cal_firstpath = auto_calibrate_toa_offset(
            tx_template=tx_template,
            ref_template=ref_template,
            rx_selectivity_h=h_sel,
            fs_hz=uwb_fs_hz,
            fc_hz=uwb_fc_hz,
            detector_mode=detector_mode,
            fp_alpha=fp_alpha,
            nlos_alpha=nlos_alpha,
            nlos_cap_ratio=nlos_cap_ratio,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
            quality_min_db=quality_min_db,
            toa_refine_method=toa_refine_method,
            corr_upsample=corr_upsample,
            corr_win=corr_win,
            first_path=True,
            first_path_thr_db=first_path_thr_db,
            first_path_peak_frac=first_path_peak_frac,
            fp_use_adaptive_thr=fp_use_adaptive_thr,
            fp_snr_switch_db=fp_snr_switch_db,
            fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=fp_thr_min_floor_mult,
            first_path_search_back=first_path_search_back,
            first_path_persist=first_path_persist,
            first_path_local_win=first_path_local_win,
            calibration_distance_m=cal_dist_m,
            calibration_trials=toa_calibration_trials,
            nf_db=nf_db,
            temperature_k=temperature_k,
            noise_bw_hz=uwb_noise_bw_hz,
        )
        toa_cal_strongest = auto_calibrate_toa_offset(
            tx_template=tx_template,
            ref_template=ref_template,
            rx_selectivity_h=h_sel,
            fs_hz=uwb_fs_hz,
            fc_hz=uwb_fc_hz,
            detector_mode=detector_mode,
            fp_alpha=fp_alpha,
            nlos_alpha=nlos_alpha,
            nlos_cap_ratio=nlos_cap_ratio,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
            quality_min_db=quality_min_db,
            toa_refine_method=toa_refine_method,
            corr_upsample=corr_upsample,
            corr_win=corr_win,
            first_path=False,
            first_path_thr_db=first_path_thr_db,
            first_path_peak_frac=first_path_peak_frac,
            fp_use_adaptive_thr=fp_use_adaptive_thr,
            fp_snr_switch_db=fp_snr_switch_db,
            fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=fp_thr_min_floor_mult,
            first_path_search_back=first_path_search_back,
            first_path_persist=first_path_persist,
            first_path_local_win=first_path_local_win,
            calibration_distance_m=cal_dist_m,
            calibration_trials=toa_calibration_trials,
            nf_db=nf_db,
            temperature_k=temperature_k,
            noise_bw_hz=uwb_noise_bw_hz,
        )
        toa_calibration_samples = toa_cal_firstpath if first_path else toa_cal_strongest
    elif enable_toa_calibration:
        toa_calibration_samples = _estimate_toa_calibration_samples(
            ref_template=ref_template,
            rx_selectivity_h=h_sel,
            detector_mode=detector_mode,
            fp_alpha=fp_alpha,
            nlos_alpha=nlos_alpha,
            nlos_cap_ratio=nlos_cap_ratio,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
            quality_min_db=quality_min_db,
            toa_refine_method=toa_refine_method,
            corr_upsample=corr_upsample,
            corr_win=corr_win,
            first_path=first_path,
            first_path_thr_db=first_path_thr_db,
            first_path_peak_frac=first_path_peak_frac,
            fp_use_adaptive_thr=fp_use_adaptive_thr,
            fp_snr_switch_db=fp_snr_switch_db,
            fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=fp_thr_min_floor_mult,
            first_path_search_back=first_path_search_back,
            first_path_persist=first_path_persist,
            first_path_local_win=first_path_local_win,
        )
        if first_path:
            toa_cal_firstpath = toa_calibration_samples
        else:
            toa_cal_strongest = toa_calibration_samples

    wifi_tx = None
    if wifi_interference_on:
        WiFiOFDMTx = _load_wifi_tx_class()
        wifi_fc = float(wifi_params.get("fc_hz", 6.49e9))
        wifi_tx = WiFiOFDMTx(rng_seed=seed + 777, center_freq_hz=wifi_fc)

    phase_start = 100_000
    i_frags = build_mms_uwb_fragments("initiator", phase_start, cfg)
    r_frags = build_mms_uwb_fragments("responder", phase_start, cfg)
    i_rsf = [f.rmarker_rstu for f in i_frags if f.kind == "RSF"][:2]
    r_rsf = [f.rmarker_rstu for f in r_frags if f.kind == "RSF"][:2]

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for d in distances_m:
        bit_err_sum = 0
        bits_sum = n_trials * rif_payload_bits
        frame_err = 0
        dist_err_sum = 0.0
        dist_err_sq_sum = 0.0
        dist_err_all_sum = 0.0
        dist_err_sq_all_sum = 0.0
        dist_all_count = 0
        dist_success_count = 0
        ranging_fail = 0
        sinr_acc = 0.0
        fail_reason_counts: dict[str, int] = {}

        for k in range(n_trials):
            trial_seed = seed + 10000 * int(d) + k
            d_f = float(d)
            debug = bool(debug_first_trial and k == 0)
            if debug:
                print(
                    "[MMS config debug] "
                    f"toa_calibration_samples={toa_calibration_samples:.4f}, "
                    f"toa_cal_firstpath={toa_cal_firstpath:.4f}, "
                    f"toa_cal_strongest={toa_cal_strongest:.4f}, "
                    f"toa_calibration_trials={int(toa_calibration_trials)}, "
                    f"toa_mode={'first_path' if first_path else 'strongest_peak'}, "
                    f"baseline_sanity_mode={baseline_sanity_mode}, "
                    f"clock_ppm_std={clock_ppm_std:.3f}, raw_mode={raw_mode}"
                )
            psd_prefix = None
            if debug and save_psd:
                psd_prefix = f"{psd_prefix_base}_d{int(d_f)}"
            corr_csv = None
            if debug and corr_debug_prefix_base is not None:
                corr_csv = f"{corr_debug_prefix_base}_d{int(d_f)}.csv"

            ppm_i = float(rng.normal(0.0, clock_ppm_std))
            ppm_r = float(rng.normal(0.0, clock_ppm_std))
            scale_i = 1.0 + ppm_i * 1e-6
            scale_r = 1.0 + ppm_r * 1e-6

            a2b_1, sinr1, ok1, info1 = _simulate_leg(
                tx_template=tx_template,
                ref_template=ref_template,
                uwb_fs_hz=uwb_fs_hz,
                uwb_fc_hz=uwb_fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                noise_bw_hz=uwb_noise_bw_hz,
                pathloss_exp=channel_pathloss_exp,
                delays_s=channel_delays_s,
                powers_db=channel_powers_db,
                k_factor_db=channel_k_factor_db,
                channel_seed=trial_seed + 101,
                noise_seed=trial_seed + 201,
                detector_mode=detector_mode,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
                quality_min_db=quality_min_db,
                toa_refine_method=toa_refine_method,
                corr_upsample=corr_upsample,
                corr_win=corr_win,
                first_path=first_path,
                first_path_thr_db=first_path_thr_db,
                first_path_peak_frac=first_path_peak_frac,
                fp_use_adaptive_thr=fp_use_adaptive_thr,
                fp_snr_switch_db=fp_snr_switch_db,
                fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
                fp_thr_min_floor_mult=fp_thr_min_floor_mult,
                first_path_search_back=first_path_search_back,
                first_path_persist=first_path_persist,
                first_path_local_win=first_path_local_win,
                rx_selectivity_h=h_sel,
                rx_stop_atten_db=float(sel_norm["stop_atten_db"]),
                interference_wf=interference_wf,
                interference_fs_hz=interference_fs_hz,
                interference_fc_hz=interference_fc_hz,
                interference_bw_hz=interference_bw_hz,
                interference_aclr_db=interference_aclr_db,
                interference_block_dbw=interference_block_dbw,
                use_leakage_equivalent_for_alias=use_leakage_equivalent_for_alias,
                wifi_interference_on=wifi_interference_on,
                wifi_tx=wifi_tx,
                wifi_params=wifi_params,
                agc_stage=agc_stage,
                enable_agc=enable_agc,
                agc_target_dbfs=agc_target_dbfs,
                enable_adc_clipping=enable_adc_clipping,
                clip_dbfs=clip_dbfs,
                quant_bits=quant_bits,
                agc_min_gain_db=agc_min_gain_db,
                agc_max_gain_db=agc_max_gain_db,
                lna_p1db_dbm=lna_p1db_dbm,
                lna_max_gain_db=lna_max_gain_db,
                debug=debug,
                psd_prefix=psd_prefix,
                psd_unit=psd_unit,
                psd_sanity_check=psd_sanity_check,
                toa_calibration_samples=toa_calibration_samples,
                corr_debug_csv_path=corr_csv,
            )
            b2a_1, sinr2, ok2, info2 = _simulate_leg(
                tx_template=tx_template,
                ref_template=ref_template,
                uwb_fs_hz=uwb_fs_hz,
                uwb_fc_hz=uwb_fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                noise_bw_hz=uwb_noise_bw_hz,
                pathloss_exp=channel_pathloss_exp,
                delays_s=channel_delays_s,
                powers_db=channel_powers_db,
                k_factor_db=channel_k_factor_db,
                channel_seed=trial_seed + 301,
                noise_seed=trial_seed + 401,
                detector_mode=detector_mode,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
                quality_min_db=quality_min_db,
                toa_refine_method=toa_refine_method,
                corr_upsample=corr_upsample,
                corr_win=corr_win,
                first_path=first_path,
                first_path_thr_db=first_path_thr_db,
                first_path_peak_frac=first_path_peak_frac,
                fp_use_adaptive_thr=fp_use_adaptive_thr,
                fp_snr_switch_db=fp_snr_switch_db,
                fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
                fp_thr_min_floor_mult=fp_thr_min_floor_mult,
                first_path_search_back=first_path_search_back,
                first_path_persist=first_path_persist,
                first_path_local_win=first_path_local_win,
                rx_selectivity_h=h_sel,
                rx_stop_atten_db=float(sel_norm["stop_atten_db"]),
                interference_wf=interference_wf,
                interference_fs_hz=interference_fs_hz,
                interference_fc_hz=interference_fc_hz,
                interference_bw_hz=interference_bw_hz,
                interference_aclr_db=interference_aclr_db,
                interference_block_dbw=interference_block_dbw,
                use_leakage_equivalent_for_alias=use_leakage_equivalent_for_alias,
                wifi_interference_on=wifi_interference_on,
                wifi_tx=wifi_tx,
                wifi_params=wifi_params,
                agc_stage=agc_stage,
                enable_agc=enable_agc,
                agc_target_dbfs=agc_target_dbfs,
                enable_adc_clipping=enable_adc_clipping,
                clip_dbfs=clip_dbfs,
                quant_bits=quant_bits,
                agc_min_gain_db=agc_min_gain_db,
                agc_max_gain_db=agc_max_gain_db,
                lna_p1db_dbm=lna_p1db_dbm,
                lna_max_gain_db=lna_max_gain_db,
                debug=False,
                psd_prefix=None,
                psd_unit=psd_unit,
                psd_sanity_check=False,
                toa_calibration_samples=toa_calibration_samples,
                corr_debug_csv_path=None,
            )
            a2b_2, sinr3, ok3, info3 = _simulate_leg(
                tx_template=tx_template,
                ref_template=ref_template,
                uwb_fs_hz=uwb_fs_hz,
                uwb_fc_hz=uwb_fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                noise_bw_hz=uwb_noise_bw_hz,
                pathloss_exp=channel_pathloss_exp,
                delays_s=channel_delays_s,
                powers_db=channel_powers_db,
                k_factor_db=channel_k_factor_db,
                channel_seed=trial_seed + 501,
                noise_seed=trial_seed + 601,
                detector_mode=detector_mode,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
                quality_min_db=quality_min_db,
                toa_refine_method=toa_refine_method,
                corr_upsample=corr_upsample,
                corr_win=corr_win,
                first_path=first_path,
                first_path_thr_db=first_path_thr_db,
                first_path_peak_frac=first_path_peak_frac,
                fp_use_adaptive_thr=fp_use_adaptive_thr,
                fp_snr_switch_db=fp_snr_switch_db,
                fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
                fp_thr_min_floor_mult=fp_thr_min_floor_mult,
                first_path_search_back=first_path_search_back,
                first_path_persist=first_path_persist,
                first_path_local_win=first_path_local_win,
                rx_selectivity_h=h_sel,
                rx_stop_atten_db=float(sel_norm["stop_atten_db"]),
                interference_wf=interference_wf,
                interference_fs_hz=interference_fs_hz,
                interference_fc_hz=interference_fc_hz,
                interference_bw_hz=interference_bw_hz,
                interference_aclr_db=interference_aclr_db,
                interference_block_dbw=interference_block_dbw,
                use_leakage_equivalent_for_alias=use_leakage_equivalent_for_alias,
                wifi_interference_on=wifi_interference_on,
                wifi_tx=wifi_tx,
                wifi_params=wifi_params,
                agc_stage=agc_stage,
                enable_agc=enable_agc,
                agc_target_dbfs=agc_target_dbfs,
                enable_adc_clipping=enable_adc_clipping,
                clip_dbfs=clip_dbfs,
                quant_bits=quant_bits,
                agc_min_gain_db=agc_min_gain_db,
                agc_max_gain_db=agc_max_gain_db,
                lna_p1db_dbm=lna_p1db_dbm,
                lna_max_gain_db=lna_max_gain_db,
                debug=False,
                psd_prefix=None,
                psd_unit=psd_unit,
                psd_sanity_check=False,
                toa_calibration_samples=toa_calibration_samples,
                corr_debug_csv_path=None,
            )
            b2a_2, sinr4, ok4, info4 = _simulate_leg(
                tx_template=tx_template,
                ref_template=ref_template,
                uwb_fs_hz=uwb_fs_hz,
                uwb_fc_hz=uwb_fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                noise_bw_hz=uwb_noise_bw_hz,
                pathloss_exp=channel_pathloss_exp,
                delays_s=channel_delays_s,
                powers_db=channel_powers_db,
                k_factor_db=channel_k_factor_db,
                channel_seed=trial_seed + 701,
                noise_seed=trial_seed + 801,
                detector_mode=detector_mode,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
                quality_min_db=quality_min_db,
                toa_refine_method=toa_refine_method,
                corr_upsample=corr_upsample,
                corr_win=corr_win,
                first_path=first_path,
                first_path_thr_db=first_path_thr_db,
                first_path_peak_frac=first_path_peak_frac,
                fp_use_adaptive_thr=fp_use_adaptive_thr,
                fp_snr_switch_db=fp_snr_switch_db,
                fp_thr_noise_cap_mult=fp_thr_noise_cap_mult,
                fp_thr_min_floor_mult=fp_thr_min_floor_mult,
                first_path_search_back=first_path_search_back,
                first_path_persist=first_path_persist,
                first_path_local_win=first_path_local_win,
                rx_selectivity_h=h_sel,
                rx_stop_atten_db=float(sel_norm["stop_atten_db"]),
                interference_wf=interference_wf,
                interference_fs_hz=interference_fs_hz,
                interference_fc_hz=interference_fc_hz,
                interference_bw_hz=interference_bw_hz,
                interference_aclr_db=interference_aclr_db,
                interference_block_dbw=interference_block_dbw,
                use_leakage_equivalent_for_alias=use_leakage_equivalent_for_alias,
                wifi_interference_on=wifi_interference_on,
                wifi_tx=wifi_tx,
                wifi_params=wifi_params,
                agc_stage=agc_stage,
                enable_agc=enable_agc,
                agc_target_dbfs=agc_target_dbfs,
                enable_adc_clipping=enable_adc_clipping,
                clip_dbfs=clip_dbfs,
                quant_bits=quant_bits,
                agc_min_gain_db=agc_min_gain_db,
                agc_max_gain_db=agc_max_gain_db,
                lna_p1db_dbm=lna_p1db_dbm,
                lna_max_gain_db=lna_max_gain_db,
                debug=False,
                psd_prefix=None,
                psd_unit=psd_unit,
                psd_sanity_check=False,
                toa_calibration_samples=toa_calibration_samples,
                corr_debug_csv_path=None,
            )

            sinr_db = 0.25 * (sinr1 + sinr2 + sinr3 + sinr4)
            # Deprecated: keep API compatibility but do not use synthetic penalty in waveform mode.
            _ = external_interference_penalty_db
            sinr_acc += sinr_db

            eff_snr_db = sinr_db - 3.0
            ber = _ber_bpsk_awgn(eff_snr_db)
            p_sync = 1.0 / (1.0 + math.exp(-(eff_snr_db - 2.0) / 1.5))
            phy_sync_success = (rng.random() < p_sync)
            fcs_check_on = bool(enable_crc and (not raw_mode))

            tx_payload = rng.integers(0, 2, rif_payload_bits, dtype=int)
            tx_bytes = np.packbits(tx_payload.astype(np.uint8), bitorder="little").tobytes()
            tx_frame = _append_fcs(tx_bytes)
            tx_payload_nbytes = len(tx_bytes)
            tx_frame_bits = np.unpackbits(np.frombuffer(tx_frame, dtype=np.uint8), bitorder="little").astype(int)
            if not phy_sync_success:
                rx_frame_bits = np.zeros_like(tx_frame_bits)
            else:
                flips = rng.binomial(1, min(max(ber, 0.0), 1.0), size=tx_frame_bits.size).astype(int)
                rx_frame_bits = tx_frame_bits ^ flips

            rx_payload_bits = rx_frame_bits[:rif_payload_bits]
            bit_err = int(np.sum(rx_payload_bits != tx_payload))
            bits_ok = bool(rx_frame_bits.size >= tx_frame_bits.size)
            crc_checked = False
            crc_pass = False
            crc_rx = None
            crc_calc = None
            if fcs_check_on and phy_sync_success and bits_ok:
                rx_frame_bytes = np.packbits(rx_frame_bits[: tx_frame_bits.size].astype(np.uint8), bitorder="little").tobytes()
                rx_payload_bytes = rx_frame_bytes[:tx_payload_nbytes]
                rx_fcs_bytes = rx_frame_bytes[tx_payload_nbytes : tx_payload_nbytes + 2]
                crc_calc = crc16_802154(rx_payload_bytes)
                crc_rx = int.from_bytes(rx_fcs_bytes.ljust(2, b"\x00"), "little")
                crc_pass = bool(crc_calc == crc_rx)
                crc_checked = True
            crc_ok = (not fcs_check_on) or (crc_checked and crc_pass)
            payload_invalid = (not crc_ok) if fcs_check_on else (bit_err > 0)
            frame_success = bool(phy_sync_success and crc_ok)
            frame_failed = bool(not frame_success)
            if debug:
                assert not (crc_pass and not crc_checked)
                assert not ((crc_checked is False) and (crc_rx is not None))
                print(
                    "[MMS frame gate] "
                    f"phy_sync_success={phy_sync_success}, bits_ok={bits_ok}, "
                    f"crc_checked={crc_checked}, crc_pass={crc_pass}, crc_ok={crc_ok}, "
                    f"frame_success={frame_success}"
                )

            i_tx1 = float(i_rsf[0])
            i_tx2 = float(i_rsf[1])
            r_tx1 = float(r_rsf[0])
            r_tx2 = float(r_rsf[1])
            r_rx1 = (i_tx1 + a2b_1) * scale_r
            r_rx2 = (i_tx2 + a2b_2) * scale_r
            i_rx1 = (r_tx1 + b2a_1) * scale_i
            i_rx2 = (r_tx2 + b2a_2) * scale_i
            r_tx1_l = r_tx1 * scale_r
            r_tx2_l = r_tx2 * scale_r
            i_tx1_l = i_tx1 * scale_i
            i_tx2_l = i_tx2 * scale_i

            reply1 = r_tx1 - r_rx1 / scale_r
            reply2 = r_tx2 - r_rx2 / scale_r
            if debug:
                q_m = (float(info1["delay_samples_frac"]) / uwb_fs_hz) * C_MPS
                quant_floor_m = (C_MPS / uwb_fs_hz) / math.sqrt(12.0) / max(1, int(corr_upsample))
                print(
                    "[MMS timing debug] "
                    f"ppm_i={ppm_i:.3f}, ppm_r={ppm_r:.3f}, reply_rstu~({reply1:.2f},{reply2:.2f}), "
                    f"toa_int={info1['delay_samples_int']:.3f}, toa_frac={info1['delay_samples_frac']:.3f}, "
                    f"toa_frac_m={q_m:.4f}, toa_refine_method={toa_refine_method}, first_path={first_path}, "
                    f"expected_quant_floor_1way_m={quant_floor_m:.4f}, "
                    f"multipath_delays_ns={[round(x*1e9,2) for x in channel_delays_s]}"
                )

            phy_ok = bool(ok1 and ok2 and ok3 and ok4)
            if not phy_ok:
                ranging_fail += 1
                frame_failed = True
                bit_err = rif_payload_bits
                if not ok1:
                    r = str(info1.get("fail_reason", "unknown"))
                    fail_reason_counts[r] = fail_reason_counts.get(r, 0) + 1
                if not ok2:
                    r = str(info2.get("fail_reason", "unknown"))
                    fail_reason_counts[r] = fail_reason_counts.get(r, 0) + 1
                if not ok3:
                    r = str(info3.get("fail_reason", "unknown"))
                    fail_reason_counts[r] = fail_reason_counts.get(r, 0) + 1
                if not ok4:
                    r = str(info4.get("fail_reason", "unknown"))
                    fail_reason_counts[r] = fail_reason_counts.get(r, 0) + 1
                if debug:
                    print(
                        "[MMS fail debug] "
                        f"ok=(1:{ok1},2:{ok2},3:{ok3},4:{ok4}), "
                        f"reason=(1:{info1.get('fail_reason')},2:{info2.get('fail_reason')},"
                        f"3:{info3.get('fail_reason')},4:{info4.get('fail_reason')})"
                    )
            else:
                tof_est_rstu = _ds_twr_tof_rstu(i_tx1_l, i_rx1, r_rx1, r_tx1_l, i_tx2_l, i_rx2, r_rx2, r_tx2_l)
                d_est = (tof_est_rstu * RSTU_S * C_MPS) - float(range_bias_correction_m)
                err = (d_est - d_f)
                dist_err_all_sum += err
                dist_err_sq_all_sum += (d_est - d_f) ** 2
                dist_all_count += 1
                ranging_valid = (not frame_failed)
                if ranging_valid:
                    dist_err_sum += err
                    dist_err_sq_sum += err ** 2
                    dist_success_count += 1
                else:
                    ranging_fail += 1
                    fail_reason_counts["payload_or_crc_fail"] = fail_reason_counts.get("payload_or_crc_fail", 0) + 1
                    if debug:
                        print(
                            "[MMS fail debug] "
                            f"phy_sync_success={phy_sync_success}, bits_ok={bits_ok}, "
                            f"fcs_check_on={fcs_check_on}, crc_checked={crc_checked}, "
                            f"crc_pass={crc_pass}, crc_ok={crc_ok}, frame_success={frame_success}, bit_err={bit_err}, "
                            f"crc_rx={('N/A' if crc_rx is None else f'0x{crc_rx:04x}')}, "
                            f"crc_calc={('N/A' if crc_calc is None else f'0x{crc_calc:04x}')}"
                        )

            bit_err_sum += bit_err
            if frame_failed:
                frame_err += 1

        rmse = float("nan") if dist_success_count == 0 else math.sqrt(dist_err_sq_sum / dist_success_count)
        rmse_all = float("nan") if dist_all_count == 0 else math.sqrt(dist_err_sq_all_sum / dist_all_count)
        bias = float("nan")
        std = float("nan")
        bias_all = float("nan")
        std_all = float("nan")
        if dist_success_count > 0:
            bias = dist_err_sum / dist_success_count
            var = max(0.0, (dist_err_sq_sum / dist_success_count) - (bias ** 2))
            std = math.sqrt(var)
        if dist_all_count > 0:
            bias_all = dist_err_all_sum / dist_all_count
            var_all = max(0.0, (dist_err_sq_all_sum / dist_all_count) - (bias_all ** 2))
            std_all = math.sqrt(var_all)
        out.append(
            {
                "distance_m": float(d),
                "snr_db_avg": sinr_acc / max(n_trials, 1),
                "ber": bit_err_sum / max(bits_sum, 1),
                "fer": frame_err / max(n_trials, 1),
                "ranging_rmse_m": rmse,
                "ranging_rmse_all_m": rmse_all,
                "ranging_bias_m": float(bias),
                "ranging_std_m": float(std),
                "ranging_bias_all_m": float(bias_all),
                "ranging_std_all_m": float(std_all),
                "ranging_fail_rate": ranging_fail / max(n_trials, 1),
                "ranging_success_count": int(dist_success_count),
                "fail_reason_counts": fail_reason_counts,
                "toa_calibration_samples": float(toa_calibration_samples),
                "toa_cal_firstpath": float(toa_cal_firstpath),
                "toa_cal_strongest": float(toa_cal_strongest),
            }
        )
    return out
