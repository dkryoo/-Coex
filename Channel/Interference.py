from __future__ import annotations

import os
import numpy as np


def resample_complex_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("fs_in and fs_out must be > 0")
    x = np.asarray(x, dtype=np.complex128).flatten()
    if x.size <= 1:
        return x.astype(np.complex128)
    t_in = np.arange(x.size, dtype=float) / fs_in
    n_out = max(1, int(np.floor((x.size - 1) * fs_out / fs_in)) + 1)
    t_out = np.arange(n_out, dtype=float) / fs_out
    re = np.interp(t_out, t_in, np.real(x))
    im = np.interp(t_out, t_in, np.imag(x))
    return (re + 1j * im).astype(np.complex128)


def freq_shift(x: np.ndarray, fs: float, f_off_hz: float) -> np.ndarray:
    if fs <= 0:
        raise ValueError("fs must be > 0")
    x = np.asarray(x, dtype=np.complex128).flatten()
    t = np.arange(x.size, dtype=float) / fs
    return (x * np.exp(1j * 2.0 * np.pi * f_off_hz * t)).astype(np.complex128)


def alias_frequency(f_off_hz: float, fs_hz: float) -> float:
    """
    Map RF/baseband offset into sampled digital frequency in (-fs/2, fs/2].
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    return float(((float(f_off_hz) + fs_hz / 2.0) % fs_hz) - fs_hz / 2.0)


def design_selectivity_fir(
    fs: float,
    passband_hz: float,
    stopband_hz: float,
    stopband_atten_db: float,
    taps: int,
) -> np.ndarray:
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if taps < 9 or taps % 2 == 0:
        raise ValueError("taps must be odd and >= 9")
    if not (0 < passband_hz < stopband_hz < fs / 2):
        raise ValueError("Require 0 < passband_hz < stopband_hz < fs/2")

    # Frequency-sampling FIR with explicit finite stopband floor so
    # stopband_atten_db directly controls adjacent-channel leakage.
    nfft = int(2 ** np.ceil(np.log2(max(1024, 16 * taps))))
    f = np.fft.fftfreq(nfft, d=1.0 / fs)
    fa = np.abs(f)

    a_sb = 10.0 ** (-float(stopband_atten_db) / 20.0)
    h_mag = np.full(nfft, a_sb, dtype=float)

    in_pb = fa <= passband_hz
    h_mag[in_pb] = 1.0

    in_tr = (fa > passband_hz) & (fa < stopband_hz)
    if np.any(in_tr):
        u = (fa[in_tr] - passband_hz) / (stopband_hz - passband_hz)
        # Cosine transition 1 -> a_sb.
        h_mag[in_tr] = a_sb + (1.0 - a_sb) * 0.5 * (1.0 + np.cos(np.pi * u))

    # Linear-phase real FIR.
    h_full = np.fft.ifft(h_mag).real
    h_full = np.fft.fftshift(h_full)
    c = nfft // 2
    s = c - taps // 2
    h = h_full[s : s + taps]

    # Taper impulse to reduce truncation ripple and normalize DC gain.
    h *= np.kaiser(taps, beta=6.0)
    h /= (np.sum(h) + 1e-30)
    return h.astype(float)


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=np.complex128), np.asarray(h, dtype=float), mode="same").astype(
        np.complex128
    )


def apply_soft_clipping(x: np.ndarray, clip_dbfs: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).flatten()
    clip_amp = 10.0 ** (clip_dbfs / 20.0)
    r = np.abs(x)
    # Smooth tanh clipping on magnitude; preserve phase.
    gain = np.tanh(r / (clip_amp + 1e-30)) / ((r / (clip_amp + 1e-30)) + 1e-30)
    y = x * gain
    return y.astype(np.complex128)


def welch_psd_db_per_hz(
    x: np.ndarray,
    fs: float,
    nfft: int = 4096,
    seg_len: int = 1024,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.complex128).flatten()
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if x.size < seg_len:
        x = np.pad(x, (0, seg_len - x.size))
    hop = max(1, int(seg_len * (1.0 - overlap)))
    w = np.hanning(seg_len)
    w_pow = np.sum(w**2) + 1e-30
    acc = np.zeros(nfft, dtype=float)
    count = 0
    for s in range(0, x.size - seg_len + 1, hop):
        seg = x[s : s + seg_len] * w
        X = np.fft.fft(seg, nfft)
        psd = (np.abs(X) ** 2) / (fs * w_pow)
        acc += psd
        count += 1
    if count == 0:
        count = 1
    psd_avg = acc / count
    psd_shift = np.fft.fftshift(psd_avg)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return f, 10.0 * np.log10(psd_shift + 1e-30)


def save_psd_triplet(
    desired: np.ndarray,
    interferer: np.ndarray,
    mixed: np.ndarray,
    fs: float,
    out_prefix: str,
    y_label_unit: str = "dBm/MHz",
    p_td_dbw: float | None = None,
    noise_dbw_ref: float | None = None,
    nf_db: float | None = None,
) -> dict[str, str]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    f, pd_dbw_hz = welch_psd_db_per_hz(desired, fs=fs)
    _, pi_dbw_hz = welch_psd_db_per_hz(interferer, fs=fs)
    _, pm_dbw_hz = welch_psd_db_per_hz(mixed, fs=fs)

    unit = str(y_label_unit).strip().lower()
    if unit == "dbw/hz":
        scale = 0.0
        hdr_unit = "dBW_per_Hz"
    elif unit == "dbm/hz":
        scale = 30.0
        hdr_unit = "dBm_per_Hz"
    elif unit == "dbw/mhz":
        scale = 60.0
        hdr_unit = "dBW_per_MHz"
    else:
        # default + fallback
        scale = 90.0
        hdr_unit = "dBm_per_MHz"

    pd = pd_dbw_hz + scale
    pi = pi_dbw_hz + scale
    pm = pm_dbw_hz + scale

    csv_path = out_prefix + ".csv"
    mat = np.column_stack([f, pd, pi, pm])
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header=f"freq_hz,psd_desired_{hdr_unit},psd_interferer_{hdr_unit},psd_mixed_{hdr_unit}",
        comments="",
    )
    out = {"csv": csv_path}

    try:
        import matplotlib.pyplot as plt

        png_path = out_prefix + ".png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(f / 1e6, pd, label="desired", lw=1.0)
        ax.plot(f / 1e6, pi, label="interferer", lw=1.0)
        ax.plot(f / 1e6, pm, label="mixed", lw=1.0)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel(f"PSD [{y_label_unit}]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Add sanity / interpretation notes directly on plot.
        p_psd_w = float(np.trapz(10.0 ** (pm_dbw_hz / 10.0), f))
        p_psd_dbw = 10.0 * np.log10(max(p_psd_w, 1e-30))
        text_lines = [f"P_psd={p_psd_dbw:.2f} dBW ({p_psd_dbw+30.0:.2f} dBm)"]
        if p_td_dbw is not None:
            delta = p_psd_dbw - float(p_td_dbw)
            text_lines.insert(0, f"P_td={p_td_dbw:.2f} dBW ({p_td_dbw+30.0:.2f} dBm)")
            text_lines.append(f"delta={delta:.2f} dB")
        if nf_db is not None:
            nf = float(nf_db)
            text_lines.append(f"Thermal floor ~ {-174.0 + nf:.1f} dBm/Hz")
        if noise_dbw_ref is not None:
            ndbw = float(noise_dbw_ref)
            text_lines.append(f"Noise(ref)={ndbw:.2f} dBW ({ndbw+30.0:.2f} dBm)")
        ax.text(
            0.01,
            0.99,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        out["png"] = png_path
    except Exception:
        pass
    return out
