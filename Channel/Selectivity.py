from __future__ import annotations

import numpy as np


def design_lowpass_fir(
    fs_hz: float,
    pass_hz: float,
    stop_hz: float,
    stop_atten_db: float,
    num_taps: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Design low-pass FIR via windowed-sinc.
    Returns (h, group_delay_samples).
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if not (0 < pass_hz < stop_hz < fs_hz / 2):
        raise ValueError("Require 0 < pass_hz < stop_hz < fs_hz/2")

    if num_taps is None:
        if stop_atten_db >= 55:
            num_taps = 1025
        elif stop_atten_db >= 35:
            num_taps = 513
        else:
            num_taps = 257
    if num_taps % 2 == 0:
        num_taps += 1

    fc = 0.5 * (pass_hz + stop_hz)
    n = np.arange(num_taps, dtype=float) - (num_taps - 1) / 2.0
    h = 2.0 * fc / fs_hz * np.sinc(2.0 * fc / fs_hz * n)

    if stop_atten_db >= 55:
        w = np.blackman(num_taps)
    elif stop_atten_db >= 30:
        w = np.hanning(num_taps)
    else:
        w = np.ones(num_taps, dtype=float)
    h *= w
    h /= (np.sum(h) + 1e-30)
    gd = (num_taps - 1) // 2
    return h.astype(float), int(gd)


def apply_fir_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=np.complex128), np.asarray(h, dtype=float), mode="same").astype(
        np.complex128
    )


def gain_at_offset_db(h: np.ndarray, fs_hz: float, f_off_hz: float, nfft: int = 8192) -> float:
    h = np.asarray(h, dtype=float).flatten()
    H = np.fft.fft(h, n=nfft)
    f = np.fft.fftfreq(nfft, d=1.0 / fs_hz)
    idx = int(np.argmin(np.abs(f - float(f_off_hz))))
    g = np.abs(H[idx]) + 1e-30
    return 20.0 * np.log10(g)
