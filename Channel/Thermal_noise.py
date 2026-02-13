import numpy as np

K_BOLTZ = 1.380649e-23  # J/K


def thermal_noise_power_w(fs_hz: float, nf_db: float = 6.0, temperature_k: float = 290.0) -> float:
    """
    Complex baseband thermal noise power E[|n|^2] in watts.

    sigma^2 = k * T * F * fs
      - k: Boltzmann constant
      - T: temperature (K)
      - F: noise factor = 10^(NF/10)
      - fs: complex sample rate (Hz)
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be > 0")

    noise_factor = 10.0 ** (nf_db / 10.0)
    return K_BOLTZ * temperature_k * noise_factor * fs_hz


def add_thermal_noise_white(
    wf: np.ndarray,
    fs_hz: float,
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add complex white thermal noise to waveform.

    Returns:
      noisy waveform with AWGN whose average power is k*T*F*fs.
    """
    sigma2 = thermal_noise_power_w(fs_hz=fs_hz, nf_db=nf_db, temperature_k=temperature_k)

    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(len(wf)) + 1j * rng.standard_normal(len(wf))) * np.sqrt(sigma2 / 2.0)
    return wf + noise
