import numpy as np
from .Thermal_noise import add_thermal_noise_white, thermal_noise_power_w

C0 = 299_792_458.0  # speed of light (m/s)


def friis_pathloss_db(fc_hz: float, distance_m: float) -> float:
    """Free-space path loss: 20log10(4*pi*d/lambda)."""
    if fc_hz <= 0:
        raise ValueError("fc_hz must be > 0")
    distance_m = max(distance_m, 1e-9)
    wavelength_m = C0 / fc_hz
    return 20.0 * np.log10(4.0 * np.pi * distance_m / wavelength_m)


def log_distance_pathloss_db(
    fc_hz: float,
    distance_m: float,
    pathloss_exp: float = 2.0,
    ref_distance_m: float = 1.0,
) -> float:
    """
    Log-distance path loss:
      PL(d) = PL(d0) + 10*n*log10(d/d0)
    """
    if fc_hz <= 0:
        raise ValueError("fc_hz must be > 0")
    if pathloss_exp <= 0:
        raise ValueError("pathloss_exp must be > 0")

    ref_distance_m = max(ref_distance_m, 1e-9)
    distance_m = max(distance_m, ref_distance_m)

    pl_ref_db = friis_pathloss_db(fc_hz=fc_hz, distance_m=ref_distance_m)
    return pl_ref_db + 10.0 * pathloss_exp * np.log10(distance_m / ref_distance_m)


def rician_multipath_channel(
    fs_hz: float,
    delays_s: list[float] | tuple[float, ...],
    powers_db: list[float] | tuple[float, ...],
    k_factor_db: float = 8.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Build tapped-delay Rician channel impulse response h[n].

    - First tap: Rician (LOS + scatter)
    - Other taps: Rayleigh
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if len(delays_s) != len(powers_db):
        raise ValueError("delays_s and powers_db must have same length")
    if len(delays_s) == 0:
        raise ValueError("at least one tap is required")

    rng = np.random.default_rng(seed)

    delays_s = np.asarray(delays_s, dtype=float)
    if np.any(delays_s < 0):
        raise ValueError("delays_s must be non-negative")

    tap_powers_lin = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)
    tap_powers_lin = tap_powers_lin / (tap_powers_lin.sum() + 1e-30)

    sample_delays = np.round(delays_s * fs_hz).astype(int)
    h = np.zeros(int(sample_delays.max()) + 1, dtype=np.complex128)

    k_lin = 10.0 ** (k_factor_db / 10.0)

    for tap_idx, (d_samp, p_lin) in enumerate(zip(sample_delays, tap_powers_lin)):
        if tap_idx == 0:
            phase = rng.uniform(0.0, 2.0 * np.pi)
            los = np.sqrt(k_lin / (k_lin + 1.0)) * np.sqrt(p_lin) * np.exp(1j * phase)
            scatter = np.sqrt(1.0 / (k_lin + 1.0)) * np.sqrt(p_lin / 2.0) * (
                rng.standard_normal() + 1j * rng.standard_normal()
            )
            h[d_samp] += los + scatter
        else:
            h[d_samp] += np.sqrt(p_lin / 2.0) * (rng.standard_normal() + 1j * rng.standard_normal())

    return h


def apply_distance_rician_channel(
    wf: np.ndarray,
    fs_hz: float,
    fc_hz: float,
    distance_m: float,
    pathloss_exp: float = 2.0,
    ref_distance_m: float = 1.0,
    delays_s: tuple[float, ...] = (0.0, 50e-9, 120e-9),
    powers_db: tuple[float, ...] = (0.0, -6.0, -10.0),
    k_factor_db: float = 8.0,
    rx_ant_gain_db: float = 0.0,
    rx_cable_loss_db: float = 0.0,
    include_toa: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Apply distance-based large-scale loss + small-scale Rician multipath.

    Returns:
      rx_wf, h, pathloss_db, toa_s, toa_samples
    """
    pl_db = log_distance_pathloss_db(
        fc_hz=fc_hz,
        distance_m=distance_m,
        pathloss_exp=pathloss_exp,
        ref_distance_m=ref_distance_m,
    )

    toa_s = max(distance_m, 0.0) / C0 if include_toa else 0.0
    toa_samples = int(np.round(toa_s * fs_hz))
    delays_eff_s = tuple(float(toa_s + d) for d in delays_s)

    h = rician_multipath_channel(
        fs_hz=fs_hz,
        delays_s=delays_eff_s,
        powers_db=powers_db,
        k_factor_db=k_factor_db,
        seed=seed,
    )

    # Pr(dBm) = Pt(dBm) + Gr(dB) - PL(dB) - Lrx(dB)
    power_scale_db = rx_ant_gain_db - pl_db - rx_cable_loss_db
    amp_scale = 10.0 ** (power_scale_db / 20.0)

    rx_wf = np.convolve(wf, h, mode="full") * amp_scale
    return rx_wf, h, pl_db, toa_s, toa_samples


def apply_distance_rician_channel_with_thermal_noise(
    tx_wf: np.ndarray,
    fs_hz: float,
    fc_hz: float,
    distance_m: float,
    tx_eirp_db: float,
    pathloss_exp: float = 2.0,
    ref_distance_m: float = 1.0,
    delays_s: tuple[float, ...] = (0.0, 50e-9, 120e-9),
    powers_db: tuple[float, ...] = (0.0, -6.0, -10.0),
    k_factor_db: float = 8.0,
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    noise_ref_bw_hz: float | None = None,
    rx_ant_gain_db: float = 0.0,
    rx_cable_loss_db: float = 0.0,
    include_toa: bool = True,
    rx_lead_zeros: int = 80,
    channel_seed: int | None = None,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Channel pipeline for simulation:
      TX waveform -> distance Rician channel -> thermal noise.

    Returns:
      rx_wf: received waveform after channel+noise
      info: link-budget dictionary in dB scale (dBW reference)
    """
    ch_wf, h, pl_db, toa_s, toa_samples = apply_distance_rician_channel(
        wf=tx_wf,
        fs_hz=fs_hz,
        fc_hz=fc_hz,
        distance_m=distance_m,
        pathloss_exp=pathloss_exp,
        ref_distance_m=ref_distance_m,
        delays_s=delays_s,
        powers_db=powers_db,
        k_factor_db=k_factor_db,
        rx_ant_gain_db=rx_ant_gain_db,
        rx_cable_loss_db=rx_cable_loss_db,
        include_toa=include_toa,
        seed=channel_seed,
    )

    rx_wf = np.concatenate([np.zeros(rx_lead_zeros, dtype=np.complex128), ch_wf])
    rx_wf = add_thermal_noise_white(
        wf=rx_wf,
        fs_hz=fs_hz,
        nf_db=nf_db,
        temperature_k=temperature_k,
        seed=noise_seed,
    )

    # Injected noise is generated at sample-rate bandwidth (fs_hz).
    noise_power_w_sample = thermal_noise_power_w(fs_hz=fs_hz, nf_db=nf_db, temperature_k=temperature_k)
    noise_dbw_sample = 10.0 * np.log10(max(noise_power_w_sample, 1e-30))

    # Optional reporting BW (e.g., channel bandwidth) for link-budget readability.
    if noise_ref_bw_hz is None:
        noise_ref_bw_hz = fs_hz
    noise_power_w_ref = thermal_noise_power_w(
        fs_hz=float(noise_ref_bw_hz), nf_db=nf_db, temperature_k=temperature_k
    )
    noise_dbw_ref = 10.0 * np.log10(max(noise_power_w_ref, 1e-30))

    pr_dbw = tx_eirp_db + rx_ant_gain_db - pl_db - rx_cable_loss_db
    snr_db_sample = pr_dbw - noise_dbw_sample
    snr_db_ref = pr_dbw - noise_dbw_ref

    info = {
        "pathloss_db": float(pl_db),
        "pr_dbw": float(pr_dbw),
        "eirp_dbw": float(tx_eirp_db),
        "noise_dbw_sample_rate": float(noise_dbw_sample),
        "noise_dbw_ref_bw": float(noise_dbw_ref),
        "noise_ref_bw_hz": float(noise_ref_bw_hz),
        "snr_db_sample_rate": float(snr_db_sample),
        "snr_db_ref_bw": float(snr_db_ref),
        "toa_s": float(toa_s),
        "toa_samples": int(toa_samples),
        "h": h,
    }
    return rx_wf, info
