import numpy as np
import importlib.util
import sys
from pathlib import Path

from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
from Channel.Rician import apply_distance_rician_channel, apply_distance_rician_channel_with_thermal_noise
from Channel.Thermal_noise import add_thermal_noise_white, thermal_noise_power_w


def run_distance_ber_test(
    distances_m=range(1, 51),
    iters_per_distance: int = 100,
    bit_len: int = 122,
    chip_rate_hz: float = 2e6,
    osr: int = 8,
    fc_hz: float = 6.5e9,
    pathloss_exp: float = 2.0,
    k_factor_db: float = 8.0,
    delays_s=(0.0, 50e-9, 120e-9),
    powers_db=(0.0, -6.0, -10.0),
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    noise_ref_bw_hz: float = 2e6,
    tx_eirp_db: float | None = None,
    regulatory_profile: str = "unlicensed_6g_lpi_ap",
    rx_lead_zeros: int = 80,
    seed_base: int = 1234,
):
    tx = OQPSK_SF32_Tx(chip_rate_hz=chip_rate_hz, osr=osr)
    rx = OQPSK_SF32_Rx(chip_rate_hz=chip_rate_hz, osr=osr)

    results = []

    print("=== NarrowBand TX/RX Distance BER Test (Rician + Thermal Noise) ===")
    print(
        f"fs={tx.fs/1e6:.3f} Msps, fc={fc_hz/1e9:.3f} GHz, "
        f"iters={iters_per_distance}, bit_len={bit_len}, distances={min(distances_m)}..{max(distances_m)} m"
    )
    resolved_eirp_db = tx.resolve_eirp_db(tx_eirp_db=tx_eirp_db, regulatory_profile=regulatory_profile)
    print(
        f"profile={regulatory_profile}, EIRP={resolved_eirp_db:.2f} dBW, "
        f"noise_ref_bw={noise_ref_bw_hz/1e6:.3f} MHz"
    )

    for d in distances_m:
        total_err = 0
        total_bits = iters_per_distance * bit_len
        ok_count = 0

        for i in range(iters_per_distance):
            rng = np.random.default_rng(seed_base + 10000 * d + i)
            tx_bits = rng.integers(0, 2, bit_len, dtype=int)

            tx_wf, _, resolved_eirp_db = tx.build_tx_waveform(
                psdu_bits=tx_bits,
                tx_eirp_db=resolved_eirp_db,
                regulatory_profile=regulatory_profile,
            )
            rx_wf, info = apply_distance_rician_channel_with_thermal_noise(
                tx_wf=tx_wf,
                fs_hz=tx.fs,
                fc_hz=fc_hz,
                distance_m=float(d),
                tx_eirp_db=resolved_eirp_db,
                pathloss_exp=pathloss_exp,
                ref_distance_m=1.0,
                delays_s=delays_s,
                powers_db=powers_db,
                k_factor_db=k_factor_db,
                nf_db=nf_db,
                temperature_k=temperature_k,
                noise_ref_bw_hz=noise_ref_bw_hz,
                rx_ant_gain_db=0.0,
                rx_cable_loss_db=0.0,
                rx_lead_zeros=rx_lead_zeros,
                channel_seed=seed_base + 200000 + 10000 * d + i,
                noise_seed=seed_base + 300000 + 10000 * d + i,
            )

            try:
                rx_bits, _, _ = rx.decode(rx_wf, tx_fir=None, verbose=False)
                bit_err = int(np.sum(tx_bits != rx_bits[:bit_len]))
                total_err += bit_err
                ok_count += 1
            except Exception:
                total_err += bit_len

        ber = total_err / total_bits
        ok_pct = 100.0 * ok_count / iters_per_distance
        results.append((int(d), float(ber), float(ok_pct)))
        pl_db = info["pathloss_db"]
        pr_dbw = info["pr_dbw"]
        noise_dbw_ref = info["noise_dbw_ref_bw"]
        snr_db_ref = info["snr_db_ref_bw"]
        toa_samples = info["toa_samples"]
        print(
            f"d={d:2d} m | BER={ber:.6e} | decode_ok={ok_pct:5.1f}% "
            f"| PL={pl_db:6.2f} dB | Pr={pr_dbw:7.2f} dBW "
            f"| N={noise_dbw_ref:7.2f} dBW | SNR={snr_db_ref:6.2f} dB "
            f"| ToA={toa_samples} samp"
        )

    return results


def _load_wifi_tx_class():
    wifi_path = Path(__file__).resolve().parent / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("wifi_tx_module", wifi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Wi-Fi TX module from {wifi_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WiFiOFDMTx


def _resample_complex_linear(x: np.ndarray, fs_in_hz: float, fs_out_hz: float) -> np.ndarray:
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("sample rates must be > 0")
    if len(x) <= 1:
        return x.astype(np.complex128)

    t_in = np.arange(len(x), dtype=float) / fs_in_hz
    n_out = max(1, int(np.floor((len(x) - 1) * fs_out_hz / fs_in_hz)) + 1)
    t_out = np.arange(n_out, dtype=float) / fs_out_hz
    re = np.interp(t_out, t_in, np.real(x))
    im = np.interp(t_out, t_in, np.imag(x))
    return (re + 1j * im).astype(np.complex128)


def _scale_to_power_dbw(x: np.ndarray, p_dbw: float) -> np.ndarray:
    p_target_w = 10.0 ** (p_dbw / 10.0)
    p_now_w = float(np.mean(np.abs(x) ** 2)) + 1e-30
    return (x * np.sqrt(p_target_w / p_now_w)).astype(np.complex128)


def run_nb_wifi_interference_ber_test(
    distance_m: float = 10.0,
    center_freq_gap_mhz: float = 20.0,
    iters: int = 100,
    bit_len: int = 122,
    nb_fc_hz: float = 6.5e9,
    nb_tx_eirp_dbw: float = -30.0,
    wifi_standard: str = "wifi7",
    wifi_bw_mhz: int = 160,
    wifi_center_freq_hz: float = 6.52e9,
    wifi_target_tp_mbps: float = 600.0,
    wifi_tx_power_dbw: float = -20.0,
    wifi_rx_power_dbw: float | None = None,
    nf_db: float = 6.0,
    seed_base: int = 2026,
):
    """
    NB BER under Wi-Fi adjacent/interfering signal.
    Wi-Fi is frequency-shifted by center_freq_gap_mhz and summed at NB RX baseband.
    """
    tx_nb = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    rx_nb = OQPSK_SF32_Rx(chip_rate_hz=2e6, osr=8)
    WiFiOFDMTx = _load_wifi_tx_class()
    tx_wifi = WiFiOFDMTx(rng_seed=seed_base, center_freq_hz=wifi_center_freq_hz)

    total_err = 0
    ok_count = 0
    decoded_err = 0
    decoded_bits_total = 0
    frame_err_count = 0
    decode_fail_count = 0
    fs_nb = tx_nb.fs

    print("=== NB BER with Wi-Fi Interference ===")
    print(
        f"d={distance_m:.1f} m, gap={center_freq_gap_mhz:.3f} MHz, "
        f"NB EIRP={nb_tx_eirp_dbw:.2f} dBW, WiFi TX={wifi_tx_power_dbw:.2f} dBW"
    )

    for i in range(iters):
        rng = np.random.default_rng(seed_base + i)
        tx_bits = rng.integers(0, 2, bit_len, dtype=int)

        nb_wf, _, _ = tx_nb.build_tx_waveform(
            psdu_bits=tx_bits,
            tx_eirp_db=nb_tx_eirp_dbw,
            regulatory_profile="unlicensed_6g_lpi_ap",
        )
        rx_nb_wf, _, pl_nb_db, _, _ = apply_distance_rician_channel(
            wf=nb_wf,
            fs_hz=fs_nb,
            fc_hz=nb_fc_hz,
            distance_m=distance_m,
            pathloss_exp=2.0,
            ref_distance_m=1.0,
            delays_s=(0.0, 50e-9, 120e-9),
            powers_db=(0.0, -6.0, -10.0),
            k_factor_db=8.0,
            rx_ant_gain_db=0.0,
            rx_cable_loss_db=0.0,
            seed=seed_base + 100000 + i,
        )
        rx_nb_wf = np.concatenate([np.zeros(80, dtype=np.complex128), rx_nb_wf])

        wifi_wf, _ = tx_wifi.generate_for_target_rx_throughput(
            target_rx_throughput_mbps=wifi_target_tp_mbps,
            duration_s=max(0.001, len(rx_nb_wf) / fs_nb),
            channel_bw_mhz=wifi_bw_mhz,
            standard=wifi_standard,
            tx_power_dbw=wifi_tx_power_dbw,
        )
        wifi_rs = _resample_complex_linear(wifi_wf, fs_in_hz=wifi_bw_mhz * 1e6, fs_out_hz=fs_nb)
        if len(wifi_rs) < len(rx_nb_wf):
            wifi_rs = np.pad(wifi_rs, (0, len(rx_nb_wf) - len(wifi_rs)))
        wifi_rs = wifi_rs[: len(rx_nb_wf)]

        # Shift Wi-Fi interferer by center-frequency gap relative to NB baseband.
        t = np.arange(len(wifi_rs), dtype=float) / fs_nb
        f_off_hz = center_freq_gap_mhz * 1e6
        wifi_shift_tx = wifi_rs * np.exp(1j * 2.0 * np.pi * f_off_hz * t)

        rx_wifi_wf, _, pl_wifi_db, _, _ = apply_distance_rician_channel(
            wf=wifi_shift_tx,
            fs_hz=fs_nb,
            fc_hz=wifi_center_freq_hz,
            distance_m=distance_m,
            pathloss_exp=2.0,
            ref_distance_m=1.0,
            delays_s=(0.0, 30e-9, 80e-9),
            powers_db=(0.0, -6.0, -10.0),
            k_factor_db=6.0,
            rx_ant_gain_db=0.0,
            rx_cable_loss_db=0.0,
            seed=seed_base + 150000 + i,
        )
        rx_wifi_wf = np.concatenate([np.zeros(80, dtype=np.complex128), rx_wifi_wf])
        if len(rx_wifi_wf) < len(rx_nb_wf):
            rx_wifi_wf = np.pad(rx_wifi_wf, (0, len(rx_nb_wf) - len(rx_wifi_wf)))
        rx_wifi_wf = rx_wifi_wf[: len(rx_nb_wf)]

        # Optional override (legacy behavior) if user still wants fixed RX interferer power.
        if wifi_rx_power_dbw is not None:
            rx_wifi_wf = _scale_to_power_dbw(rx_wifi_wf, wifi_rx_power_dbw)

        # One receiver front-end: sum all incoming signals first, then add thermal noise once.
        rx_mix = rx_nb_wf + rx_wifi_wf
        rx_mix = add_thermal_noise_white(
            wf=rx_mix,
            fs_hz=fs_nb,
            nf_db=nf_db,
            temperature_k=290.0,
            seed=seed_base + 200000 + i,
        )

        try:
            rx_bits, _, _ = rx_nb.decode(rx_mix, tx_fir=None, verbose=False)
            bit_err = int(np.sum(tx_bits != rx_bits[:bit_len]))
            total_err += bit_err
            ok_count += 1
            decoded_err += bit_err
            decoded_bits_total += bit_len
            if bit_err > 0:
                frame_err_count += 1
        except Exception:
            total_err += bit_len
            frame_err_count += 1
            decode_fail_count += 1

    ber = total_err / (iters * bit_len)
    ok_pct = 100.0 * ok_count / iters
    frame_error_rate = frame_err_count / iters
    decode_fail_rate = decode_fail_count / iters
    ber_on_decoded = (decoded_err / decoded_bits_total) if decoded_bits_total > 0 else np.nan
    nb_pr = nb_tx_eirp_dbw - pl_nb_db
    wifi_pr = wifi_tx_power_dbw - pl_wifi_db
    nb_noise = 10.0 * np.log10(max(thermal_noise_power_w(fs_hz=2e6, nf_db=nf_db, temperature_k=290.0), 1e-30))
    nb_snr = nb_pr - nb_noise
    sir_db = nb_pr - wifi_pr
    print(
        f"BER_total={ber:.6e} | BER_on_decoded={ber_on_decoded:.6e} | "
        f"FER_packet={frame_error_rate:.3f} | decode_fail={decode_fail_rate:.3f} | "
        f"decode_ok={ok_pct:.1f}% | NB Pr={nb_pr:.2f} dBW | "
        f"WiFi Pr={wifi_pr:.2f} dBW | N={nb_noise:.2f} dBW | "
        f"SNR={nb_snr:.2f} dB | SIR={sir_db:.2f} dB"
    )
    return {
        "ber_total": ber,
        "ber_on_decoded": ber_on_decoded,
        "fer_packet": frame_error_rate,
        "decode_fail_rate": decode_fail_rate,
        "decode_ok_pct": ok_pct,
        "sir_db": sir_db,
    }


if __name__ == "__main__":
    run_nb_wifi_interference_ber_test()
