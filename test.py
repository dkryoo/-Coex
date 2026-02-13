import numpy as np

from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
from Channel.Rician import apply_distance_rician_channel
from Channel.Thermal_noise import add_thermal_noise_white


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def scale_waveform_to_psd(
    wf: np.ndarray,
    psd_dbm_per_mhz: float,
    bw_hz: float,
    ant_gain_tx_db: float = 0.0,
    cable_loss_tx_db: float = 0.0,
) -> np.ndarray:
    """
    Scale mean(|wf|^2) to conducted Tx power from PSD target.

    EIRP(dBm) = PSD(dBm/MHz) + 10log10(BW/1e6)
    P_cond(dBm) = EIRP - Gt + Lcable_tx
    """
    if bw_hz <= 0:
        raise ValueError("bw_hz must be > 0")

    eirp_dbm = psd_dbm_per_mhz + 10.0 * np.log10(bw_hz / 1e6)
    p_cond_dbm = eirp_dbm - ant_gain_tx_db + cable_loss_tx_db
    p_cond_w = dbm_to_w(p_cond_dbm)

    p_now = float(np.mean(np.abs(wf) ** 2)) + 1e-30
    return wf * np.sqrt(p_cond_w / p_now)


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
    psd_dbm_per_mhz: float = -41.3,
    bw_hz: float = 2e6,
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

    for d in distances_m:
        total_err = 0
        total_bits = iters_per_distance * bit_len
        ok_count = 0

        for i in range(iters_per_distance):
            rng = np.random.default_rng(seed_base + 10000 * d + i)
            tx_bits = rng.integers(0, 2, bit_len, dtype=int)

            frame_bits, _ = tx.build_frame_bits(tx_bits)
            tx_wf = tx.bits_to_baseband(frame_bits)
            tx_wf = scale_waveform_to_psd(
                tx_wf,
                psd_dbm_per_mhz=psd_dbm_per_mhz,
                bw_hz=bw_hz,
            )

            ch_wf, _, _ = apply_distance_rician_channel(
                wf=tx_wf,
                fs_hz=tx.fs,
                fc_hz=fc_hz,
                distance_m=float(d),
                pathloss_exp=pathloss_exp,
                delays_s=delays_s,
                powers_db=powers_db,
                k_factor_db=k_factor_db,
                seed=seed_base + 200000 + 10000 * d + i,
            )

            rx_wf = np.concatenate([np.zeros(rx_lead_zeros, dtype=np.complex128), ch_wf])
            rx_wf = add_thermal_noise_white(
                wf=rx_wf,
                fs_hz=tx.fs,
                nf_db=nf_db,
                temperature_k=temperature_k,
                seed=seed_base + 300000 + 10000 * d + i,
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
        print(f"d={d:2d} m | BER={ber:.6e} | decode_ok={ok_pct:5.1f}%")

    return results


if __name__ == "__main__":
    run_distance_ber_test(
        distances_m=range(1, 51),
        iters_per_distance=100,
        bit_len=122,
        chip_rate_hz=2e6,
        osr=8,
        fc_hz=6.5e9,
        pathloss_exp=2.0,
        k_factor_db=8.0,
        delays_s=(0.0, 50e-9, 120e-9),
        powers_db=(0.0, -6.0, -10.0),
        nf_db=6.0,
        psd_dbm_per_mhz=-41.3,
        bw_hz=2e6,
        rx_lead_zeros=80,
        seed_base=1234,
    )
