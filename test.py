import numpy as np

from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
from Channel.Rician import apply_distance_rician_channel_with_thermal_noise


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
        print(
            f"d={d:2d} m | BER={ber:.6e} | decode_ok={ok_pct:5.1f}% "
            f"| PL={pl_db:6.2f} dB | Pr={pr_dbw:7.2f} dBW "
            f"| N={noise_dbw_ref:7.2f} dBW | SNR={snr_db_ref:6.2f} dB"
        )

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
        noise_ref_bw_hz=2e6,
        tx_eirp_db=None,
        regulatory_profile="unlicensed_6g_lpi_ap",
        rx_lead_zeros=80,
        seed_base=1234,
    )
