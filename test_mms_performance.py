from simulation.mms.performance import simulate_mms_performance


def main() -> None:
    print("=== MMS Performance (Initiation assumed complete) ===")
    print("Model: symbolic MMS phase + RMarker-wise timing over Rician+thermal-noise channel")

    rows = simulate_mms_performance(
        distances_m=(5.0, 10.0, 20.0, 40.0),
        n_trials=200,
        rif_payload_bits=256,
        fc_hz=6.5e9,
        tx_eirp_dbw=-20.0,
        nf_db=6.0,
        seed=20260214,
    )

    for r in rows:
        print(
            f"d={r['distance_m']:4.1f} m | "
            f"SNR(avg)={r['snr_db_avg']:6.2f} dB | "
            f"BER={r['ber']:.6e} | FER={r['fer']:.3f} | "
            f"Ranging RMSE={r['ranging_rmse_m']:.3f} m"
        )


if __name__ == "__main__":
    main()
