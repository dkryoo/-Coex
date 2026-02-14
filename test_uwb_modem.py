from UWB.uwb_modem import run_uwb_modem_ber_test


def main() -> None:
    print("=== UWB Modulation/Demodulation Test ===")

    no_ch = run_uwb_modem_ber_test(
        n_frames=100,
        bits_per_frame=128,
        distance_m=20.0,
        tx_eirp_dbw=-20.0,
        use_channel=False,
        seed=12345,
    )
    print(
        "[No channel] "
        f"BER={no_ch['ber']:.6e}, FER={no_ch['fer']:.3f}, frame_ok={no_ch['frame_ok_pct']:.1f}%"
    )

    ch_20 = run_uwb_modem_ber_test(
        n_frames=120,
        bits_per_frame=128,
        distance_m=20.0,
        tx_eirp_dbw=-20.0,
        nf_db=6.0,
        use_channel=True,
        seed=22345,
    )
    print(
        "[Rician+Thermal, d=20m] "
        f"BER={ch_20['ber']:.6e}, FER={ch_20['fer']:.3f}, frame_ok={ch_20['frame_ok_pct']:.1f}%"
    )

    ch_40 = run_uwb_modem_ber_test(
        n_frames=120,
        bits_per_frame=128,
        distance_m=40.0,
        tx_eirp_dbw=-20.0,
        nf_db=6.0,
        use_channel=True,
        seed=32345,
    )
    print(
        "[Rician+Thermal, d=40m] "
        f"BER={ch_40['ber']:.6e}, FER={ch_40['fer']:.3f}, frame_ok={ch_40['frame_ok_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()

