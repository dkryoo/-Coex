from __future__ import annotations

from simulation.mms.performance import simulate_mms_performance


def _run_case(name: str, **kwargs) -> None:
    print(f"\n=== {name} ===")
    rows = simulate_mms_performance(
        distances_m=(20.0,),
        n_trials=120,
        rif_payload_bits=256,
        tx_eirp_dbw=-20.0,
        nf_db=6.0,
        seed=20260219,
        debug_first_trial=True,
        save_psd=True,
        **kwargs,
    )
    r = rows[0]
    print(
        f"BER={r['ber']:.6e} | FER={r['fer']:.3f} | RMSE={r['ranging_rmse_m']:.3f} m | "
        f"SINR(avg)={r['snr_db_avg']:.2f} dB"
    )


def main() -> None:
    print("=== Wi-Fi -> UWB MMS Waveform Interference Demo ===")
    print("Setup: UWB distance=20 m, Wi-Fi interferer distance=2 m (near-far)")

    common_wifi = {
        "fc_hz": 6.6896e9,
        "bw_hz": 160e6,
        "tx_power_dbw": -20.0,
        "distance_m": 2.0,
        "standard": "wifi7",
        "target_rx_throughput_mbps": 800.0,
    }

    _run_case(
        "Case A: Wi-Fi OFF",
        uwb_fc_hz=6.4896e9,
        wifi_interference_on=False,
        wifi_params=common_wifi,
        uwb_rx_selectivity={"passband_hz": 60e6, "stopband_hz": 90e6, "stopband_atten_db": 25.0, "taps": 257},
        enable_adc_clipping=False,
    )

    _run_case(
        "Case B: Wi-Fi ON, selectivity 25 dB",
        uwb_fc_hz=6.4896e9,
        wifi_interference_on=True,
        wifi_params=common_wifi,
        uwb_rx_selectivity={"passband_hz": 60e6, "stopband_hz": 90e6, "stopband_atten_db": 25.0, "taps": 257},
        enable_adc_clipping=False,
    )

    _run_case(
        "Case C: Wi-Fi ON, selectivity 60 dB",
        uwb_fc_hz=6.4896e9,
        wifi_interference_on=True,
        wifi_params=common_wifi,
        uwb_rx_selectivity={"passband_hz": 60e6, "stopband_hz": 90e6, "stopband_atten_db": 60.0, "taps": 257},
        enable_adc_clipping=False,
    )

    _run_case(
        "Case D: Wi-Fi ON, selectivity 25 dB + ADC clipping",
        uwb_fc_hz=6.4896e9,
        wifi_interference_on=True,
        wifi_params=common_wifi,
        uwb_rx_selectivity={"passband_hz": 60e6, "stopband_hz": 90e6, "stopband_atten_db": 25.0, "taps": 257},
        enable_adc_clipping=True,
        clip_dbfs=-105.0,
    )


if __name__ == "__main__":
    main()
