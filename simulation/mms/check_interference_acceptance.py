from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from simulation.mms.full_stack_mms_demo import FullStackConfig, run_full_stack_case


def main() -> None:
    cfg = FullStackConfig(
        distance_m=20.0,
        nb_channel=1,
        uwb_channel=5,
        wifi_channel=108,
        wifi_bw_mhz=160,
        wifi_standard="wifi7",
        nb_center_override_hz=6.4890e9,
        nb_eirp_dbw=-16.0,
        wifi_tx_power_dbw=-20.0,
        nf_db=6.0,
        seed=20260214,
    )

    cases = [
        ("off", run_full_stack_case(cfg, wifi_on=False, case_tag="chk_off", n_trials=80)),
        (
            "inband",
            run_full_stack_case(
                replace(cfg, wifi_tx_power_dbw=-20.0, rx_stop_db=25.0),
                wifi_on=True,
                wifi_channel_override=108,
                case_tag="chk_inband",
                n_trials=80,
            ),
        ),
        (
            "oob25",
            run_full_stack_case(
                replace(
                    cfg,
                    wifi_tx_power_dbw=-6.0,
                    wifi_aclr_db=30.0,
                    rx_stop_db=25.0,
                    enable_agc=True,
                    agc_stage="pre_selectivity",
                    agc_target_dbfs=-3.0,
                    agc_max_gain_db=35.0,
                    adc_clip_dbfs=-1.0,
                    quant_bits=6,
                ),
                wifi_on=True,
                wifi_channel_override=200,
                case_tag="chk_oob25",
                n_trials=80,
            ),
        ),
        (
            "oob60",
            run_full_stack_case(
                replace(cfg, wifi_tx_power_dbw=-6.0, wifi_aclr_db=30.0, rx_stop_db=60.0),
                wifi_on=True,
                wifi_channel_override=200,
                case_tag="chk_oob60",
                n_trials=80,
            ),
        ),
    ]

    for name, r in cases:
        rmse = r["ranging_rmse_m"]
        rmse_txt = f"{rmse:.3f}" if rmse == rmse else "nan"
        print(
            f"{name:7s} | BER={r['ber']:.4f} FER={r['fer']:.3f} "
            f"RMSE={rmse_txt} "
            f"Fail={r['ranging_fail_rate']:.3f}"
        )

    off = dict(cases)["off"]
    inb = dict(cases)["inband"]
    o25 = dict(cases)["oob25"]
    o60 = dict(cases)["oob60"]

    assert inb["ber"] >= 0.20 and inb["fer"] >= 0.70, "In-band collapse check failed"
    assert o60["fer"] <= max(0.20, off["fer"] + 0.10), "OOB stop60 should be near baseline"
    assert (
        (o25["fer"] >= off["fer"] + 0.10)
        or (o25["ber"] >= off["ber"] + 0.05)
        or (o25["ranging_fail_rate"] >= off["ranging_fail_rate"] + 0.10)
    ), "OOB stop25 + nonlinearity should degrade vs baseline"


if __name__ == "__main__":
    main()
