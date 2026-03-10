from __future__ import annotations

"""
Distance sweep for MMS (NB initiation + one MMS UWB ranging) performance, Wi-Fi OFF.

Example (quick):
python simulation/mms/run_mms_ranging_distance_sweep.py \
  --distances "5,10,20,30,40" --trials 50 --out-dir simulation/mms/results/mms_ranging_distance

Example (paper):
python simulation/mms/run_mms_ranging_distance_sweep.py \
  --distances "5,10,15,20,25,30,35,40,50" --trials 200 \
  --max-attempts 1 --until-success 0 \
  --out-dir simulation/mms/results/mms_ranging_distance
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from simulation.mms.full_stack_mms_demo import FullStackConfig, uwb_center_freq_hz
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.performance import simulate_mms_performance
    from simulation.mms.standard_params import get_default_params
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.full_stack_mms_demo import FullStackConfig, uwb_center_freq_hz
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.performance import simulate_mms_performance
    from simulation.mms.standard_params import get_default_params


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_db_tuple(s: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 1:
        raise ValueError("channel powers list must have at least one value")
    return tuple(vals)


def _parse_ns_tuple(s: str) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 1:
        raise ValueError("channel delays list must have at least one value")
    return tuple(v * 1e-9 for v in vals)


def _summary_stats(vals: np.ndarray) -> tuple[float, float, float]:
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95))


def _compute_fixed_toa_calibration_samples(cfg: FullStackConfig, distance_m: float) -> float:
    """
    Compute one distance-independent ToA calibration and reuse for entire sweep.
    This avoids per-trial/per-distance calibration jitter masking RMSE trends.
    """
    res = simulate_mms_performance(
        distances_m=(float(distance_m),),
        n_trials=16,
        fc_hz=uwb_center_freq_hz(int(cfg.uwb_channel)),
        uwb_fs_hz=499.2e6,
        uwb_fc_hz=uwb_center_freq_hz(int(cfg.uwb_channel)),
        tx_eirp_dbw=float(cfg.nb_eirp_dbw),
        nf_db=float(cfg.nf_db),
        temperature_k=float(cfg.temperature_k),
        seed=int(cfg.seed + 505050),
        detector_mode="first_path",
        toa_refine_method=str(cfg.toa_refine_method),
        corr_upsample=int(cfg.corr_upsample),
        corr_win=int(cfg.corr_win),
        first_path=bool(cfg.first_path),
        first_path_thr_db=float(cfg.first_path_thr_db),
        first_path_peak_frac=cfg.first_path_peak_frac,
        fp_use_adaptive_thr=bool(cfg.fp_use_adaptive_thr),
        fp_snr_switch_db=float(cfg.fp_snr_switch_db),
        fp_thr_min_floor_mult=float(cfg.fp_thr_min_floor_mult),
        first_path_search_back=int(cfg.first_path_search_back),
        # Sanity calibration: fixed algorithmic delay only.
        baseline_sanity_mode=True,
        auto_calibrate=True,
        enable_crc=True,
        debug_first_trial=False,
        save_psd=False,
    )[0]
    return float(res.get("toa_calibration_samples", 0.0))


def _to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    s = x.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"])


def _plot(agg_df: pd.DataFrame, trial_df: pd.DataFrame, out_dir: Path) -> list[str]:
    files: list[str] = []
    if agg_df.empty:
        return files

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Success probabilities
    fig, ax = plt.subplots(figsize=(9, 5))
    x = agg_df["distance_m"].to_numpy(dtype=float)
    ax.plot(x, agg_df["P_init"], marker="o", linewidth=2.0, label="P_init")
    ax.plot(x, agg_df["P_range_given_init"], marker="s", linewidth=2.0, label="P_range|init")
    ax.plot(x, agg_df["P_overall"], marker="^", linewidth=2.4, label="P_overall")
    ax.set_title("MMS Session Success Probability vs Distance (Wi-Fi OFF)")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fn = "plot_success_prob_vs_distance.png"
    fig.savefig(out_dir / fn, dpi=170)
    plt.close(fig)
    files.append(fn)

    # 2) Latency
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, agg_df["lat_succ_median_ms"], marker="o", linestyle="--", linewidth=2.0, label="success median")
    ax.plot(x, agg_df["lat_succ_p95_ms"], marker="^", linestyle=":", linewidth=2.0, label="success p95")
    ax.plot(x, agg_df["lat_all_median_ms"], marker="s", linestyle="-", linewidth=2.0, label="all median")
    ax.plot(x, agg_df["lat_all_p95_ms"], marker="x", linestyle="-.", linewidth=2.0, label="all p95")
    for _, r in agg_df.iterrows():
        if int(r["success_count"]) == 0:
            ax.annotate("0 success", (float(r["distance_m"]), float(r["lat_all_median_ms"])), fontsize=8)
    ax.set_title("MMS Session Latency vs Distance (Wi-Fi OFF)")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Latency [ms]")
    ax.legend(ncol=2)
    fig.tight_layout()
    fn = "plot_latency_vs_distance.png"
    fig.savefig(out_dir / fn, dpi=170)
    plt.close(fig)
    files.append(fn)

    # 3) Range error
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, agg_df["range_bias_m"], marker="o", linewidth=2.0, label="Bias [m]")
    ax.plot(x, agg_df["range_rmse_m"], marker="s", linewidth=2.2, label="RMSE [m]")
    ax.plot(x, agg_df["range_mae_m"], marker="^", linewidth=2.0, label="MAE [m]")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_title("MMS Ranging Error vs Distance (success-only)")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Error [m]")
    ax.legend()
    fig.tight_layout()
    fn = "plot_range_error_vs_distance.png"
    fig.savefig(out_dir / fn, dpi=170)
    plt.close(fig)
    files.append(fn)

    # 4) Error distribution (boxplot)
    ok = trial_df[trial_df["success"] == True].copy()
    ok = ok[np.isfinite(pd.to_numeric(ok["range_error_m"], errors="coerce"))]
    if not ok.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        dvals = sorted(ok["distance_m"].unique())
        data = [
            pd.to_numeric(ok[ok["distance_m"] == d]["range_error_m"], errors="coerce").dropna().to_numpy(dtype=float)
            for d in dvals
        ]
        ax.boxplot(data, tick_labels=[f"{d:g}" for d in dvals], showfliers=False)
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title("Ranging Error Distribution by Distance (success-only)")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Estimated - True [m]")
        fig.tight_layout()
        fn = "plot_range_error_box_by_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 5) First-path / peak index diagnostics
    if {"distance_m", "first_path_index_mean", "peak_index_mean"}.issubset(set(agg_df.columns)):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, agg_df["first_path_index_mean"], marker="o", linewidth=2.0, label="first_path_index_mean")
        ax.plot(x, agg_df["peak_index_mean"], marker="s", linewidth=2.0, label="peak_index_mean")
        ax.set_title("First-path vs Peak Index by Distance")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Index [samples]")
        ax.legend()
        fig.tight_layout()
        fn = "plot_fp_index_vs_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 6) Bias equivalent ticks
    if {"distance_m", "range_bias_equiv_ticks"}.issubset(set(agg_df.columns)):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, agg_df["range_bias_equiv_ticks"], marker="o", linewidth=2.2, label="bias equivalent ticks")
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title("Range Bias Converted to ToF Ticks")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Ticks [sample periods]")
        ax.legend()
        fig.tight_layout()
        fn = "plot_bias_ticks_vs_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 7) Noise-window margin diagnostic: noise_win_end - first_path_index
    if {"distance_m", "noise_win_end", "first_path_index"}.issubset(set(trial_df.columns)):
        td = trial_df.copy()
        td["noise_margin"] = pd.to_numeric(td["noise_win_end"], errors="coerce") - pd.to_numeric(td["first_path_index"], errors="coerce")
        td = td[np.isfinite(td["noise_margin"])]
        if not td.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            dvals = sorted(td["distance_m"].unique())
            data = [td[td["distance_m"] == d]["noise_margin"].to_numpy(dtype=float) for d in dvals]
            ax.boxplot(data, tick_labels=[f"{d:g}" for d in dvals], showfliers=False)
            ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
            ax.set_title("Noise Window End - First Path Index (positive means overlap risk)")
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("noise_win_end - k_fp [samples]")
            fig.tight_layout()
            fn = "plot_noise_window_margin_vs_distance.png"
            fig.savefig(out_dir / fn, dpi=170)
            plt.close(fig)
            files.append(fn)

    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="MMS ranging distance sweep (initiation + one ranging), Wi-Fi OFF")
    ap.add_argument("--distances", type=str, default="5,10,20,30,40")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260305)
    ap.add_argument("--uwb-channel", type=int, default=5)
    ap.add_argument("--nb-eirp-dbw", type=float, default=-16.0)
    ap.add_argument("--nf-db", type=float, default=6.0)
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--until-success", type=int, default=0, help="0/1")
    ap.add_argument("--max-trial-ms", type=float, default=200.0)
    ap.add_argument("--uwb-shots-per-session", type=int, default=1)
    ap.add_argument("--require-k-successes", type=int, default=1)
    ap.add_argument("--aggregation", choices=["median", "mean", "min"], default="median")
    ap.add_argument("--uwb-shot-gap-ms", type=float, default=0.5)
    ap.add_argument("--enable-init-scan-model", type=int, default=0)
    ap.add_argument("--scan-interval-ms", type=float, default=100.0)
    ap.add_argument("--scan-window-ms", type=float, default=20.0)
    ap.add_argument("--adv-interval-ms", type=float, default=100.0)
    ap.add_argument("--adv-tx-duration-ms", type=float, default=2.0)
    ap.add_argument("--random-start-phase", type=int, default=1)
    ap.add_argument("--enable-report-phase-model", type=int, default=0)
    ap.add_argument("--initiator-report-request", type=int, default=0)
    ap.add_argument("--responder-report-request", type=int, default=0)
    ap.add_argument("--mms1stReportNSlots", type=int, default=1)
    ap.add_argument("--mms2ndReportNSlots", type=int, default=0)
    ap.add_argument("--assume-oob-report-on-missing", type=int, default=1)
    ap.add_argument("--dump-timeline-k", type=int, default=1)
    ap.add_argument("--first-path-thr-db", type=float, default=13.0)
    # Tuned default for distance-sweep stability (especially 3~10 m near range).
    ap.add_argument("--first-path-peak-frac", type=float, default=0.18)
    ap.add_argument("--corr-upsample", type=int, default=8)
    ap.add_argument("--corr-win", type=int, default=64)
    ap.add_argument("--fp-snr-switch-db", type=float, default=12.0)
    ap.add_argument("--fp-thr-min-floor-mult", type=float, default=2.0)
    ap.add_argument("--first-path-search-back", type=int, default=8)
    ap.add_argument("--toa-detector", choices=["auto", "first_path", "strongest_peak"], default="auto")
    ap.add_argument("--range-bias-correction-m", type=float, default=0.0)
    ap.add_argument("--toa-calibration-override-samples", type=float, default=float("nan"))
    # For this distance-study runner, runtime-channel-matched calibration is a better default.
    ap.add_argument("--toa-calibration-use-runtime-channel", type=int, default=1)
    ap.add_argument("--freeze-toa-calibration", type=int, default=0)
    ap.add_argument("--toa-calibration-distance-m", type=float, default=float("nan"))
    # Default to LOS-like channel for distance-only ranging trend studies.
    ap.add_argument("--channel-delays-ns", type=str, default="0")
    ap.add_argument("--channel-powers-db", type=str, default="0")
    ap.add_argument("--channel-k-factor-db", type=float, default=20.0)
    ap.add_argument("--channel-delay-jitter-ns", type=float, default=0.0)
    ap.add_argument("--channel-power-jitter-db", type=float, default=0.0)
    ap.add_argument("--channel-profile", choices=["custom", "los", "mild_mp", "harsh_mp"], default="los")
    ap.add_argument("--realism-preset", choices=["none", "balanced", "realistic"], default="none")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="simulation/mms/results/mms_ranging_distance")
    args = ap.parse_args()

    distances = _parse_float_list(args.distances)
    if args.channel_profile == "los":
        ch_delays_s = (0.0,)
        ch_powers_db = (0.0,)
        ch_k_factor_db = 20.0
    elif args.channel_profile == "mild_mp":
        ch_delays_s = (0.0, 4e-9, 8e-9)
        ch_powers_db = (0.0, -10.0, -16.0)
        ch_k_factor_db = 12.0
    elif args.channel_profile == "harsh_mp":
        ch_delays_s = (0.0, 6e-9, 12e-9)
        ch_powers_db = (0.0, -6.0, -10.0)
        ch_k_factor_db = 8.0
    else:
        ch_delays_s = _parse_ns_tuple(args.channel_delays_ns)
        ch_powers_db = _parse_db_tuple(args.channel_powers_db)
        ch_k_factor_db = float(args.channel_k_factor_db)

    delay_jitter_ns = float(max(0.0, args.channel_delay_jitter_ns))
    power_jitter_db = float(max(0.0, args.channel_power_jitter_db))
    if str(args.realism_preset) == "balanced":
        # Middle-ground setting: avoid idealized LOS and avoid over-harsh fixed multipath.
        if args.channel_profile == "los":
            ch_delays_s = (0.0, 3e-9, 7e-9)
            ch_powers_db = (0.0, -12.0, -18.0)
            ch_k_factor_db = 12.0
        if delay_jitter_ns <= 0.0:
            delay_jitter_ns = 0.8
        if power_jitter_db <= 0.0:
            power_jitter_db = 1.5
    elif str(args.realism_preset) == "realistic":
        if args.channel_profile == "los":
            ch_delays_s = (0.0, 4e-9, 9e-9)
            ch_powers_db = (0.0, -11.0, -17.0)
            ch_k_factor_db = 10.0
        if delay_jitter_ns <= 0.0:
            delay_jitter_ns = 1.2
        if power_jitter_db <= 0.0:
            power_jitter_db = 2.0

    if len(ch_delays_s) != len(ch_powers_db):
        raise ValueError("channel-delays-ns and channel-powers-db must have same length")
    out_root = Path(args.out_dir)
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir = out_dir / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    std = get_default_params("802154ab")
    ssbd_cfg = SsbdConfig(
        phy_cca_duration_ms=float(std.nb_phy_cca_duration_ms),
        phy_cca_ed_threshold_dbm=float(std.nb_phy_cca_ed_threshold_dbm),
        cca_mode=1,
        mac_ssbd_unit_backoff_ms=float(std.nb_ssbd_unit_backoff_ms),
        mac_ssbd_min_bf=int(std.nb_ssbd_min_bf),
        mac_ssbd_max_bf=int(std.nb_ssbd_max_bf),
        mac_ssbd_max_backoffs=int(std.nb_ssbd_max_backoffs),
        mac_ssbd_tx_on_end=bool(std.nb_ssbd_tx_on_end),
        mac_ssbd_persistence=bool(std.nb_ssbd_persistence),
    )

    nb_cfg = NbChannelSwitchConfig(
        enable_switching=False,
        allow_list=(0,),
        mms_prng_seed=0,
        channel_switching_field=0,
        nb_channel_spacing_mhz=2.0,
    )

    cfg_base = FullStackConfig(
        distance_m=float(distances[0]),
        nb_channel=1,
        uwb_channel=int(args.uwb_channel),
        wifi_channel=108,
        wifi_bw_mhz=160,
        wifi_standard="wifi7",
        nb_center_override_hz=6.489e9,
        nb_eirp_dbw=float(args.nb_eirp_dbw),
        wifi_tx_power_dbw=-10.0,
        nf_db=float(args.nf_db),
        seed=int(args.seed),
        first_path_thr_db=float(args.first_path_thr_db),
        first_path_peak_frac=float(args.first_path_peak_frac),
        corr_upsample=int(args.corr_upsample),
        corr_win=int(args.corr_win),
        fp_snr_switch_db=float(args.fp_snr_switch_db),
        fp_thr_min_floor_mult=float(args.fp_thr_min_floor_mult),
        first_path_search_back=int(args.first_path_search_back),
        range_bias_correction_m=float(args.range_bias_correction_m),
        toa_calibration_override=(
            None if not np.isfinite(float(args.toa_calibration_override_samples))
            else float(args.toa_calibration_override_samples)
        ),
        toa_calibration_use_runtime_channel=bool(int(args.toa_calibration_use_runtime_channel)),
        channel_delays_s=ch_delays_s,
        channel_powers_db=ch_powers_db,
        channel_k_factor_db=float(ch_k_factor_db),
        channel_delay_jitter_std_ns=float(delay_jitter_ns),
        channel_power_jitter_std_db=float(power_jitter_db),
    )

    # Detector policy:
    # - LOS/single-path: strongest-peak is more stable than edge-threshold first-path.
    # - Multipath: first-path helps reduce late-peak bias.
    detector_sel = str(args.toa_detector)
    if detector_sel == "auto":
        if len(ch_delays_s) == 1 and abs(float(ch_delays_s[0])) < 1e-15:
            use_first_path = False
        else:
            # In weak-multipath / high-K conditions, strongest peak is often
            # more stable than aggressive leading-edge detection.
            delayed = list(ch_powers_db[1:]) if len(ch_powers_db) > 1 else []
            max_delayed_db = max(delayed) if delayed else -999.0
            weak_mp = (float(ch_k_factor_db) >= 10.0) and (float(max_delayed_db) <= -10.0)
            use_first_path = not weak_mp
    elif detector_sel == "first_path":
        use_first_path = True
    else:
        use_first_path = False
    cfg_base = replace(cfg_base, first_path=bool(use_first_path))
    print(f"[toa-detector] mode={detector_sel} -> first_path={int(use_first_path)}")

    # Use one fixed ToA calibration across all distances unless explicitly overridden.
    if np.isfinite(float(args.toa_calibration_override_samples)):
        cfg_base = replace(cfg_base, toa_calibration_override=float(args.toa_calibration_override_samples))
    elif int(args.freeze_toa_calibration) == 1:
        cal_d = float(args.toa_calibration_distance_m) if np.isfinite(float(args.toa_calibration_distance_m)) else float(distances[0])
        cal_samp = _compute_fixed_toa_calibration_samples(cfg_base, distance_m=cal_d)
        print(f"[toa-cal] fixed calibration={cal_samp:.6f} samples (distance={cal_d:.3f} m)")
        cfg_base = replace(
            cfg_base,
            toa_calibration_override=float(cal_samp),
            toa_calibration_use_runtime_channel=False,
        )

    rows: list[dict] = []
    total = len(distances) * int(args.trials)
    done = 0

    for d in distances:
        succ = 0
        print(f"[distance] d={d:.1f}m trials={int(args.trials)}")
        for t in range(int(args.trials)):
            seed = int(args.seed + int(d * 1000) + t)
            row = _run_trial(
                cfg_base=cfg_base,
                wifi_mode="off",
                wifi_density=0.0,
                distance_m=float(d),
                uwb_channel=int(args.uwb_channel),
                wifi_offset_mhz=None,
                trial_idx=t,
                seed=seed,
                max_attempts=int(args.max_attempts),
                until_success=bool(int(args.until_success)),
                max_trial_ms=float(args.max_trial_ms),
                uwb_shots_per_session=int(args.uwb_shots_per_session),
                require_k_successes=int(args.require_k_successes),
                aggregation=str(args.aggregation),
                uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
                wifi_model="occupancy",
                spatial_model=None,
                nb_lbt_slot_ms=float(std.nb_lbt_slot_ms),
                nb_lbt_cca_slots=int(std.nb_lbt_cca_slots),
                ssbd_cfg=ssbd_cfg,
                ssbd_debug=False,
                print_ssbd_trace=False,
                nb_switch_cfg=nb_cfg,
                enable_init_scan_model=bool(int(args.enable_init_scan_model)),
                scan_interval_ms=float(args.scan_interval_ms),
                scan_window_ms=float(args.scan_window_ms),
                adv_interval_ms=float(args.adv_interval_ms),
                adv_tx_duration_ms=float(args.adv_tx_duration_ms),
                random_start_phase=bool(int(args.random_start_phase)),
                enable_report_phase_model=bool(int(args.enable_report_phase_model)),
                initiator_report_request=bool(int(args.initiator_report_request)),
                responder_report_request=bool(int(args.responder_report_request)),
                mms1st_report_nslots=int(args.mms1stReportNSlots),
                mms2nd_report_nslots=int(args.mms2ndReportNSlots),
                assume_oob_report_on_missing=bool(int(args.assume_oob_report_on_missing)),
            )
            row["init_success"] = bool(row.get("control_ok_last", False))
            rr = float(row.get("range_result_m", float("nan")))
            row["range_error_m"] = float(rr - d) if np.isfinite(rr) else float("nan")
            if int(args.dump_timeline_k) > 0 and t < int(args.dump_timeline_k):
                import json

                ev = row.get("event_trace_json", "[]")
                try:
                    parsed = json.loads(ev if isinstance(ev, str) else "[]")
                except Exception:
                    parsed = []
                (timeline_dir / f"trial_d{int(d)}_t{t}.json").write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=True)
                )
            rows.append(row)
            succ += int(bool(row.get("success", False)))
            done += 1
            if int(args.progress_every) > 0 and ((t + 1) % int(args.progress_every) == 0 or (t + 1) == int(args.trials)):
                print(f"  - progress {t+1}/{int(args.trials)} succ={succ} ({done}/{total})")

    df = pd.DataFrame(rows)
    trial_csv = out_dir / "trial_results.csv"
    df.to_csv(trial_csv, index=False)

    agg_rows: list[dict] = []
    for d, g in df.groupby("distance_m", dropna=False):
        n = int(len(g))
        if "init_success" in g.columns:
            init_bool = _to_bool_series(g["init_success"])
            p_init = float(np.mean(init_bool.astype(float)))
        else:
            p_init = float("nan")

        init_ok = g[_to_bool_series(g["init_success"])] if "init_success" in g.columns else g
        p_range_given_init = float(np.mean(init_ok["success"].astype(float))) if len(init_ok) else float("nan")
        p_overall = float(np.mean(g["success"].astype(float)))

        succ = g[g["success"] == True]
        lat_succ = pd.to_numeric(succ["latency_to_success_ms"], errors="coerce").dropna().to_numpy(dtype=float)
        lat_all = pd.to_numeric(g["time_spent_ms"], errors="coerce").dropna().to_numpy(dtype=float)
        _, lat_s_med, lat_s_p95 = _summary_stats(lat_succ)
        _, lat_a_med, lat_a_p95 = _summary_stats(lat_all)
        lbt = pd.to_numeric(g.get("lat_lbt_cca_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        bo = pd.to_numeric(g.get("lat_backoff_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        uwb = pd.to_numeric(g.get("lat_uwb_exchange_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        advscan = pd.to_numeric(g.get("lat_adv_scan_wait_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        initw = pd.to_numeric(g.get("init_scan_wait_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        nbctrl = pd.to_numeric(g.get("nb_ctrl_txrx_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        _, lbt_med, lbt_p95 = _summary_stats(lbt)
        _, bo_med, bo_p95 = _summary_stats(bo)
        _, uwb_med, uwb_p95 = _summary_stats(uwb)
        _, advscan_med, advscan_p95 = _summary_stats(advscan)
        _, initw_med, initw_p95 = _summary_stats(initw)
        _, nbctrl_med, nbctrl_p95 = _summary_stats(nbctrl)
        rpt = pd.to_numeric(g.get("report_phase_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        rpt_nb = pd.to_numeric(g.get("nb_report_txrx_ms"), errors="coerce").dropna().to_numpy(dtype=float)
        _, rpt_med, rpt_p95 = _summary_stats(rpt)
        _, rpt_nb_med, rpt_nb_p95 = _summary_stats(rpt_nb)

        err = pd.to_numeric(succ["range_error_m"], errors="coerce").dropna().to_numpy(dtype=float)
        fr_counts = (
            g[g["success"] == False]["fail_reason"].astype(str).value_counts().to_dict()
            if "fail_reason" in g.columns
            else {}
        )
        fp_idx = pd.to_numeric(succ.get("first_path_index"), errors="coerce").dropna().to_numpy(dtype=float)
        pk_idx = pd.to_numeric(succ.get("peak_index"), errors="coerce").dropna().to_numpy(dtype=float)
        thr_abs = pd.to_numeric(succ.get("detection_threshold_abs"), errors="coerce").dropna().to_numpy(dtype=float)
        thr_noise = pd.to_numeric(succ.get("first_path_thr_noise_abs"), errors="coerce").dropna().to_numpy(dtype=float)
        thr_peak = pd.to_numeric(succ.get("first_path_thr_peak_abs"), errors="coerce").dropna().to_numpy(dtype=float)
        fp_snr_corr = pd.to_numeric(succ.get("first_path_snr_corr_db"), errors="coerce").dropna().to_numpy(dtype=float)
        fp_ratio = pd.to_numeric(succ.get("first_path_peak_ratio_db"), errors="coerce").dropna().to_numpy(dtype=float)
        fp_fallback = pd.to_numeric(succ.get("first_path_fallback_rate"), errors="coerce").dropna().to_numpy(dtype=float)
        noise_floor = pd.to_numeric(succ.get("first_path_noise_floor"), errors="coerce").dropna().to_numpy(dtype=float)
        noise_win_s = pd.to_numeric(succ.get("noise_win_start"), errors="coerce").dropna().to_numpy(dtype=float)
        noise_win_e = pd.to_numeric(succ.get("noise_win_end"), errors="coerce").dropna().to_numpy(dtype=float)
        ds_tof = pd.to_numeric(succ.get("ds_tof_rstu"), errors="coerce").dropna().to_numpy(dtype=float)
        ds_ra = pd.to_numeric(succ.get("ds_ra_rstu"), errors="coerce").dropna().to_numpy(dtype=float)
        ds_rb = pd.to_numeric(succ.get("ds_rb_rstu"), errors="coerce").dropna().to_numpy(dtype=float)
        ds_da = pd.to_numeric(succ.get("ds_da_rstu"), errors="coerce").dropna().to_numpy(dtype=float)
        ds_db = pd.to_numeric(succ.get("ds_db_rstu"), errors="coerce").dropna().to_numpy(dtype=float)
        tof_ns = pd.to_numeric(succ.get("estimated_tof_ns"), errors="coerce").dropna().to_numpy(dtype=float)
        sample_ns = pd.to_numeric(succ.get("sample_period_ns"), errors="coerce").dropna().to_numpy(dtype=float)
        cal_ns = pd.to_numeric(succ.get("applied_calibration_offset_ns"), errors="coerce").dropna().to_numpy(dtype=float)
        if err.size:
            bias = float(np.mean(err))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
        else:
            bias = float("nan")
            rmse = float("nan")
            mae = float("nan")
        fp_mean = float(np.mean(fp_idx)) if fp_idx.size else float("nan")
        fp_std = float(np.std(fp_idx)) if fp_idx.size else float("nan")
        pk_mean = float(np.mean(pk_idx)) if pk_idx.size else float("nan")
        thr_mean = float(np.mean(thr_abs)) if thr_abs.size else float("nan")
        thr_noise_mean = float(np.mean(thr_noise)) if thr_noise.size else float("nan")
        thr_peak_mean = float(np.mean(thr_peak)) if thr_peak.size else float("nan")
        fp_snr_corr_mean = float(np.mean(fp_snr_corr)) if fp_snr_corr.size else float("nan")
        fp_ratio_mean = float(np.mean(fp_ratio)) if fp_ratio.size else float("nan")
        fp_fallback_mean = float(np.mean(fp_fallback)) if fp_fallback.size else float("nan")
        noise_floor_mean = float(np.mean(noise_floor)) if noise_floor.size else float("nan")
        noise_win_s_mean = float(np.mean(noise_win_s)) if noise_win_s.size else float("nan")
        noise_win_e_mean = float(np.mean(noise_win_e)) if noise_win_e.size else float("nan")
        ds_tof_mean = float(np.mean(ds_tof)) if ds_tof.size else float("nan")
        ds_ra_mean = float(np.mean(ds_ra)) if ds_ra.size else float("nan")
        ds_rb_mean = float(np.mean(ds_rb)) if ds_rb.size else float("nan")
        ds_da_mean = float(np.mean(ds_da)) if ds_da.size else float("nan")
        ds_db_mean = float(np.mean(ds_db)) if ds_db.size else float("nan")
        tof_mean = float(np.mean(tof_ns)) if tof_ns.size else float("nan")
        sample_mean = float(np.mean(sample_ns)) if sample_ns.size else float("nan")
        cal_mean = float(np.mean(cal_ns)) if cal_ns.size else float("nan")
        # Convert range bias to equivalent ToF tick delay for diagnostic consistency.
        bias_tof_ns = float((bias / 299_792_458.0) * 1e9) if np.isfinite(bias) else float("nan")
        bias_ticks = float(bias_tof_ns / sample_mean) if (np.isfinite(bias_tof_ns) and np.isfinite(sample_mean) and sample_mean > 0.0) else float("nan")
        noise_margin = float(noise_win_e_mean - fp_mean) if (np.isfinite(noise_win_e_mean) and np.isfinite(fp_mean)) else float("nan")

        agg_rows.append(
            {
                "distance_m": float(d),
                "n_trials": n,
                "success_count": int(np.sum(g["success"] == True)),
                "P_init": p_init,
                "P_range_given_init": p_range_given_init,
                "P_overall": p_overall,
                "lat_succ_median_ms": lat_s_med,
                "lat_succ_p95_ms": lat_s_p95,
                "lat_all_median_ms": lat_a_med,
                "lat_all_p95_ms": lat_a_p95,
                "lat_lbt_cca_median_ms": lbt_med,
                "lat_lbt_cca_p95_ms": lbt_p95,
                "lat_backoff_median_ms": bo_med,
                "lat_backoff_p95_ms": bo_p95,
                "lat_uwb_exchange_median_ms": uwb_med,
                "lat_uwb_exchange_p95_ms": uwb_p95,
                "lat_adv_scan_wait_median_ms": advscan_med,
                "lat_adv_scan_wait_p95_ms": advscan_p95,
                "init_scan_wait_median_ms": initw_med,
                "init_scan_wait_p95_ms": initw_p95,
                "nb_ctrl_txrx_median_ms": nbctrl_med,
                "nb_ctrl_txrx_p95_ms": nbctrl_p95,
                "lat_report_median_ms": rpt_med,
                "lat_report_p95_ms": rpt_p95,
                "nb_report_txrx_median_ms": rpt_nb_med,
                "nb_report_txrx_p95_ms": rpt_nb_p95,
                "fail_reason_top": json.dumps(fr_counts, separators=(",", ":"), ensure_ascii=True),
                "range_bias_m": bias,
                "range_rmse_m": rmse,
                "range_mae_m": mae,
                "first_path_index_mean": fp_mean,
                "first_path_index_std": fp_std,
                "peak_index_mean": pk_mean,
                "detection_threshold_abs_mean": thr_mean,
                "first_path_thr_noise_abs_mean": thr_noise_mean,
                "first_path_thr_peak_abs_mean": thr_peak_mean,
                "first_path_noise_floor_mean": noise_floor_mean,
                "first_path_snr_corr_db_mean": fp_snr_corr_mean,
                "first_path_peak_ratio_db_mean": fp_ratio_mean,
                "first_path_fallback_rate_mean": fp_fallback_mean,
                "noise_win_start_mean": noise_win_s_mean,
                "noise_win_end_mean": noise_win_e_mean,
                "noise_win_end_minus_kfp_mean": noise_margin,
                "ds_tof_rstu_mean": ds_tof_mean,
                "ds_ra_rstu_mean": ds_ra_mean,
                "ds_rb_rstu_mean": ds_rb_mean,
                "ds_da_rstu_mean": ds_da_mean,
                "ds_db_rstu_mean": ds_db_mean,
                "estimated_tof_ns_mean": tof_mean,
                "sample_period_ns_mean": sample_mean,
                "applied_calibration_offset_ns_mean": cal_mean,
                "range_bias_equiv_tof_ns": bias_tof_ns,
                "range_bias_equiv_ticks": bias_ticks,
            }
        )

    agg_df = pd.DataFrame(agg_rows).sort_values("distance_m")
    agg_csv = out_dir / "aggregate_results.csv"
    agg_df.to_csv(agg_csv, index=False)

    plot_files = _plot(agg_df, df, out_dir)

    rep = out_dir / "report.md"
    # Timeline sample from first trial JSON, if present.
    timeline_sample_lines: list[str] = []
    tjs = sorted(timeline_dir.glob("trial_*.json"))
    if tjs:
        try:
            ev = json.loads(tjs[0].read_text())
            for e in ev[:20]:
                timeline_sample_lines.append(
                    f"- {float(e.get('start_ms', float('nan'))):.6f} -> {float(e.get('end_ms', float('nan'))):.6f} ms : {e.get('label', 'unknown')}"
                )
        except Exception:
            timeline_sample_lines.append("- (failed to parse timeline sample)")

    lines = [
        "# MMS Ranging Distance Sweep Report",
        "",
        "- Scenario: initiation + one MMS ranging session, Wi-Fi OFF",
        f"- enable_init_scan_model: {int(args.enable_init_scan_model)}",
        f"- scan_interval_ms={float(args.scan_interval_ms)}, scan_window_ms={float(args.scan_window_ms)}, "
        f"adv_interval_ms={float(args.adv_interval_ms)}, adv_tx_duration_ms={float(args.adv_tx_duration_ms)}, "
        f"random_start_phase={int(args.random_start_phase)}",
        f"- report_phase_model={int(args.enable_report_phase_model)}, "
        f"initiator_report_request={int(args.initiator_report_request)}, "
        f"responder_report_request={int(args.responder_report_request)}, "
        f"mms1stReportNSlots={int(args.mms1stReportNSlots)}, mms2ndReportNSlots={int(args.mms2ndReportNSlots)}, "
        f"assume_oob_report_on_missing={int(args.assume_oob_report_on_missing)}",
        "- Note: ToA compute wall-clock is not explicitly modeled in this path (lat_processing_ms NaN by design).",
        "- Trial timeline JSON: results/.../timelines/trial_d*_t*.json",
        f"- distances: {distances}",
        f"- trials per distance: {int(args.trials)}",
        f"- max_attempts: {int(args.max_attempts)}, until_success={int(args.until_success)}",
        f"- nb_eirp_dbw={float(args.nb_eirp_dbw):.2f}, nf_db={float(args.nf_db):.2f}",
        f"- uwb_shots_per_session: {int(args.uwb_shots_per_session)}, require_k_successes={int(args.require_k_successes)}",
        f"- first_path_thr_db={float(args.first_path_thr_db):.3f}, first_path_peak_frac={float(args.first_path_peak_frac):.3f}, "
        f"corr_upsample={int(args.corr_upsample)}, corr_win={int(args.corr_win)}, "
        f"fp_snr_switch_db={float(args.fp_snr_switch_db):.3f}, fp_thr_min_floor_mult={float(args.fp_thr_min_floor_mult):.3f}, "
        f"first_path_search_back={int(args.first_path_search_back)}",
        f"- range_bias_correction_m={float(args.range_bias_correction_m):.6f}, "
        f"toa_calibration_override_samples={('None' if not np.isfinite(float(args.toa_calibration_override_samples)) else float(args.toa_calibration_override_samples))}",
        f"- freeze_toa_calibration={int(args.freeze_toa_calibration)}, toa_calibration_distance_m={('auto(first distance)' if not np.isfinite(float(args.toa_calibration_distance_m)) else float(args.toa_calibration_distance_m))}",
        f"- toa_calibration_use_runtime_channel={int(args.toa_calibration_use_runtime_channel)}",
        f"- channel_profile={args.channel_profile}",
        f"- channel_delays_ns={list(float(v*1e9) for v in ch_delays_s)}, channel_powers_db={list(ch_powers_db)}, "
        f"channel_k_factor_db={float(ch_k_factor_db):.2f}, channel_delay_jitter_ns={float(delay_jitter_ns):.3f}, "
        f"channel_power_jitter_db={float(power_jitter_db):.3f}, "
        f"realism_preset={args.realism_preset}",
        "",
        "## Latency Field Definitions",
        "| field | definition |",
        "|---|---|",
        "| latency_to_success_ms | trial start -> first successful session completion (when success=True) |",
        "| time_spent_ms | trial start -> trial termination (success or fail) |",
        "| latency_to_conf_done_ms | trial start -> NB control completion timestamp |",
        "| lat_total_ms | trial start -> trial end (same axis as time_spent_ms) |",
        "| lat_adv_scan_wait_ms | adv/scan acquisition wait (NaN when init-scan model disabled) |",
        "| lat_lbt_cca_ms | SSBD CCA duration sum |",
        "| lat_backoff_ms | SSBD deferral/backoff sum |",
        "| lat_uwb_exchange_ms | UWB ranging exchange airtime sum |",
        "| report_phase_ms | modeled NB report phase duration sum |",
        "",
        "## Aggregate",
    ]
    try:
        lines.append(agg_df.to_markdown(index=False))
    except Exception:
        lines.append(agg_df.to_string(index=False))
    lines.append("")
    lines.append("## Diagnosis Summary")
    if int(args.enable_init_scan_model) == 0:
        lines.append("- `enable_init_scan_model=0`: adv/scan wait is intentionally unmodeled, so latency is mostly fixed NB/UWB airtime.")
    else:
        lines.append("- `enable_init_scan_model=1`: random phase alignment introduces wait variance (see `lat_adv_scan_wait_*` and `init_scan_wait_*`).")
    lines.append("- 5 m bias triage uses `range_bias_equiv_ticks`, `first_path_index_mean`, `peak_index_mean`, and noise-window metrics.")
    lines.append("- If `noise_win_end_minus_kfp_mean > 0`, noise window can overlap first-path region (possible threshold bias source).")
    if not agg_df.empty and "distance_m" in agg_df.columns:
        d5 = agg_df[np.isclose(pd.to_numeric(agg_df["distance_m"], errors="coerce"), 5.0)]
        dfar = agg_df[~np.isclose(pd.to_numeric(agg_df["distance_m"], errors="coerce"), 5.0)]
        if not d5.empty:
            r5 = d5.iloc[0]
            lines.append(
                f"- 5m evidence: bias={float(r5.get('range_bias_m', float('nan'))):.4f} m, "
                f"bias_ticks={float(r5.get('range_bias_equiv_ticks', float('nan'))):.4f}, "
                f"k_fp_mean={float(r5.get('first_path_index_mean', float('nan'))):.4f}, "
                f"k_peak_mean={float(r5.get('peak_index_mean', float('nan'))):.4f}, "
                f"noise_win_end-kfp={float(r5.get('noise_win_end_minus_kfp_mean', float('nan'))):.4f}"
            )
        if not dfar.empty:
            lines.append(
                f"- 10m+ average bias={float(pd.to_numeric(dfar['range_bias_m'], errors='coerce').mean()):.4f} m, "
                f"average bias_ticks={float(pd.to_numeric(dfar['range_bias_equiv_ticks'], errors='coerce').mean()):.4f}"
            )
    lines.append("")
    lines.append("## Timeline Sample (first dumped trial, first 20 events)")
    if timeline_sample_lines:
        lines.extend(timeline_sample_lines)
    else:
        lines.append("- (no timeline dumped)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- If latency appears near-constant with init-scan model disabled, this is expected: fixed NB+UWB airtime dominates.")
    lines.append("- For init-scan enabled runs, lat_adv_scan_wait_ms should show non-zero spread from random phase alignment.")
    lines.append("- 5 m bias diagnosis uses first_path_index / peak_index / threshold / tick-equivalent bias columns above.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- trial_results.csv")
    lines.append(f"- aggregate_results.csv")
    for f in plot_files:
        lines.append(f"- {f}")
    rep.write_text("\n".join(lines))

    print(f"saved: {out_dir}")
    print(f"- {trial_csv.name}")
    print(f"- {agg_csv.name}")
    print(f"- {rep.name}")
    for f in plot_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
