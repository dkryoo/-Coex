from __future__ import annotations

"""
Duty-cycle based Wi-Fi layout study for end-to-end MMS (NB init/control + UWB ranging).

Examples:
1) Uniform, hopping OFF/ON, low/mid/high duty
   python simulation/mms/run_layout_duty_study.py \
     --layout-file simulation/mms/layouts/wifi_layout_6g_160_example.json \
     --trials 200 --duty-levels "0.1,0.5,0.8"

2) Bursty-only
   python simulation/mms/run_layout_duty_study.py \
     --layout-file simulation/mms/layouts/wifi_layout_6g_160_example.json \
     --occupancy-modes "bursty" --trials 300

3) Force SSBD timeout behavior (TxOnEnd=0)
   python simulation/mms/run_layout_duty_study.py \
     --layout-file simulation/mms/layouts/wifi_layout_6g_160_example.json \
     --ssbd-tx-on-end 0 --trials 150
"""

import argparse
import copy
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.wifi_spatial_model import WiFiSpatialModel, load_wifi_layout
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.wifi_spatial_model import WiFiSpatialModel, load_wifi_layout


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _condition_label(occupancy_mode: str, hopping_on: bool) -> str:
    return f"{occupancy_mode}_{'hop_on' if hopping_on else 'hop_off'}"


def _quantile_or_nan(x: pd.Series, q: float) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _lat_stats_success(df: pd.DataFrame) -> tuple[float, float, float]:
    ok = df[df["success"] == True]
    vals = pd.to_numeric(ok["latency_to_success_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95))


def _lat_stats_fail(df: pd.DataFrame) -> tuple[float, float]:
    fail = df[df["success"] == False]
    vals = pd.to_numeric(fail["time_spent_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.median(vals)), float(np.percentile(vals, 95))


def _attempt_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    vals = pd.to_numeric(df["attempts_used"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95))


def _set_layout_condition(base_layout, duty_cycle: float, occupancy_mode: str, seed: int):
    lc = copy.deepcopy(base_layout)
    lc.seed = int(seed)
    lc.duty_cycle = float(np.clip(duty_cycle, 0.0, 1.0))
    if lc.aps:
        for ap in lc.aps:
            ap["duty_cycle"] = float(np.clip(duty_cycle, 0.0, 1.0))
            ap["traffic_load"] = float(np.clip(duty_cycle, 0.0, 1.0))
            ap["occupancy_mode"] = str(occupancy_mode)
            if occupancy_mode == "bursty":
                ap.setdefault("burst_mean_on_ms", 2.0)
                ap.setdefault("burst_mean_off_ms", 2.0)
    return lc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Duty-cycle based Wi-Fi layout MMS study")
    p.add_argument("--layout-file", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="simulation/mms/results/layout_duty_study")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260303)
    p.add_argument("--paired-trials", action="store_true", default=True)
    p.add_argument("--duty-levels", type=str, default="0.1,0.5,0.8")
    p.add_argument("--duty-sweep", choices=["custom", "wide", "knee"], default="custom")
    p.add_argument("--occupancy-modes", type=str, default="uniform,bursty")
    p.add_argument("--hopping-modes", type=str, default="off,on")
    p.add_argument("--dist-m", type=float, default=20.0)
    p.add_argument("--uwb-channel", type=int, default=5)
    p.add_argument("--wifi-offset-mhz", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=16)
    p.add_argument("--max-attempts-list", type=str, default=None, help="Comma list, e.g. 8,16,32")
    p.add_argument("--until-success", action="store_true", default=True)
    p.add_argument("--single-attempt", dest="until_success", action="store_false")
    p.add_argument("--max-trial-ms", type=float, default=200.0)
    p.add_argument("--uwb-shots-per-session", type=int, default=1)
    p.add_argument("--require-k-successes", type=int, default=1)
    p.add_argument("--aggregation", choices=["median", "mean", "min"], default="median")
    p.add_argument("--uwb-shot-gap-ms", type=float, default=0.5)

    p.add_argument("--nb-allow-list-off", type=str, default="0")
    p.add_argument("--nb-allow-list-on", type=str, default="0,1,2,3,4,5")
    p.add_argument("--enable-uwb-channel-hopping", type=int, default=0)
    p.add_argument("--mms-prng-seed", type=int, default=0)
    p.add_argument("--nb-channel-spacing-mhz", type=float, default=2.0)

    p.add_argument("--ssbd-tx-on-end", type=int, default=1)
    p.add_argument("--ssbd-persistence", type=int, default=0)
    p.add_argument("--ssbd-max-backoffs", type=int, default=5)
    p.add_argument("--ssbd-min-bf", type=int, default=1)
    p.add_argument("--ssbd-max-bf", type=int, default=5)
    p.add_argument("--ssbd-unit-backoff-us", type=float, default=1.0)
    p.add_argument("--phy-cca-duration-ms", type=float, default=None)
    p.add_argument("--phy-cca-ed-threshold-dbm", type=float, default=None)

    p.add_argument("--print-ssbd-trace", action="store_true")
    p.add_argument("--ssbd-debug", action="store_true")
    p.add_argument("--dump-nb-channel-seq", type=int, default=0)
    p.add_argument("--print-health-check", action="store_true")
    p.add_argument("--health-check", action="store_true", help="Run compact automatic health suite (duty=0/0.1/0.5).")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--early-stop-zero-success-min-trials", type=int, default=0)
    return p


def _plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    agg = (
        df.groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "success_prob": float(np.mean(g["success"].astype(float))),
                    "lat_all_mean_ms": float(pd.to_numeric(g["time_spent_ms"], errors="coerce").mean()),
                    "lat_succ_median_ms": _quantile_or_nan(g[g["success"] == True]["latency_to_success_ms"], 50),
                    "lat_succ_p95_ms": _quantile_or_nan(g[g["success"] == True]["latency_to_success_ms"], 95),
                    "success_count": int(np.sum(g["success"] == True)),
                    "n": int(len(g)),
                }
            )
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for (occ, hop, ma), g in agg.groupby(["occupancy_mode", "hopping", "max_attempts_cfg"], dropna=False):
        gg = g.sort_values("duty_cycle")
        ax.plot(gg["duty_cycle"], gg["success_prob"], marker="o", label=f"{occ}-{hop}-m{int(ma)}")
    ax.set_title("P_overall vs duty_cycle")
    ax.set_xlabel("duty_cycle")
    ax.set_ylabel("P_overall")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_p_overall_vs_duty.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (occ, hop, ma), g in agg.groupby(["occupancy_mode", "hopping", "max_attempts_cfg"], dropna=False):
        gg = g.sort_values("duty_cycle")
        tag = f"{occ}-{hop}-m{int(ma)}"
        ax.plot(gg["duty_cycle"], gg["lat_all_mean_ms"], marker="o", linestyle="-", label=f"{tag} all_mean")
        ax.plot(gg["duty_cycle"], gg["lat_succ_median_ms"], marker="x", linestyle="--", label=f"{tag} succ_med")
        ax.plot(gg["duty_cycle"], gg["lat_succ_p95_ms"], marker="^", linestyle=":", label=f"{tag} succ_p95")
        for _, r in gg.iterrows():
            if int(r["success_count"]) == 0:
                ax.annotate("0 success", (r["duty_cycle"], r["lat_all_mean_ms"]), fontsize=7)
    ax.set_title("Latency vs duty_cycle")
    ax.set_xlabel("duty_cycle")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_latency_vs_duty.png", dpi=160)
    plt.close(fig)

    # Fail reason stacked bar
    fr = (
        df[df["success"] == False]
        .groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle", "terminal_fail_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not fr.empty:
        fr["x"] = fr.apply(
            lambda r: f"{r['occupancy_mode']}-{r['hopping']}-m{int(r['max_attempts_cfg'])}-d{r['duty_cycle']:.2f}",
            axis=1,
        )
        piv = fr.pivot_table(index="x", columns="terminal_fail_reason", values="count", aggfunc="sum", fill_value=0)
        ax = piv.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
        ax.set_title("Fail reason stacked counts")
        ax.set_xlabel("condition")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_fail_reason_stacked.png", dpi=160)
        plt.close()


def _write_health_report(df: pd.DataFrame, agg_df: pd.DataFrame, out_dir: Path, *, layout_file: str) -> Path:
    def _tbl(d: pd.DataFrame) -> str:
        try:
            return d.to_markdown(index=False)
        except Exception:
            return d.to_string(index=False)

    report = out_dir / "health_report.md"
    lines: list[str] = []
    lines.append("# MMS End-to-End Health Report")
    lines.append("")
    lines.append(f"- layout_source = fixed_file")
    lines.append(f"- layout_file = {layout_file}")
    lines.append("")
    lines.append("## 1st Conclusion (before fixes)")
    lines.append(
        "- Low success at duty=0.1 can be caused by model composition/run-mode mismatch (single-attempt vs until-success), "
        "and by incorrect interference path composition (double counting or fixed-power gating)."
    )
    lines.append("")
    lines.append("## Core Indicators")
    core_cols = [
        "occupancy_mode",
        "hopping",
        "max_attempts_cfg",
        "duty_cycle",
        "P_init",
        "P_range_given_init",
        "P_overall",
        "attempts_used_mean",
        "nb_ssbd_deferral_mean_ms",
    ]
    lines.append(_tbl(agg_df[core_cols]))
    lines.append("")
    if {"wifi_busy_frac_nb", "wifi_busy_frac_uwb", "nb_cca_busy_rate", "wifi_realized_duty_mean"}.issubset(df.columns):
        d2 = (
            df.groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"], dropna=False)[
                ["wifi_busy_frac_nb", "wifi_busy_frac_uwb", "nb_cca_busy_rate", "wifi_realized_duty_mean"]
            ]
            .mean()
            .reset_index()
        )
        lines.append("## Busy Fraction / CCA Busy Rate / Realized Duty")
        lines.append(_tbl(d2))
        lines.append("")
    fr = (
        df[df["success"] == False]
        .groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle", "terminal_fail_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    lines.append("## Top Fail Reasons")
    if fr.empty:
        lines.append("- none")
    else:
        lines.append(_tbl(fr.head(20)))
    lines.append("")
    if "nb_channel_seq_first8_json" in df.columns:
        gcols = ["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"]
        seq_diag = (
            df.groupby(gcols, dropna=False)["nb_channel_seq_first8_json"]
            .nunique()
            .reset_index(name="unique_seq_first8")
        )
        seq_first = (
            df.groupby(gcols, dropna=False)["nb_channel_seq_first8_json"]
            .first()
            .reset_index(name="sample_seq_first8")
        )
        seq_mean = (
            df.groupby(gcols, dropna=False)["last_attempt_nb_channel"]
            .mean()
            .reset_index(name="mean_last_nb_channel")
        )
        seq_diag = seq_diag.merge(seq_first, on=gcols, how="left").merge(seq_mean, on=gcols, how="left")
        lines.append("## Hopping Sequence Diagnostic (unique first8 seq count)")
        lines.append(_tbl(seq_diag))
        lines.append("")
        lines.append("- If hop on/off metrics are identical while sequence diagnostics differ and P_init≈1, NB is likely not the bottleneck.")
        lines.append("- If sequence diagnostics are also identical, hopping wiring/allow-list config should be rechecked.")
    lines.append("")
    report.write_text("\n".join(lines))
    return report


def main() -> None:
    args = _build_parser().parse_args()
    print(
        "WARNING: This runner uses a fixed layout JSON and DOES NOT generate full-band partitions. "
        "Use run_random_6g_duty_sweep.py for random full-band partitions."
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_layout = load_wifi_layout(args.layout_file)
    std = get_default_params("802154ab")

    ssbd_cfg = SsbdConfig(
        phy_cca_duration_ms=float(args.phy_cca_duration_ms if args.phy_cca_duration_ms is not None else std.nb_phy_cca_duration_ms),
        phy_cca_ed_threshold_dbm=float(args.phy_cca_ed_threshold_dbm if args.phy_cca_ed_threshold_dbm is not None else std.nb_phy_cca_ed_threshold_dbm),
        cca_mode=1,
        mac_ssbd_unit_backoff_ms=float(args.ssbd_unit_backoff_us) / 1000.0,
        mac_ssbd_min_bf=int(args.ssbd_min_bf),
        mac_ssbd_max_bf=int(args.ssbd_max_bf),
        mac_ssbd_max_backoffs=int(args.ssbd_max_backoffs),
        mac_ssbd_tx_on_end=bool(int(args.ssbd_tx_on_end)),
        mac_ssbd_persistence=bool(int(args.ssbd_persistence)),
    )

    allow_off = tuple(sorted(set(_parse_int_list(args.nb_allow_list_off))))
    allow_on = tuple(sorted(set(_parse_int_list(args.nb_allow_list_on))))
    if not allow_off:
        allow_off = (0,)
    if not allow_on:
        allow_on = (0,)

    nb_cfg_off = NbChannelSwitchConfig(
        enable_switching=False,
        allow_list=allow_off,
        mms_prng_seed=int(args.mms_prng_seed) & 0xFF,
        channel_switching_field=0,
        nb_channel_spacing_mhz=float(args.nb_channel_spacing_mhz),
    )
    nb_cfg_on = NbChannelSwitchConfig(
        enable_switching=True,
        allow_list=allow_on,
        mms_prng_seed=int(args.mms_prng_seed) & 0xFF,
        channel_switching_field=1,
        nb_channel_spacing_mhz=float(args.nb_channel_spacing_mhz),
    )

    if int(args.dump_nb_channel_seq) > 0:
        from simulation.mms.nb_channel_switching import selected_nb_channel_for_block

        n_dump = int(args.dump_nb_channel_seq)
        seq_off = [int(selected_nb_channel_for_block(nb_cfg_off, i)) for i in range(n_dump)]
        seq_on = [int(selected_nb_channel_for_block(nb_cfg_on, i)) for i in range(n_dump)]
        print(f"[nb-switch-seq] off_first_{n_dump}={seq_off}")
        print(f"[nb-switch-seq] on_first_{n_dump}={seq_on}")

    cfg_base = FullStackConfig(
        distance_m=float(args.dist_m),
        nb_channel=1,
        uwb_channel=int(args.uwb_channel),
        wifi_channel=108,
        wifi_bw_mhz=160,
        wifi_standard="wifi7",
        nb_center_override_hz=6.489e9,
        nb_eirp_dbw=-16.0,
        wifi_tx_power_dbw=-10.0,
        nf_db=6.0,
        seed=int(args.seed),
    )

    if args.duty_sweep == "wide":
        duties = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
    elif args.duty_sweep == "knee":
        duties = [round(x, 2) for x in np.arange(0.40, 0.95, 0.05)]
    else:
        duties = _parse_float_list(args.duty_levels)
    occ_modes = [m.lower() for m in _parse_str_list(args.occupancy_modes)]
    hop_modes = [m.lower() for m in _parse_str_list(args.hopping_modes)]
    max_attempts_list = _parse_int_list(args.max_attempts_list) if args.max_attempts_list else [int(args.max_attempts)]
    if args.health_check:
        duties = [0.0, 0.1, 0.5]
        occ_modes = ["uniform", "bursty"]
        hop_modes = ["off", "on"]
        if int(args.trials) < 20:
            args.trials = 20

    rows: list[dict] = []
    total_conditions = len(occ_modes) * len(hop_modes) * len(duties) * len(max_attempts_list)
    cond_idx = 0
    for occ_mode in occ_modes:
        if occ_mode not in {"uniform", "bursty"}:
            raise ValueError(f"invalid occupancy_mode: {occ_mode}")
        for hop in hop_modes:
            if hop not in {"off", "on"}:
                raise ValueError(f"invalid hopping mode: {hop}")
            nb_cfg = nb_cfg_on if hop == "on" else nb_cfg_off
            for ma in max_attempts_list:
                for duty in duties:
                    cond_idx += 1
                    succ_count = 0
                    cond_runtime_s = 0.0
                    print(
                        f"[cond {cond_idx}/{total_conditions}] mode={occ_mode} hop={hop} duty={duty:.3f} "
                        f"trials={int(args.trials)} until_success={int(args.until_success)} max_attempts={int(ma)}"
                    )
                    for t in range(int(args.trials)):
                        seed = int(args.seed + t) if args.paired_trials else int(args.seed + 100000 * (len(rows) + 1) + t)
                        lc = _set_layout_condition(base_layout, duty_cycle=float(duty), occupancy_mode=occ_mode, seed=seed + 77)
                        spatial_model = WiFiSpatialModel(lc)

                        row = _run_trial(
                            cfg_base=replace(cfg_base, seed=seed),
                            wifi_mode="dense",
                            wifi_density=float(duty),
                            distance_m=float(args.dist_m),
                            uwb_channel=int(args.uwb_channel),
                            wifi_offset_mhz=float(args.wifi_offset_mhz),
                            trial_idx=t,
                            seed=seed,
                            max_attempts=int(ma),
                            until_success=bool(args.until_success),
                            max_trial_ms=float(args.max_trial_ms),
                            uwb_shots_per_session=int(args.uwb_shots_per_session),
                            require_k_successes=int(args.require_k_successes),
                            aggregation=str(args.aggregation),
                            uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
                            wifi_model="layout",
                            spatial_model=spatial_model,
                            nb_lbt_slot_ms=float(std.nb_lbt_slot_ms),
                            nb_lbt_cca_slots=int(std.nb_lbt_cca_slots),
                            ssbd_cfg=ssbd_cfg,
                            ssbd_debug=bool(args.ssbd_debug),
                            print_ssbd_trace=bool(args.print_ssbd_trace),
                            nb_switch_cfg=nb_cfg,
                        )
                        row["duty_cycle"] = float(duty)
                        row["occupancy_mode"] = occ_mode
                        row["hopping"] = hop
                        row["condition"] = _condition_label(occ_mode, hop == "on")
                        row["max_attempts_cfg"] = int(ma)
                        row["uwb_channel_hopping"] = int(args.enable_uwb_channel_hopping)
                        row["layout_file"] = str(args.layout_file)
                        row["init_success"] = bool(np.isfinite(float(row.get("latency_to_conf_done_ms", float("nan")))) and float(row.get("latency_to_conf_done_ms", float("nan"))) >= 0.0)
                        row["range_success_given_init"] = bool(row["success"] and row["init_success"])
                        rows.append(row)
                        succ_count += int(bool(row["success"]))
                        cond_runtime_s += float(row.get("sim_runtime_s", 0.0))
                        if int(args.progress_every) > 0 and ((t + 1) % int(args.progress_every) == 0 or (t + 1) == int(args.trials)):
                            print(
                                f"  - progress {t+1}/{int(args.trials)} "
                                f"succ={succ_count} runtime_avg={cond_runtime_s/max(1,(t+1)):.3f}s"
                            )
                        if (
                            int(args.early_stop_zero_success_min_trials) > 0
                            and (t + 1) >= int(args.early_stop_zero_success_min_trials)
                            and succ_count == 0
                        ):
                            print(
                                f"  - early-stop: zero successes after {t+1} trials "
                                f"(mode={occ_mode}, hop={hop}, duty={duty:.3f})"
                            )
                            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no rows generated")

    per_trial_csv = out_dir / f"layout_duty_trials_{stamp}.csv"
    df.to_csv(per_trial_csv, index=False)

    agg_rows: list[dict] = []
    for (occ, hop, ma, duty), g in df.groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"], dropna=False):
        p_init = float(np.mean(g["init_success"].astype(float)))
        init_ok = g[g["init_success"] == True]
        p_range_given_init = float(np.mean(init_ok["success"].astype(float))) if len(init_ok) > 0 else float("nan")
        p_overall = float(np.mean(g["success"].astype(float)))
        lat_mean, lat_median, lat_p95 = _lat_stats_success(g)
        fail_med, fail_p95 = _lat_stats_fail(g)
        att_mean, att_median, att_p95 = _attempt_stats(g)
        ssbd_def_ms_mean = float(pd.to_numeric(g["nb_ssbd_total_deferral_ms"], errors="coerce").mean())
        fail_counts = Counter(str(x) for x in g[g["success"] == False]["terminal_fail_reason"].tolist())
        agg_rows.append(
            {
                "occupancy_mode": str(occ),
                "hopping": str(hop),
                "max_attempts_cfg": int(ma),
                "duty_cycle": float(duty),
                "n_trials": int(len(g)),
                "success_count": int(np.sum(g["success"] == True)),
                "fail_count": int(np.sum(g["success"] == False)),
                "P_init": p_init,
                "P_range_given_init": p_range_given_init,
                "P_overall": p_overall,
                "latency_to_success_mean_ms": lat_mean,
                "latency_to_success_median_ms": lat_median,
                "latency_to_success_p95_ms": lat_p95,
                "fail_time_median_ms": fail_med,
                "fail_time_p95_ms": fail_p95,
                "attempts_used_mean": att_mean,
                "attempts_used_median": att_median,
                "attempts_used_p95": att_p95,
                "nb_ssbd_deferral_mean_ms": ssbd_def_ms_mean,
                "fail_reason_top": dict(fail_counts.most_common(5)),
            }
        )

    agg_df = pd.DataFrame(agg_rows).sort_values(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"])
    agg_csv = out_dir / f"layout_duty_aggregate_{stamp}.csv"
    agg_df.to_csv(agg_csv, index=False)

    _plot_results(df, out_dir)
    report = _write_health_report(df, agg_df, out_dir, layout_file=str(args.layout_file))

    if args.print_health_check:
        print("[health-check] duty=0.0 baseline probe")
        lc0 = _set_layout_condition(base_layout, duty_cycle=0.0, occupancy_mode="uniform", seed=int(args.seed + 999))
        r0 = _run_trial(
            cfg_base=replace(cfg_base, seed=int(args.seed + 999)),
            wifi_mode="dense",
            wifi_density=0.0,
            distance_m=float(args.dist_m),
            uwb_channel=int(args.uwb_channel),
            wifi_offset_mhz=float(args.wifi_offset_mhz),
            trial_idx=0,
            seed=int(args.seed + 999),
            max_attempts=int(args.max_attempts),
            until_success=True,
            max_trial_ms=float(args.max_trial_ms),
            uwb_shots_per_session=int(args.uwb_shots_per_session),
            require_k_successes=int(args.require_k_successes),
            aggregation=str(args.aggregation),
            uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
            wifi_model="layout",
            spatial_model=WiFiSpatialModel(lc0),
            nb_lbt_slot_ms=float(std.nb_lbt_slot_ms),
            nb_lbt_cca_slots=int(std.nb_lbt_cca_slots),
            ssbd_cfg=ssbd_cfg,
            ssbd_debug=False,
            print_ssbd_trace=False,
            nb_switch_cfg=nb_cfg_off,
        )
        print(
            f"[health-check] success={r0['success']} latency_to_success_ms={r0['latency_to_success_ms']} "
            f"fail_reason={r0['fail_reason']}"
        )

    print(f"saved per-trial CSV: {per_trial_csv}")
    print(f"saved aggregate CSV: {agg_csv}")
    print(f"saved plots: {out_dir / 'plot_p_overall_vs_duty.png'}, {out_dir / 'plot_latency_vs_duty.png'}, {out_dir / 'plot_fail_reason_stacked.png'}")
    print(f"saved health report: {report}")
    print("\nAggregate summary:")
    if not args.until_success:
        print("[note] single-attempt mode: latency is mostly one-shot airtime (~4.25 ms baseline).")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
