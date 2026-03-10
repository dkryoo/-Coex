from __future__ import annotations

"""
Random 6 GHz full-band duty sweep runner (Wi-Fi coexistence vs MMS E2E).

Quick:
python simulation/mms/run_random_6g_duty_sweep.py \
  --trials 10 --duty-levels "0.1,0.5,0.8" --max-attempts-list "8,16" \
  --occupancy-modes "mixed" --hopping-modes "off,on" --progress-every 5

Full:
python simulation/mms/run_random_6g_duty_sweep.py \
  --trials 200 --duty-sweep wide --max-attempts-list "8,16,32" \
  --occupancy-modes "mixed" --hopping-modes "off,on" --progress-every 20
"""

import argparse
from collections import Counter
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig, selected_nb_channel_for_block
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.random_wifi_layout import (
        RandomWiFiLayoutConfig,
        generate_random_wifi_layout,
        layout_distance_stats,
        validate_occupancy,
    )
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.wifi6g_channels import ChannelSegment
    from simulation.mms.validate_wifi6g_partition import validate_partition
    from simulation.mms.wifi_spatial_model import WiFiSpatialConfig, WiFiSpatialModel
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig, selected_nb_channel_for_block
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.random_wifi_layout import (
        RandomWiFiLayoutConfig,
        generate_random_wifi_layout,
        layout_distance_stats,
        validate_occupancy,
    )
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.wifi6g_channels import ChannelSegment
    from simulation.mms.validate_wifi6g_partition import validate_partition
    from simulation.mms.wifi_spatial_model import WiFiSpatialConfig, WiFiSpatialModel


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_bw_mix(s: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split(":")
        out[int(k)] = float(v)
    return out


def _wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    ph = success / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt((ph * (1.0 - ph) + z * z / (4 * n)) / n) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def _duty_values(args: argparse.Namespace) -> list[float]:
    if args.duty_levels:
        return _parse_float_list(args.duty_levels)
    if args.duty_sweep == "knee":
        return [round(x, 2) for x in np.arange(0.40, 0.95, 0.05)]
    return [round(x, 2) for x in np.arange(args.duty_start, args.duty_stop + 1e-12, args.duty_step)]


def _to_spatial_cfg(
    layout: dict,
    seed: int,
    *,
    aci_model: str = "overlap",
    aci_lut_path: str | None = None,
) -> WiFiSpatialConfig:
    return WiFiSpatialConfig(
        area_size_m=float(max(layout.get("area_w_m", 50.0), layout.get("area_h_m", 50.0))),
        n_ap=int(len(layout.get("wifi_aps", []))),
        ap_tx_power_dbm=20.0,
        pathloss_n=float(layout.get("pathloss_n", 2.8)),
        pl0_db=float(layout.get("pl0_db", 46.0)),
        shadowing_sigma_db=float(layout.get("shadowing_sigma_db", 4.0)),
        duty_cycle=0.0,
        seed=int(seed),
        enable_rician_fading=bool(layout.get("enable_rician_fading", True)),
        rician_k_factor_db=float(layout.get("rician_k_factor_db", 6.0)),
        aci_model=str(aci_model),
        aci_lut_path=aci_lut_path,
        aps=list(layout.get("wifi_aps", [])),
    )


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    g = (
        df.groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"], dropna=False)
        .agg(
            n=("success", "count"),
            succ=("success", "sum"),
            p_init=("init_success", "mean"),
            p_range_given_init=("range_success_given_init", "mean"),
            p_overall=("success", "mean"),
            lat_succ_med=("latency_to_success_ms", "median"),
            lat_succ_p95=("latency_to_success_ms", lambda x: float(np.nanpercentile(pd.to_numeric(x, errors="coerce"), 95))),
            lat_all_med=("time_spent_ms", "median"),
            lat_all_p95=("time_spent_ms", lambda x: float(np.nanpercentile(pd.to_numeric(x, errors="coerce"), 95))),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for (occ, hop, ma), d in g.groupby(["occupancy_mode", "hopping", "max_attempts_cfg"], dropna=False):
        dd = d.sort_values("duty_cycle")
        ax.plot(dd["duty_cycle"], dd["p_overall"], marker="o", label=f"{occ}-{hop}-m{int(ma)}")
    ax.set_title("P_overall vs duty")
    ax.set_xlabel("duty")
    ax.set_ylabel("P_overall")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_p_overall_vs_duty.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (occ, hop, ma), d in g.groupby(["occupancy_mode", "hopping", "max_attempts_cfg"], dropna=False):
        dd = d.sort_values("duty_cycle")
        tag = f"{occ}-{hop}-m{int(ma)}"
        ax.plot(dd["duty_cycle"], dd["lat_succ_med"], marker="x", linestyle="--", label=f"{tag} succ_med")
        ax.plot(dd["duty_cycle"], dd["lat_succ_p95"], marker="^", linestyle=":", label=f"{tag} succ_p95")
        ax.plot(dd["duty_cycle"], dd["lat_all_med"], marker="o", linestyle="-", label=f"{tag} all_med")
    ax.set_title("Latency vs duty")
    ax.set_xlabel("duty")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_latency_vs_duty.png", dpi=160)
    plt.close(fig)

    fr = (
        df[df["success"] == False]
        .groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle", "terminal_fail_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not fr.empty:
        fr["x"] = fr.apply(lambda r: f"{r['occupancy_mode']}-{r['hopping']}-m{int(r['max_attempts_cfg'])}-d{r['duty_cycle']:.2f}", axis=1)
        piv = fr.pivot_table(index="x", columns="terminal_fail_reason", values="count", aggfunc="sum", fill_value=0)
        ax = piv.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
        ax.set_title("Terminal fail reason vs duty")
        ax.set_xlabel("condition")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_terminal_fail_reason_stacked.png", dpi=160)
        plt.close()


def _health_report(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    out_dir: Path,
    n_dump: int,
    *,
    sample_partition_json: str | None = None,
    sample_aps_json: str | None = None,
) -> None:
    lines = ["# Health Report", ""]
    lines.append("- layout_source = random_fullband_partition")
    lines.append("")
    if not agg.empty:
        lines.append("## Coverage/Hopping/Duty sanity")
        lines.append(agg.to_string(index=False))
        lines.append("")
    if not df.empty:
        lines.append("## Partition validation summary")
        cov = pd.to_numeric(df.get("coverage_ratio"), errors="coerce")
        ok = df.get("partition_ok")
        lines.append(
            f"- coverage_ratio min/mean/max = {float(np.nanmin(cov)):.4f}/{float(np.nanmean(cov)):.4f}/{float(np.nanmax(cov)):.4f}"
        )
        lines.append(f"- partition_ok count = {int(np.sum(ok == True))}/{int(len(df))}")
        n_ap_vals = pd.to_numeric(df.get("n_ap"), errors="coerce")
        lines.append(
            f"- n_ap min/mean/max = {int(np.nanmin(n_ap_vals))}/{float(np.nanmean(n_ap_vals)):.2f}/{int(np.nanmax(n_ap_vals))}"
        )
        lines.append("")
        lines.append("## Occupancy validation summary")
        occ_frac = pd.to_numeric(df.get("occ_bursty_frac"), errors="coerce")
        lines.append(
            f"- occ_bursty_frac min/mean/max = {float(np.nanmin(occ_frac)):.3f}/{float(np.nanmean(occ_frac)):.3f}/{float(np.nanmax(occ_frac)):.3f}"
        )
        duty_hat = pd.to_numeric(df.get("duty_hat_mean_bursty"), errors="coerce")
        if np.any(np.isfinite(duty_hat.to_numpy(dtype=float))):
            lines.append(
                f"- duty_hat_mean_bursty min/mean/max = {float(np.nanmin(duty_hat)):.3f}/{float(np.nanmean(duty_hat)):.3f}/{float(np.nanmax(duty_hat)):.3f}"
            )
        if "partition_bw_hist_json" in df.columns:
            hists = [str(v) for v in df["partition_bw_hist_json"].dropna().astype(str).tolist()]
            lines.append("- bw_hist top patterns:")
            for k, v in Counter(hists).most_common(5):
                lines.append(f"  - {v}x {k}")
        if "aci_lut_hit_rate" in df.columns:
            hit = pd.to_numeric(df.get("aci_lut_hit_rate"), errors="coerce")
            miss_off = pd.to_numeric(df.get("aci_lut_missing_offsets_count"), errors="coerce")
            lines.append("- ACI LUT:")
            lines.append(
                f"  - hit_rate min/mean/max = {float(np.nanmin(hit)):.3f}/{float(np.nanmean(hit)):.3f}/{float(np.nanmax(hit)):.3f}"
            )
            lines.append(
                f"  - missing_offsets_count sum = {int(np.nansum(miss_off))}"
            )
        lines.append("")
    lines.append("## Answers")
    lines.append("- Coverage ratio should be 1.0 in partition summary per trial.")
    lines.append("- Hopping OFF must keep NB channel constant lowest; ON should vary with block index.")
    lines.append("- duty scaling sanity uses wifi_realized_duty_mean / wifi_busy_frac proxies.")
    lines.append("")
    if "nb_channel_seq_first8_json" in df.columns and not df.empty:
        ex_off = df[df["hopping"] == "off"].head(1)
        ex_on = df[df["hopping"] == "on"].head(1)
        lines.append(f"- sample OFF seq first{n_dump}: {ex_off['nb_channel_seq_first8_json'].iloc[0] if not ex_off.empty else 'N/A'}")
        lines.append(f"- sample ON  seq first{n_dump}: {ex_on['nb_channel_seq_first8_json'].iloc[0] if not ex_on.empty else 'N/A'}")
    if sample_partition_json:
        lines.append("")
        lines.append("## Sample trial partition (first trial)")
        lines.append(sample_partition_json)
    if sample_aps_json:
        lines.append("")
        lines.append("## Sample AP configs (first 5)")
        lines.append(sample_aps_json)
    (out_dir / "health_report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Random 6GHz duty sweep for MMS coexistence")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260304)
    ap.add_argument("--duty-start", type=float, default=0.05)
    ap.add_argument("--duty-stop", type=float, default=0.95)
    ap.add_argument("--duty-step", type=float, default=0.05)
    ap.add_argument("--duty-levels", type=str, default=None)
    ap.add_argument("--duty-sweep", choices=["wide", "knee"], default="wide")
    ap.add_argument("--occupancy-modes", type=str, default="mixed")
    ap.add_argument("--p-bursty", type=float, default=0.5)
    ap.add_argument("--hopping-modes", type=str, default="off,on")
    ap.add_argument("--max-attempts-list", type=str, default="8,16,32")
    ap.add_argument("--bw-mix-weights", type=str, default="20:0.3,40:0.3,80:0.2,160:0.2")
    ap.add_argument("--area-w-m", type=float, default=50.0)
    ap.add_argument("--area-h-m", type=float, default=50.0)
    ap.add_argument("--dist-m", type=float, default=20.0)
    ap.add_argument("--uwb-channel", type=int, default=5)
    ap.add_argument("--wifi-offset-mhz", type=float, default=0.0)
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="simulation/mms/results/layout_duty_study_v2")
    ap.add_argument("--dump-nb-hopping-seq", type=int, default=8)
    ap.add_argument("--health-dump-first-k-trials", type=int, default=2)
    ap.add_argument("--partition-debug", action="store_true")
    ap.add_argument("--occupancy-debug", action="store_true")
    ap.add_argument("--mixed-assign-mode", choices=["binomial", "fixed"], default="binomial")
    ap.add_argument("--aci-model", choices=["overlap", "lut"], default="overlap")
    ap.add_argument("--aci-lut-path", type=str, default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

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

    duties = _duty_values(args)
    occ_modes = _parse_str_list(args.occupancy_modes)
    hop_modes = _parse_str_list(args.hopping_modes)
    max_attempts_list = _parse_int_list(args.max_attempts_list)
    bw_mix = _parse_bw_mix(args.bw_mix_weights)

    rows: list[dict] = []
    sample_partition_json: str | None = None
    sample_aps_json: str | None = None
    cond_total = len(duties) * len(occ_modes) * len(hop_modes) * len(max_attempts_list)
    cond_idx = 0

    for occ in occ_modes:
        for hop in hop_modes:
            for ma in max_attempts_list:
                allow_off = (0,)
                allow_on = (0, 1, 2, 3, 4, 5)
                nb_cfg = NbChannelSwitchConfig(
                    enable_switching=(hop == "on"),
                    allow_list=allow_on if hop == "on" else allow_off,
                    mms_prng_seed=int(args.seed) & 0xFF,
                    channel_switching_field=1 if hop == "on" else 0,
                    nb_channel_spacing_mhz=2.0,
                )
                if cond_idx == 0 and int(args.dump_nb_hopping_seq) > 0:
                    n = int(args.dump_nb_hopping_seq)
                    seq = [int(selected_nb_channel_for_block(nb_cfg, i)) for i in range(n)]
                    print(f"[nb-hopping-seq] hop={hop} first_{n}={seq}")
                for duty in duties:
                    cond_idx += 1
                    succ = 0
                    print(
                        f"[cond {cond_idx}/{cond_total}] occ={occ} hop={hop} duty={duty:.2f} max_attempts={ma} trials={args.trials}"
                    )
                    for t in range(int(args.trials)):
                        trial_seed = int(args.seed + 100000 * cond_idx + t)
                        rng = np.random.default_rng(trial_seed)
                        lcfg = RandomWiFiLayoutConfig(
                            area_w_m=float(args.area_w_m),
                            area_h_m=float(args.area_h_m),
                            duty_cycle=float(duty),
                            p_bursty=float(args.p_bursty),
                        )
                        layout = generate_random_wifi_layout(
                            rng=rng,
                            layout_cfg=lcfg,
                            bw_mix_weights=bw_mix,
                            occupancy_mode=str(occ),
                            generated_trial_id=f"c{cond_idx}_t{t}",
                            mixed_assign_mode=str(args.mixed_assign_mode),
                        )
                        segs = []
                        for s in list(layout.get("band_partition_segments", [])):
                            segs.append(
                                ChannelSegment(
                                    bw_mhz=int(s["bw_mhz"]),
                                    center_ch=int(s["center_ch"]),
                                    f_center_hz=float(s["f_center_hz"]),
                                    f_lo_hz=float(s["f_lo_hz"]),
                                    f_hi_hz=float(s["f_hi_hz"]),
                                    covered_20ch_list=[int(v) for v in s["covered_20ch_list"]],
                                )
                            )
                        try:
                            part_val = validate_partition(segs)
                            occ_tol = max(0.10, 1.5 / max(len(layout.get("wifi_aps", [])), 1))
                            occ_val = validate_occupancy(layout, str(occ), float(args.p_bursty), mixed_tol=occ_tol)
                        except AssertionError as exc:
                            raise RuntimeError(
                                f"validation failed at cond={cond_idx}, trial={t}, seed={trial_seed}: {exc}"
                            ) from exc
                        dump_this = t < int(max(0, args.health_dump_first_k_trials))
                        if dump_this:
                            print(
                                f"[health] cond={cond_idx} trial={t} partition_ok={bool(part_val.get('coverage_ok', False))} "
                                f"coverage_ratio={float(part_val.get('coverage_ratio', float('nan'))):.3f} "
                                f"occ_uniform={int(occ_val.get('uniform_count', 0))} "
                                f"occ_bursty={int(occ_val.get('bursty_count', 0))} "
                                f"bursty_frac={float(occ_val.get('bursty_frac', float('nan'))):.3f}"
                            )
                        if dump_this and bool(args.partition_debug):
                            seg_dump = list(layout.get("band_partition_segments", []))[:10]
                            print(
                                f"[partition-debug] cond={cond_idx} trial={t} partition_ok={part_val['coverage_ok']} "
                                f"coverage_ratio={part_val['coverage_ratio']:.3f} first10={seg_dump}"
                            )
                        if dump_this and bool(args.occupancy_debug):
                            print(
                                f"[occupancy-debug] cond={cond_idx} trial={t} occ={occ_val} "
                                f"target_mode={occ} p_bursty={args.p_bursty:.3f}"
                            )
                        sm = WiFiSpatialModel(
                            _to_spatial_cfg(
                                layout,
                                seed=trial_seed + 7,
                                aci_model=str(args.aci_model),
                                aci_lut_path=args.aci_lut_path,
                            )
                        )
                        ap_summary = sm.summarize_aps(uwb_nodes_xy=[(0.0, 0.0), (float(args.dist_m), 0.0)])
                        row = _run_trial(
                            cfg_base=cfg_base,
                            wifi_mode="dense",
                            wifi_density=float(duty),
                            distance_m=float(args.dist_m),
                            uwb_channel=int(args.uwb_channel),
                            wifi_offset_mhz=float(args.wifi_offset_mhz),
                            trial_idx=t,
                            seed=trial_seed,
                            max_attempts=int(ma),
                            until_success=True,
                            max_trial_ms=400.0,
                            uwb_shots_per_session=1,
                            require_k_successes=1,
                            aggregation="median",
                            uwb_shot_gap_ms=0.5,
                            wifi_model="layout",
                            spatial_model=sm,
                            nb_lbt_slot_ms=float(std.nb_lbt_slot_ms),
                            nb_lbt_cca_slots=int(std.nb_lbt_cca_slots),
                            ssbd_cfg=ssbd_cfg,
                            ssbd_debug=False,
                            print_ssbd_trace=False,
                            nb_switch_cfg=nb_cfg,
                        )
                        ls = layout_distance_stats(layout, uwb_nodes_xy=[(0.0, 0.0), (float(args.dist_m), 0.0)])
                        row["duty_cycle"] = float(duty)
                        row["occupancy_mode"] = str(occ)
                        row["hopping"] = str(hop)
                        row["max_attempts_cfg"] = int(ma)
                        row["wifi_layout_stats_json"] = json.dumps(ls, separators=(",", ":"))
                        row["band_partition_summary_json"] = json.dumps(layout.get("band_partition_summary", {}), separators=(",", ":"))
                        row["partition_ok"] = bool(part_val.get("coverage_ok", False))
                        row["coverage_ratio"] = float(part_val.get("coverage_ratio", float("nan")))
                        row["partition_bw_hist_json"] = json.dumps(part_val.get("bw_hist", {}), separators=(",", ":"))
                        row["partition_edge_lo_mhz"] = float(part_val.get("f_lo_hz", float("nan"))) / 1e6
                        row["partition_edge_hi_mhz"] = float(part_val.get("f_hi_hz", float("nan"))) / 1e6
                        row["occ_counts_json"] = json.dumps(
                            {"uniform": int(occ_val.get("uniform_count", 0)), "bursty": int(occ_val.get("bursty_count", 0))},
                            separators=(",", ":"),
                        )
                        row["occ_bursty_frac"] = float(occ_val.get("bursty_frac", float("nan")))
                        row["duty_hat_mean_bursty"] = float(occ_val.get("duty_hat_mean_bursty", float("nan")))
                        row["duty_hat_p95_bursty"] = float(occ_val.get("duty_hat_p95_bursty", float("nan")))
                        row["n_ap"] = int(ap_summary.get("n_ap", 0))
                        row["wifi_ap_summary_json"] = json.dumps(ap_summary, separators=(",", ":"))
                        aci_stats = sm.aci_stats()
                        row["aci_model"] = str(aci_stats.get("aci_model", args.aci_model))
                        row["aci_lut_hit_rate"] = float(aci_stats.get("aci_lut_hit_rate", float("nan")))
                        row["aci_lut_missing_bws_json"] = json.dumps(
                            list(aci_stats.get("aci_lut_missing_bws", [])),
                            separators=(",", ":"),
                        )
                        row["aci_lut_missing_offsets_count"] = int(aci_stats.get("aci_lut_missing_offsets_count", 0))
                        row["generated_trial_id"] = str(layout.get("generated_trial_id", ""))
                        row["init_success"] = bool(np.isfinite(float(row.get("latency_to_conf_done_ms", np.nan))) and float(row.get("latency_to_conf_done_ms", np.nan)) >= 0.0)
                        row["range_success_given_init"] = bool(row["success"] and row["init_success"])
                        if sample_partition_json is None:
                            sample_partition_json = json.dumps(layout.get("band_partition_segments", [])[:10], indent=2)
                        if sample_aps_json is None:
                            sample_aps_json = json.dumps(layout.get("wifi_aps", [])[:5], indent=2)
                        rows.append(row)
                        succ += int(bool(row["success"]))
                        if dump_this:
                            print(
                                f"[health-aci] cond={cond_idx} trial={t} model={row['aci_model']} "
                                f"hit_rate={row['aci_lut_hit_rate']:.3f} "
                                f"miss_bw={row['aci_lut_missing_bws_json']} miss_off={row['aci_lut_missing_offsets_count']}"
                            )
                        if int(args.progress_every) > 0 and ((t + 1) % int(args.progress_every) == 0 or (t + 1) == int(args.trials)):
                            print(f"  - progress {t+1}/{args.trials} succ={succ}")

    df = pd.DataFrame(rows)
    trial_csv = out_dir / "trial_results.csv"
    df.to_csv(trial_csv, index=False)

    agg_rows = []
    for (occ, hop, ma, d), g in df.groupby(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"], dropna=False):
        n = int(len(g))
        succ = int(np.sum(g["success"] == True))
        p_l, p_u = _wilson_ci(succ, n)
        ok = g[g["success"] == True]
        agg_rows.append(
            {
                "occupancy_mode": occ,
                "hopping": hop,
                "max_attempts_cfg": int(ma),
                "duty_cycle": float(d),
                "n_trials": n,
                "success_count": succ,
                "P_init": float(np.mean(g["init_success"].astype(float))),
                "P_range_given_init": float(np.mean(g[g["init_success"] == True]["success"].astype(float))) if np.any(g["init_success"] == True) else float("nan"),
                "P_overall": float(np.mean(g["success"].astype(float))),
                "P_overall_wilson_lo": p_l,
                "P_overall_wilson_hi": p_u,
                "latency_to_success_mean_ms": float(pd.to_numeric(ok["latency_to_success_ms"], errors="coerce").mean()) if len(ok) else float("nan"),
                "latency_to_success_median_ms": float(pd.to_numeric(ok["latency_to_success_ms"], errors="coerce").median()) if len(ok) else float("nan"),
                "latency_to_success_p90_ms": float(np.nanpercentile(pd.to_numeric(ok["latency_to_success_ms"], errors="coerce"), 90)) if len(ok) else float("nan"),
                "latency_to_success_p95_ms": float(np.nanpercentile(pd.to_numeric(ok["latency_to_success_ms"], errors="coerce"), 95)) if len(ok) else float("nan"),
                "latency_all_median_ms": float(pd.to_numeric(g["time_spent_ms"], errors="coerce").median()),
                "latency_all_p95_ms": float(np.nanpercentile(pd.to_numeric(g["time_spent_ms"], errors="coerce"), 95)),
                "attempts_used_mean": float(pd.to_numeric(g["attempts_used"], errors="coerce").mean()),
                "attempts_used_median": float(pd.to_numeric(g["attempts_used"], errors="coerce").median()),
                "attempts_used_p95": float(np.nanpercentile(pd.to_numeric(g["attempts_used"], errors="coerce"), 95)),
                "aci_model": str(pd.Series(g["aci_model"]).dropna().iloc[0]) if "aci_model" in g.columns and len(g) else str(args.aci_model),
                "aci_lut_hit_rate_mean": float(pd.to_numeric(g["aci_lut_hit_rate"], errors="coerce").mean()) if "aci_lut_hit_rate" in g.columns else float("nan"),
                "aci_lut_missing_offsets_count_sum": int(np.nansum(pd.to_numeric(g["aci_lut_missing_offsets_count"], errors="coerce"))) if "aci_lut_missing_offsets_count" in g.columns else 0,
                "aci_lut_missing_bws_top": dict(pd.Series(g["aci_lut_missing_bws_json"]).value_counts().head(5)) if "aci_lut_missing_bws_json" in g.columns else {},
                "terminal_fail_reason_top": dict(pd.Series(g[g["success"] == False]["terminal_fail_reason"]).value_counts().head(5)),
                "nb_hopping_seq_sample": str(g["nb_channel_seq_first8_json"].iloc[0]) if len(g) else "[]",
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values(["occupancy_mode", "hopping", "max_attempts_cfg", "duty_cycle"])
    agg_csv = out_dir / "aggregate_results.csv"
    agg.to_csv(agg_csv, index=False)

    _plot(df, out_dir)
    _health_report(
        df,
        agg,
        out_dir,
        n_dump=int(args.dump_nb_hopping_seq),
        sample_partition_json=sample_partition_json,
        sample_aps_json=sample_aps_json,
    )

    print(f"saved trial csv: {trial_csv}")
    print(f"saved aggregate csv: {agg_csv}")
    print(f"saved plots+report under: {out_dir}")


if __name__ == "__main__":
    main()
