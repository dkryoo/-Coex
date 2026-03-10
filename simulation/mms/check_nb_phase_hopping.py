from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import (
        NbChannelSwitchConfig,
        build_nb_channel_allow_list,
        selected_nb_channel_for_phase,
    )
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.standard_params import get_default_params
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.full_stack_mms_demo import FullStackConfig
    from simulation.mms.mms_latency_sweep import _run_trial
    from simulation.mms.nb_channel_switching import (
        NbChannelSwitchConfig,
        build_nb_channel_allow_list,
        selected_nb_channel_for_phase,
    )
    from simulation.mms.nb_ssbd_access import SsbdConfig
    from simulation.mms.standard_params import get_default_params


def _mk_ssbd_cfg() -> SsbdConfig:
    std = get_default_params("802154ab")
    return SsbdConfig(
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


def _run_case(
    enable_switching: bool,
    allow_list: tuple[int, ...],
    sim_seed: int,
    mac_seed: int,
    include_report_phase: bool = False,
) -> dict:
    cfg = FullStackConfig(
        distance_m=20.0,
        nb_channel=1,
        uwb_channel=5,
        nb_center_override_hz=6.489e9,
        seed=int(sim_seed),
        toa_calibration_override=0.0,
        first_path=False,
    )
    nb_cfg = NbChannelSwitchConfig(
        enable_switching=bool(enable_switching),
        allow_list=tuple(int(x) for x in allow_list),
        mms_prng_seed=int(mac_seed) & 0xFF,
        channel_switching_field=1 if enable_switching else 0,
        nb_channel_spacing_mhz=2.0,
        mms_nb_init_channel=2,
    )
    row = _run_trial(
        cfg_base=cfg,
        wifi_mode="off",
        wifi_model="occupancy",
        spatial_model=None,
        wifi_density=0.0,
        distance_m=20.0,
        uwb_channel=5,
        wifi_offset_mhz=None,
        trial_idx=0,
        seed=int(sim_seed),
        max_attempts=4,
        until_success=True,
        max_trial_ms=200.0,
        uwb_shots_per_session=1,
        require_k_successes=2,  # force retries so per-attempt channel use is observable
        aggregation="median",
        uwb_shot_gap_ms=0.5,
        ssbd_cfg=_mk_ssbd_cfg(),
        nb_lbt_slot_ms=0.02,
        nb_lbt_cca_slots=4,
        ssbd_debug=False,
        print_ssbd_trace=False,
        nb_switch_cfg=nb_cfg,
        enable_init_scan_model=False,
        enable_report_phase_model=bool(include_report_phase),
        initiator_report_request=bool(include_report_phase),
        responder_report_request=False,
        mms1st_report_nslots=1,
        mms2nd_report_nslots=0,
        assume_oob_report_on_missing=True,
    )
    return row


def _fmt_seq(row: dict, key: str) -> str:
    try:
        v = json.loads(str(row.get(key, "[]")))
    except Exception:
        v = []
    return str(v)


def _parse_allow_expr(s: str) -> tuple[int, ...]:
    out: list[int] = []
    for tok in [x.strip() for x in str(s).split(",") if x.strip()]:
        if ".." in tok:
            a, b = tok.split("..", 1)
            lo, hi = int(a), int(b)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(tok))
    return tuple(sorted(set(int(x) for x in out)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check phase-specific NB hopping behavior (H1/H2 evidence)")
    ap.add_argument("--sim-seed", type=int, default=None, help="Simulator RNG seed (fading/noise/retry flow).")
    ap.add_argument("--seed", type=int, default=None, help="Deprecated alias of --sim-seed.")
    ap.add_argument("--mac-seed", type=int, default=0, help="macMmsPrngSeed for NB channel switching (0..255).")
    ap.add_argument("--allow-list-mode", choices=["explicit", "small_test", "all_250"], default="explicit")
    ap.add_argument("--allow-list", type=str, default="0,1,2,3", help='supports range syntax like "0..249"')
    ap.add_argument("--derive-allow-list", type=int, choices=[0, 1], default=0)
    ap.add_argument("--nb-channel-start", type=int, default=0)
    ap.add_argument("--nb-channel-step", type=int, default=1)
    ap.add_argument("--nb-channel-step-code", type=int, default=None)
    ap.add_argument("--nb-channel-bitmask-hex", type=str, default=None)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--include-report-phase", type=int, choices=[0, 1], default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    sim_seed = int(args.sim_seed) if args.sim_seed is not None else int(args.seed) if args.seed is not None else 20260307

    if int(args.mac_seed) < 0 or int(args.mac_seed) > 255:
        raise ValueError("--mac-seed must be in [0,255]")

    if str(args.allow_list_mode) == "small_test":
        allow_in = (0, 1, 2, 3)
    elif str(args.allow_list_mode) == "all_250":
        allow_in = tuple(range(250))
    else:
        allow_in = _parse_allow_expr(args.allow_list)
    if bool(int(args.derive_allow_list)):
        derived = set(
            int(x)
            for x in build_nb_channel_allow_list(
                explicit_allow_list=None,
                nb_channel_start=int(args.nb_channel_start),
                nb_channel_step=int(args.nb_channel_step),
                nb_channel_step_code=args.nb_channel_step_code,
                nb_channel_bitmask_hex=args.nb_channel_bitmask_hex,
            )
        )
        allow = tuple(sorted(set(int(x) for x in allow_in if int(x) in derived)))
    else:
        allow = tuple(sorted(set(int(x) for x in allow_in if 0 <= int(x) <= 249)))
    if not allow:
        raise ValueError("effective allow list is empty after parsing/derivation")

    print(
        "[effective-allow-list] "
        f"len={len(allow)} min={min(allow)} max={max(allow)} "
        f"head={list(allow[:16])}"
    )
    off = _run_case(
        False,
        allow,
        sim_seed=sim_seed,
        mac_seed=int(args.mac_seed),
        include_report_phase=bool(int(args.include_report_phase)),
    )
    on = _run_case(
        True,
        allow,
        sim_seed=sim_seed,
        mac_seed=int(args.mac_seed),
        include_report_phase=bool(int(args.include_report_phase)),
    )
    rows = [("off", off), ("on", on)]

    print(
        "[phase-rules] "
        f"init=mmsNbInitChannel({int(off.get('mmsNbInitChannel', 2))}), "
        f"ctrl/report=({'allow-list lowest' if not bool(on.get('nb_channel_switching_enabled', False)) else 'allow-list + AES-PRNG'})"
    )
    print(
        "[phase-inputs] "
        f"switching_field_off={off.get('nb_channel_switching_enabled', False)}, "
        f"switching_field_on={on.get('nb_channel_switching_enabled', False)}, "
        f"allow_list={allow}, macMmsPrngSeed={int(args.mac_seed)}"
    )

    print("hopping,phase,unique_count,example_seq")
    for mode, r in rows:
        print(f"{mode},init,{int(r.get('nb_ch_unique_count_init', 0))},{_fmt_seq(r, 'nb_ch_init_seq_first8_json')}")
        print(f"{mode},ctrl,{int(r.get('nb_ch_unique_count_ctrl', 0))},{_fmt_seq(r, 'nb_ch_ctrl_seq_first8_json')}")
        print(f"{mode},report,{int(r.get('nb_ch_unique_count_report', 0))},{_fmt_seq(r, 'nb_ch_report_seq_first8_json')}")
        if int(r.get("nb_ch_unique_count_report", 0)) == 0:
            print(f"[note:{mode}] report phase sequence empty (no report phase generated in this scenario).")

    # Long-block selector coverage experiment (H1/H2/H3 triage).
    n_blocks = max(1, int(args.n_blocks))
    cfg_off = NbChannelSwitchConfig(
        enable_switching=False,
        allow_list=tuple(allow),
        mms_prng_seed=int(args.mac_seed),
        channel_switching_field=0,
        nb_channel_spacing_mhz=2.0,
        mms_nb_init_channel=2,
    )
    cfg_on = NbChannelSwitchConfig(
        enable_switching=True,
        allow_list=tuple(allow),
        mms_prng_seed=int(args.mac_seed),
        channel_switching_field=1,
        nb_channel_spacing_mhz=2.0,
        mms_nb_init_channel=2,
    )
    seq_on_ctrl = [int(selected_nb_channel_for_phase(cfg_on, "ctrl", k)) for k in range(n_blocks)]
    seq_on_report = [int(selected_nb_channel_for_phase(cfg_on, "report", k)) for k in range(n_blocks)]
    seq_off_ctrl = [int(selected_nb_channel_for_phase(cfg_off, "ctrl", k)) for k in range(n_blocks)]
    uniq_on = sorted(set(seq_on_ctrl))
    hist_on = {int(ch): int(seq_on_ctrl.count(ch)) for ch in uniq_on}
    cov = float(len(uniq_on) / max(1, len(allow)))
    print(
        "[selector-coverage] "
        f"n_blocks={n_blocks} unique_on={len(uniq_on)}/{len(allow)} coverage={cov:.3f} "
        f"hist_on={hist_on}"
    )
    print(f"[selector-first16] on_ctrl={seq_on_ctrl[:16]} on_report={seq_on_report[:16]} off_ctrl={seq_off_ctrl[:16]}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "rules": {
                "init": f"mmsNbInitChannel={int(off.get('mmsNbInitChannel', 2))}",
                "ctrl_report_off": "channel switching disabled -> lowest(allow_list)",
                "ctrl_report_on": "channel switching enabled -> allow_list[PrngValue mod n]",
            },
            "inputs": {
                "sim_seed": int(sim_seed),
                "mac_seed": int(args.mac_seed),
                "allow_list": [int(x) for x in allow],
                "allow_list_mode": str(args.allow_list_mode),
                "n_blocks": int(n_blocks),
            },
            "off": off,
            "on": on,
            "table": [
                {"hopping": "off", "phase": "init", "unique_count": int(off.get("nb_ch_unique_count_init", 0)), "seq": _fmt_seq(off, "nb_ch_init_seq_first8_json")},
                {"hopping": "off", "phase": "ctrl", "unique_count": int(off.get("nb_ch_unique_count_ctrl", 0)), "seq": _fmt_seq(off, "nb_ch_ctrl_seq_first8_json")},
                {"hopping": "off", "phase": "report", "unique_count": int(off.get("nb_ch_unique_count_report", 0)), "seq": _fmt_seq(off, "nb_ch_report_seq_first8_json")},
                {"hopping": "on", "phase": "init", "unique_count": int(on.get("nb_ch_unique_count_init", 0)), "seq": _fmt_seq(on, "nb_ch_init_seq_first8_json")},
                {"hopping": "on", "phase": "ctrl", "unique_count": int(on.get("nb_ch_unique_count_ctrl", 0)), "seq": _fmt_seq(on, "nb_ch_ctrl_seq_first8_json")},
                {"hopping": "on", "phase": "report", "unique_count": int(on.get("nb_ch_unique_count_report", 0)), "seq": _fmt_seq(on, "nb_ch_report_seq_first8_json")},
            ],
            "selector_coverage": {
                "n_blocks": int(n_blocks),
                "on_ctrl_first16": seq_on_ctrl[:16],
                "on_report_first16": seq_on_report[:16],
                "off_ctrl_first16": seq_off_ctrl[:16],
                "unique_on_ctrl": [int(x) for x in uniq_on],
                "hist_on_ctrl": hist_on,
                "coverage_ratio_allowlist": float(cov),
            },
        }
        p.write_text(json.dumps(data, indent=2))
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
