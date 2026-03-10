"""
MMS latency sweep tool.

Examples:
1) Wi-Fi OFF distance sweep
   python simulation/mms/mms_latency_sweep.py --wifi-mode off --dist-list "5,10,20,30" --trials 10

2) Wi-Fi DENSE distance sweep
   python simulation/mms/mms_latency_sweep.py --wifi-mode dense --dist-start 5 --dist-stop 30 --dist-step 5 --trials 10

3) Wi-Fi DENSE + Wi-Fi offset sweep
   python simulation/mms/mms_latency_sweep.py --wifi-mode dense --dist-list "10,20" --wifi-offsets-mhz "[-80,-40,0,40,80]" --trials 8

4) Wi-Fi DENSE + offsets + UWB channels
   python simulation/mms/mms_latency_sweep.py --wifi-mode dense --dist-list "10,20" --uwb-channels "5,7,8,10" --wifi-offsets-mhz "[-80,-40,0,40,80]" --trials 8
"""

from __future__ import annotations

import argparse
import csv
import math
import json
import copy
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from packet.narrowband_compact_frames import (
        AdvConf,
        AdvPoll,
        AdvResp,
        CFID_ADV_CONF,
        CFID_ADV_RESP,
        SmidTlv,
        encode_adv_conf,
        encode_adv_poll,
        encode_adv_resp,
    )
    from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
    from UWB.mms_uwb_packet_mode import MmsUwbConfig, RSTU_S, build_mms_uwb_fragments
    from simulation.mms.full_stack_mms_demo import (
        FullStackConfig,
        run_full_stack_case,
        uwb_center_freq_hz,
        wifi6_center_freq_hz,
    )
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.performance import simulate_mms_performance
    from simulation.mms.wifi_spatial_model import WiFiSpatialConfig, WiFiSpatialModel, load_wifi_layout
    from simulation.mms.nb_ssbd_access import SsbdConfig, run_ssbd_access
    from simulation.mms.nb_channel_switching import (
        NbChannelSwitchConfig,
        build_nb_channel_allow_list,
        selected_nb_channel_for_phase,
        selected_nb_channel_for_block,
    )
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from packet.narrowband_compact_frames import (
        AdvConf,
        AdvPoll,
        AdvResp,
        CFID_ADV_CONF,
        CFID_ADV_RESP,
        SmidTlv,
        encode_adv_conf,
        encode_adv_poll,
        encode_adv_resp,
    )
    from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
    from UWB.mms_uwb_packet_mode import MmsUwbConfig, RSTU_S, build_mms_uwb_fragments
    from simulation.mms.full_stack_mms_demo import (
        FullStackConfig,
        run_full_stack_case,
        uwb_center_freq_hz,
        wifi6_center_freq_hz,
    )
    from simulation.mms.standard_params import get_default_params
    from simulation.mms.performance import simulate_mms_performance
    from simulation.mms.wifi_spatial_model import WiFiSpatialConfig, WiFiSpatialModel, load_wifi_layout
    from simulation.mms.nb_ssbd_access import SsbdConfig, run_ssbd_access
    from simulation.mms.nb_channel_switching import (
        NbChannelSwitchConfig,
        build_nb_channel_allow_list,
        selected_nb_channel_for_phase,
        selected_nb_channel_for_block,
    )


def _parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    cleaned = text.strip().replace("[", "").replace("]", "")
    out: list[float] = []
    for tok in cleaned.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def _parse_int_list(text: str | None) -> list[int]:
    if text is None:
        return []
    cleaned = text.strip().replace("[", "").replace("]", "")
    out: list[int] = []
    for tok in cleaned.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _scaled_layout_cfg(
    layout_cfg: WiFiSpatialConfig | None,
    *,
    seed: int,
    duty_scale: float,
) -> WiFiSpatialConfig | None:
    if layout_cfg is None:
        return None
    lc = copy.deepcopy(layout_cfg)
    lc.seed = int(seed)
    scale = float(np.clip(duty_scale, 0.0, 1.0))
    lc.duty_cycle = float(np.clip(lc.duty_cycle * scale, 0.0, 1.0))
    if lc.aps:
        for ap in lc.aps:
            base = float(ap.get("duty_cycle", ap.get("traffic_load", layout_cfg.duty_cycle)))
            d = float(np.clip(base * scale, 0.0, 1.0))
            ap["duty_cycle"] = d
            ap["traffic_load"] = d
    return lc


def _distances_from_args(args: argparse.Namespace) -> list[float]:
    if args.dist_list:
        return _parse_float_list(args.dist_list)
    if args.dist_step <= 0:
        raise ValueError("--dist-step must be > 0")
    vals = np.arange(args.dist_start, args.dist_stop + 1e-12, args.dist_step, dtype=float)
    return [float(v) for v in vals]


def _nearest_wifi_channel_from_fc(fc_hz: float) -> int:
    # 6 GHz mapping: fc_MHz = 5950 + 5*ch
    ch = int(round((fc_hz / 1e6 - 5950.0) / 5.0))
    return max(1, ch)


class DenseWiFiOccupancy:
    """Simple TXOP-burst occupancy model for channel busy/defer timing."""

    def __init__(self, duty_cycle: float, txop_ms: float, gap_jitter_ms: float, rng: np.random.Generator):
        self.rng = rng
        self.duty_cycle = float(np.clip(duty_cycle, 0.01, 0.99))
        self.txop_ms = float(max(0.05, txop_ms))
        # mean gap selected to satisfy duty = txop / (txop + gap)
        self.gap_mean_ms = self.txop_ms * (1.0 - self.duty_cycle) / self.duty_cycle
        self.gap_jitter_ms = float(max(0.0, gap_jitter_ms))
        self.win_start = 0.0
        self.win_end = self.txop_ms

    def _advance_until(self, t_ms: float) -> None:
        while t_ms >= self.win_end:
            gap = self.gap_mean_ms + self.rng.uniform(-self.gap_jitter_ms, self.gap_jitter_ms)
            gap = max(0.0, gap)
            self.win_start = self.win_end + gap
            self.win_end = self.win_start + self.txop_ms

    def defer_ms(self, t_ms: float) -> float:
        self._advance_until(t_ms)
        if self.win_start <= t_ms < self.win_end:
            return float(self.win_end - t_ms)
        return 0.0


def _nb_selectivity_atten_db(offset_mhz: float, nb_bw_mhz: float = 2.0) -> float:
    """Simple NB in-band selectivity attenuation for sensing path."""
    x = abs(float(offset_mhz))
    edge = max(0.1, float(nb_bw_mhz) * 0.5)
    if x <= edge:
        return 0.0
    # 2nd-order roll-off-like attenuation with ceiling.
    r = max(0.0, (x - edge) / edge)
    att = 10.0 * math.log10(1.0 + r * r)
    return float(min(80.0, max(0.0, att)))


def _occupancy_interference_dbm(
    *,
    wifi_density: float,
    wifi_offset_mhz: float,
    distance_m: float,
) -> float:
    # Coarse RSSI model for occupancy mode.
    p_base = -40.0 - 20.0 * math.log10(max(distance_m, 1.0) / 2.0)
    p_load = 10.0 * math.log10(max(1e-3, min(1.0, wifi_density)))
    p_sel = -_nb_selectivity_atten_db(offset_mhz=wifi_offset_mhz, nb_bw_mhz=2.0)
    return float(p_base + p_load + p_sel)


def _approx_uwb_rx_signal_dbm(
    *,
    distance_m: float,
    fc_hz: float,
    tx_eirp_dbw: float,
    pathloss_exp: float = 2.0,
) -> float:
    # Coarse large-scale estimate used only for layout-side extra gating.
    # Pr(dBW) = EIRP(dBW) - PL(dB), with PL(d) anchored at FSPL(1m).
    c0 = 299_792_458.0
    wavelength_m = c0 / max(float(fc_hz), 1.0)
    fspl_1m_db = 20.0 * math.log10(4.0 * math.pi / max(wavelength_m, 1e-12))
    d = max(float(distance_m), 1e-3)
    pl_db = fspl_1m_db + 10.0 * float(pathloss_exp) * math.log10(max(d, 1.0))
    pr_dbw = float(tx_eirp_dbw) - pl_db
    return float(pr_dbw + 30.0)


def _build_nb_frames() -> tuple[bytes, bytes, bytes]:
    poll = AdvPoll(
        init_slot_dur=10,
        cap_dur=8,
        supported_mod_modes=0x03,
        smid_tlvs=[SmidTlv(tag=CFID_ADV_RESP, values=b"\x00"), SmidTlv(tag=CFID_ADV_CONF, values=b"\x00")],
    )
    resp = AdvResp(
        rpa_hash=b"\xAA\xBB\xCC",
        message_id=0x42,
        nb_full_channel_map=b"\x01\x02\x03\x04\x05\x06",
        mgmt_phy_cfg=0x21,
        mgmt_mac_cfg=b"\x10\x11\x12\x13\x14\x15\x16",
        ranging_phy_cfg=b"\x20\x21\x22\x23",
        mms_num_frags=2,
    )
    conf = AdvConf(
        rpa_hash=b"\x11\x22\x33",
        message_id=0x42,
        responder_addr=b"\xAA\xBB\xCC",
        sor_time_offset=b"\x01\x02\x03",
    )
    return encode_adv_poll(poll), encode_adv_resp(resp), encode_adv_conf(conf)


def _estimate_nb_frame_ms(cfg: FullStackConfig, frame_bytes: bytes) -> float:
    tx = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    bits = np.unpackbits(np.frombuffer(frame_bytes, dtype=np.uint8), bitorder="little").astype(int)
    wf, _, _ = tx.build_tx_waveform(psdu_bits=bits, tx_eirp_db=cfg.nb_eirp_dbw, regulatory_profile="unlicensed_6g_lpi_ap")
    return float(len(wf) / tx.fs * 1000.0)


def _nb_timing_breakdown_ms(cfg: FullStackConfig, nb_poll_attempts: int) -> dict:
    poll_b, resp_b, conf_b = _build_nb_frames()
    t_adv = _estimate_nb_frame_ms(cfg, poll_b)
    t_poll = 0.0  # compact handshake currently models ADV-POLL as first NB ADV transmission
    t_resp = _estimate_nb_frame_ms(cfg, resp_b)
    t_conf = _estimate_nb_frame_ms(cfg, conf_b)
    ifs_ms = 0.15
    turnaround_ms = 0.0
    one_exchange = t_adv + t_poll + t_resp + t_conf + 3.0 * ifs_ms + turnaround_ms
    total = max(1, int(nb_poll_attempts)) * one_exchange
    return {
        "t_nb_adv_ms": t_adv,
        "t_nb_poll_ms": t_poll,
        "t_nb_resp_ms": t_resp,
        "t_nb_conf_ms": t_conf,
        "t_nb_ifs_total_ms": 3.0 * ifs_ms,
        "t_nb_turnaround_ms": turnaround_ms,
        "nb_one_exchange_ms": one_exchange,
        "nb_total_ms": total,
        "nb_poll_attempts": int(max(1, int(nb_poll_attempts))),
        "params_source": "NarrowBand/TX_NarrowBand.py (waveform duration) + simulation/mms/mms_latency_sweep.py (IFS=0.15ms)",
    }


def _uwb_timing_breakdown_ms() -> dict:
    cfg = MmsUwbConfig(phy_uwb_mms_rsf_number_frags=2, phy_uwb_mms_rif_number_frags=1)
    phase_start = 100_000
    i_frags = build_mms_uwb_fragments("initiator", phase_start, cfg)
    r_frags = build_mms_uwb_fragments("responder", phase_start, cfg)
    end_i = max(f.start_rstu + f.duration_rstu for f in i_frags)
    end_r = max(f.start_rstu + f.duration_rstu for f in r_frags)
    span_rstu = max(end_i, end_r) - phase_start
    responder_reply_delay_rstu = min(f.start_rstu for f in r_frags) - phase_start
    return {
        "t_uwb_airtime_ms": float(span_rstu * RSTU_S * 1000.0 + 0.1),
        "t_uwb_reply_delay_ms": float(responder_reply_delay_rstu * RSTU_S * 1000.0),
        "uwb_params_source": "UWB/mms_uwb_packet_mode.py (MmsUwbConfig, build_mms_uwb_fragments, RSTU_S)",
    }


def _summary_stats(vals: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 95))


def _one_time_toa_calibration_samples(cfg_base: FullStackConfig) -> float:
    """Calibrate ToA offset once and reuse it across all trials/attempts."""
    res = simulate_mms_performance(
        distances_m=(float(cfg_base.distance_m),),
        n_trials=1,
        rif_payload_bits=64,
        fc_hz=uwb_center_freq_hz(int(cfg_base.uwb_channel)),
        uwb_fs_hz=499.2e6,
        tx_eirp_dbw=float(cfg_base.nb_eirp_dbw),
        nf_db=float(cfg_base.nf_db),
        temperature_k=float(cfg_base.temperature_k),
        seed=int(cfg_base.seed + 7070),
        detector_mode="first_path",
        baseline_sanity_mode=True,
        auto_calibrate=True,
        enable_crc=True,
        debug_first_trial=False,
        save_psd=False,
    )[0]
    return float(res.get("toa_calibration_samples", 0.0))


def _run_trial(
    cfg_base: FullStackConfig,
    wifi_mode: str,
    wifi_density: float,
    distance_m: float,
    uwb_channel: int,
    wifi_offset_mhz: float | None,
    trial_idx: int,
    seed: int,
    max_attempts: int,
    until_success: bool,
    max_trial_ms: float,
    uwb_shots_per_session: int,
    require_k_successes: int,
    aggregation: str,
    uwb_shot_gap_ms: float,
    wifi_model: str,
    spatial_model: WiFiSpatialModel | None,
    nb_lbt_slot_ms: float,
    nb_lbt_cca_slots: int,
    ssbd_cfg: SsbdConfig,
    ssbd_debug: bool,
    print_ssbd_trace: bool,
    nb_switch_cfg: NbChannelSwitchConfig,
    enable_init_scan_model: bool = False,
    scan_interval_ms: float = 100.0,
    scan_window_ms: float = 20.0,
    adv_interval_ms: float = 100.0,
    adv_tx_duration_ms: float = 2.0,
    random_start_phase: bool = True,
    enable_report_phase_model: bool = False,
    initiator_report_request: bool = False,
    responder_report_request: bool = False,
    mms1st_report_nslots: int = 1,
    mms2nd_report_nslots: int = 0,
    assume_oob_report_on_missing: bool = True,
) -> dict:
    t_run0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    nb_center_hz = float(cfg_base.nb_center_override_hz if cfg_base.nb_center_override_hz is not None else 6.489e9)
    nb_center_base_hz = float(nb_center_hz)
    if wifi_mode == "off":
        wifi_fc_target_hz = wifi6_center_freq_hz(cfg_base.wifi_channel)
        wifi_on = False
    else:
        wifi_fc_target_hz = nb_center_hz + float(wifi_offset_mhz or 0.0) * 1e6
        wifi_on = True
    wifi_ch = _nearest_wifi_channel_from_fc(wifi_fc_target_hz)
    wifi_fc_hz = wifi6_center_freq_hz(wifi_ch)
    uwb_fc_hz = uwb_center_freq_hz(uwb_channel)
    uwb_sig_ref_dbm = _approx_uwb_rx_signal_dbm(
        distance_m=float(distance_m),
        fc_hz=float(uwb_fc_hz),
        tx_eirp_dbw=float(cfg_base.nb_eirp_dbw),
        pathloss_exp=2.0,
    )
    # In layout mode, interference is injected via spatial_model (time-varying duty per AP).
    # Disable always-on waveform interferer inside run_full_stack_case to avoid double counting
    # and to preserve duty-cycle dependence.
    wifi_on_fullstack = bool(wifi_on and wifi_model != "layout")

    occ = DenseWiFiOccupancy(
        duty_cycle=wifi_density,
        txop_ms=0.9,
        gap_jitter_ms=0.3,
        rng=rng,
    ) if wifi_mode == "dense" else None

    t_ms = 0.0
    conf_done_ms = float("nan")
    latency_to_success_ms = float("nan")
    nb_handshake_ms_acc = 0.0
    uwb_ranging_ms_acc = 0.0
    waiting_ms_acc = 0.0
    retries_nb = 0
    retries_uwb = 0
    nb_lbt_busy_events = 0
    nb_lbt_wait_ms = 0.0
    nb_backoff_slots = 0
    nb_ssbd_total_deferral_ms = 0.0
    nb_ssbd_total_cca_ms = 0.0
    nb_ssbd_nb_count = 0
    nb_ssbd_bf_final = 0
    nb_ssbd_cca_count = 0
    nb_ssbd_busy_count = 0
    nb_access_fail = False
    last_ssbd_reason = "idle"
    prev_ssbd_terminating_bf = int(ssbd_cfg.mac_ssbd_min_bf)
    ssbd_attempt_summaries: list[dict] = []
    selected_channels: list[int] = []
    init_channels: list[int] = []
    ctrl_channels: list[int] = []
    report_channels: list[int] = []
    wifi_cca_busy_events = 0
    wifi_cca_wait_ms = 0.0
    wifi_backoff_slots = 0
    wifi_tx_occupancy_time_ms = 0.0
    attempts_used = 0
    success = False
    last_res: dict | None = None
    shot_range_estimates_m: list[float] = []
    last_attempt_root_reason = "none"
    last_attempt_stage = "none"
    count_nb_ssbd_timeout = 0
    count_nb_control_decode_fail = 0
    count_uwb_frame_fail_snr = 0
    count_uwb_frame_fail_interference = 0
    wifi_busy_frac_nb = float("nan")
    wifi_busy_frac_uwb = float("nan")
    wifi_realized_duty_mean = float("nan")
    wifi_realized_duty_min = float("nan")
    wifi_realized_duty_max = float("nan")

    nb_bd_last: dict | None = None
    uwb_bd = _uwb_timing_breakdown_ms()
    events: list[dict] = []
    ranging_block_index = -1
    # Latency breakdown timestamps (simulation-time axis, ms).
    t0_trial_start_ms = 0.0
    t_init_done_ms = 0.0  # stack init/prep not explicitly modeled; treated as t0
    t_scan_start_ms = float("nan")  # adv/scan model not explicitly implemented in this simulator path
    t_adv_start_ms = float("nan")  # adv/scan model not explicitly implemented
    t_adv_rx_detected_ms = float("nan")  # adv/scan model not explicitly implemented
    t_cca_start_ms = float("nan")
    t_cca_done_ms = float("nan")
    t_backoff_done_ms = float("nan")
    t_uwb_tx_start_ms = float("nan")
    t_uwb_rx_done_ms = float("nan")
    t_toa_est_start_ms = float("nan")  # inside PHY/performance model; not exposed as simulation-time timestamp
    t_toa_est_done_ms = float("nan")  # inside PHY/performance model; not exposed as simulation-time timestamp
    t_aggregation_done_ms = float("nan")
    t_report_start_ms = float("nan")
    t_report1_tx_ms = float("nan")
    t_report1_rx_done_ms = float("nan")
    t_report2_tx_ms = float("nan")
    t_report2_rx_done_ms = float("nan")
    init_scan_wait_ms = 0.0
    advertising_wait_ms = 0.0
    report_phase_ms = 0.0
    nb_report_txrx_ms = 0.0
    nb_channel_match = True
    report_enabled = bool(enable_report_phase_model)
    report_required_r2i = bool(initiator_report_request)
    report_required_i2r = bool(responder_report_request)
    report_tx_attempted_r2i = False
    report_tx_attempted_i2r = False
    report_rx_ok_r2i = False
    report_rx_ok_i2r = False
    report_fail_reason = "none"

    def _first_detection_wait_ms() -> tuple[float, float, float, float]:
        if not bool(enable_init_scan_model):
            return 0.0, float("nan"), float("nan"), float("nan")
        si = max(1e-6, float(scan_interval_ms))
        sw = float(np.clip(scan_window_ms, 1e-6, si))
        ai = max(1e-6, float(adv_interval_ms))
        ad = max(1e-6, float(adv_tx_duration_ms))
        sp = float(rng.uniform(0.0, si)) if bool(random_start_phase) else 0.0
        ap = float(rng.uniform(0.0, ai)) if bool(random_start_phase) else 0.0
        max_n = 50000
        for n in range(max_n):
            a0 = ap + n * ai
            a1 = a0 + ad
            m0 = max(0, int(math.floor((a0 - sp - sw) / si)) - 2)
            m1 = m0 + 8
            for m in range(m0, m1 + 1):
                s0 = sp + m * si
                s1 = s0 + sw
                if s1 >= a0 and s0 <= a1:
                    td = max(a0, s0)
                    return float(max(0.0, td)), float(sp), float(a0), float(td)
        td = float(ap + max_n * ai)
        return td, float(sp), float(ap), float(td)

    if bool(enable_init_scan_model):
        wait_ms, sp0, a0, td0 = _first_detection_wait_ms()
        t_scan_start_ms = 0.0
        t_adv_start_ms = float(a0)
        t_adv_rx_detected_ms = float(td0)
        init_scan_wait_ms = float(wait_ms)
        advertising_wait_ms = float(max(0.0, td0 - a0))
        if wait_ms > 0.0:
            events.append({"start_ms": 0.0, "end_ms": float(wait_ms), "label": "init_scan_wait"})
        t_ms += float(wait_ms)
        waiting_ms_acc += float(wait_ms)

    attempt = 0
    while attempt < max_attempts and t_ms <= max_trial_ms:
        attempt += 1
        attempts_used = attempt
        # RangingBlockIndex policy:
        # In this simulator, each session-level ranging attempt is treated as one ranging block.
        # This keeps initiator/responder channel selection deterministic for retries.
        # (IEEE P802.15.4ab/D03 10.39.8.4.3 uses RangingBlockIndex as PRNG data input.)
        ranging_block_index = attempt - 1
        # Phase split:
        # - init phase must use mmsNbInitChannel
        # - control/report phases use allow-list + switching policy.
        init_nb_channel = int(selected_nb_channel_for_phase(nb_switch_cfg, "init", int(ranging_block_index)))
        ctrl_nb_channel = int(selected_nb_channel_for_phase(nb_switch_cfg, "ctrl", int(ranging_block_index)))
        # Check initiator/responder deterministic agreement on control/report channel sequence.
        ch_i = int(selected_nb_channel_for_phase(nb_switch_cfg, "ctrl", int(ranging_block_index)))
        ch_r = int(selected_nb_channel_for_phase(nb_switch_cfg, "ctrl", int(ranging_block_index)))
        if ch_i != ch_r:
            raise RuntimeError("NB channel selection mismatch between initiator/responder")
        nb_channel_match = bool(nb_channel_match and (ch_i == ch_r))

        selected_channels.append(int(ctrl_nb_channel))
        init_channels.append(int(init_nb_channel))
        nb_center_hz_init = nb_center_base_hz + float(init_nb_channel) * float(nb_switch_cfg.nb_channel_spacing_mhz) * 1e6
        nb_center_hz_ctrl = nb_center_base_hz + float(ctrl_nb_channel) * float(nb_switch_cfg.nb_channel_spacing_mhz) * 1e6
        nb_center_hz = float(nb_center_hz_ctrl)

        def _sense_nb_inband_dbm(t_probe_ms: float) -> float:
            off = float(wifi_offset_mhz or 0.0) - float(init_nb_channel) * float(nb_switch_cfg.nb_channel_spacing_mhz)
            if wifi_mode != "dense":
                return -200.0
            if spatial_model is not None:
                # Layout/spatial model already supports receiver-center/band aware in-band estimate.
                if wifi_model == "layout":
                    return float(
                        spatial_model.estimate_interference_dbm(
                            t_ms=t_probe_ms,
                            rx_xy=(0.0, 0.0),
                            rx_center_hz=float(nb_center_hz_init),
                            rx_bw_hz=2.0e6,
                        )
                    )
                p_wide = float(spatial_model.estimate_interference_dbm(t_probe_ms, rx_xy=(0.0, 0.0)))
                return float(p_wide - _nb_selectivity_atten_db(off, nb_bw_mhz=2.0))
            return _occupancy_interference_dbm(
                wifi_density=float(wifi_density),
                wifi_offset_mhz=off,
                distance_m=float(distance_m),
            )

        prev_bf_before = int(prev_ssbd_terminating_bf)
        if not np.isfinite(t_cca_start_ms):
            t_cca_start_ms = float(t_ms)
        ssbd_res = run_ssbd_access(
            t_start_ms=float(t_ms),
            sense_inband_dbm_fn=_sense_nb_inband_dbm,
            cfg=ssbd_cfg,
            rng=rng,
            is_retransmission_attempt=attempt > 1,
            prev_terminating_bf=int(prev_bf_before),
            debug=bool(ssbd_debug and trial_idx == 0),
        )
        nb_ssbd_total_deferral_ms += float(ssbd_res.total_deferral_ms)
        nb_ssbd_total_cca_ms += float(ssbd_res.total_cca_ms)
        nb_ssbd_cca_count += int(ssbd_res.cca_count)
        nb_ssbd_nb_count = max(nb_ssbd_nb_count, int(ssbd_res.nb_count))
        nb_ssbd_bf_final = int(ssbd_res.bf_final)
        prev_ssbd_terminating_bf = int(ssbd_res.bf_final)
        last_ssbd_reason = str(ssbd_res.reason)
        ssbd_attempt_summaries.append(
            {
                "attempt": int(attempt),
                "cca_count": int(ssbd_res.cca_count),
                "total_deferral_ms": float(ssbd_res.total_deferral_ms),
                "final_NB": int(ssbd_res.nb_count),
                "final_BF": int(ssbd_res.bf_final),
                "bf_init": int(ssbd_res.bf_init),
                "terminated_by": str(ssbd_res.reason),
            }
        )
        if bool((ssbd_debug or print_ssbd_trace) and trial_idx == 0):
            print(
                "[nb_ssbd_attempt] "
                f"attempt={attempt} bf_init={ssbd_res.bf_init} prev_term_bf={prev_bf_before} "
                f"cca_count={ssbd_res.cca_count} final_nb={ssbd_res.nb_count} "
                f"final_bf={ssbd_res.bf_final} terminated_by={ssbd_res.reason}"
            )
        for s in ssbd_res.trace:
            d_ms = float(s.get("deferral_ms", 0.0))
            if str(s.get("decision", "")) == "busy":
                nb_ssbd_busy_count += 1
            if d_ms > 0:
                nb_lbt_busy_events += 1
                nb_lbt_wait_ms += d_ms
                nb_backoff_slots += int(math.ceil(d_ms / max(nb_lbt_slot_ms, 1e-9)))
                events.append(
                    {
                        "start_ms": float(s["t_def_start_ms"]),
                        "end_ms": float(s["t_def_end_ms"]),
                        "label": "nb_ssbd_deferral",
                        "phase": "init",
                        "nb_ch": int(init_nb_channel),
                    }
                )
            if bool((ssbd_debug or print_ssbd_trace) and trial_idx == 0):
                print(
                    "[nb_ssbd] "
                    f"sensed_inband_dbm={float(s['sensed_inband_dbm']):.2f}, "
                    f"threshold_dbm={float(s['threshold_dbm']):.2f}, "
                    f"cca_duration_us={ssbd_cfg.phy_cca_duration_ms*1000.0:.1f}, "
                    f"cca_mode={int(ssbd_cfg.cca_mode)}, "
                    f"decision={s['decision']}, deferral_ms={d_ms:.3f}, "
                    f"NB={int(s['nb'])}, BF={int(s['bf'])}"
                )
        t_ms += float(ssbd_res.total_cca_ms + ssbd_res.total_deferral_ms)
        if not np.isfinite(t_cca_done_ms):
            t_cca_done_ms = float(t_ms)
            t_backoff_done_ms = float(t_ms)
        waiting_ms_acc += float(ssbd_res.total_deferral_ms)
        if not ssbd_res.access_granted:
            nb_access_fail = True
            retries_nb += 1
            count_nb_ssbd_timeout += 1
            last_attempt_root_reason = "nb_ssbd_timeout"
            last_attempt_stage = "init"
            if not until_success:
                break
            continue

        cfg_attempt = replace(
            cfg_base,
            distance_m=float(distance_m),
            uwb_channel=int(uwb_channel),
            seed=int(seed + 1009 * attempt),
            nb_center_override_hz=nb_center_hz_ctrl,
            wifi_channel=wifi_ch,
        )

        # Shot-1 in this attempt includes NB init/control.
        res = run_full_stack_case(
            cfg=cfg_attempt,
            wifi_on=wifi_on_fullstack,
            wifi_channel_override=wifi_ch,
            baseline_sanity_mode=False,
            case_tag=f"lat_{wifi_mode}_d{int(distance_m)}_u{uwb_channel}_t{trial_idx}_a{attempt}_s1",
            n_trials=1,
            debug_first_trial=False,
            save_psd=False,
            skip_nb_control=False,
        )
        last_res = res

        nb_bd = _nb_timing_breakdown_ms(cfg_attempt, int(res.get("nb_poll_attempts", 1)))
        nb_bd_last = nb_bd
        t0 = t_ms
        t_ms += nb_bd["nb_total_ms"]
        nb_handshake_ms_acc += nb_bd["nb_total_ms"]
        ctrl_channels.append(int(ctrl_nb_channel))
        events.append(
            {
                "start_ms": t0,
                "end_ms": t_ms,
                "label": "nb_control_tx",
                "phase": "ctrl",
                "nb_ch": int(ctrl_nb_channel),
            }
        )

        control_ok = bool(res.get("control_ok", False))
        if not control_ok:
            retries_nb += 1
            count_nb_control_decode_fail += 1
            last_attempt_root_reason = "nb_control_decode_fail"
            last_attempt_stage = "init"
        if math.isnan(conf_done_ms) and control_ok:
            conf_done_ms = t_ms

        shot_successes = 0
        shot_attempts = 0

        # Evaluate shot-1
        shot_attempts += 1
        u0 = t_ms
        if not np.isfinite(t_uwb_tx_start_ms):
            t_uwb_tx_start_ms = float(u0)
        t_ms += uwb_bd["t_uwb_airtime_ms"]
        t_uwb_rx_done_ms = float(t_ms)
        uwb_ranging_ms_acc += uwb_bd["t_uwb_airtime_ms"]
        events.append({"start_ms": u0, "end_ms": t_ms, "label": "uwb_shot"})
        shot_ok = control_ok and float(res.get("fer", 1.0)) <= 0.0 and float(res.get("ranging_fail_rate", 1.0)) <= 0.0
        if shot_ok and wifi_model == "layout" and spatial_model is not None:
            p_int_uwb_dbm = float(
                spatial_model.estimate_interference_avg_dbm(
                    t_start_ms=u0,
                    t_end_ms=t_ms,
                    rx_xy=(float(distance_m), 0.0),
                    rx_center_hz=float(uwb_fc_hz),
                    rx_bw_hz=499.2e6,
                    n_samples=7,
                )
            )
            # Lightweight mapping from interference power to ranging detectability.
            # Keeps existing PHY model while adding layout-driven blocking effect.
            snr_eff_db = float(uwb_sig_ref_dbm - p_int_uwb_dbm)
            if snr_eff_db < 3.0:
                shot_ok = False
        if shot_ok:
            shot_successes += 1
            b0 = float(res.get("ranging_bias_m", float("nan")))
            if np.isfinite(b0):
                shot_range_estimates_m.append(float(distance_m + b0))
        else:
            retries_uwb += 1
            last_attempt_stage = "ranging"

        # Additional UWB shots reuse control session.
        while control_ok and shot_attempts < max(1, int(uwb_shots_per_session)) and shot_successes < max(1, int(require_k_successes)):
            if occ is not None:
                d2 = occ.defer_ms(t_ms)
                t_ms += d2
                waiting_ms_acc += d2
                if d2 > 0:
                    wifi_cca_busy_events += 1
                    wifi_cca_wait_ms += d2
                    wifi_backoff_slots += int(math.ceil(d2 / max(nb_lbt_slot_ms, 1e-9)))
                    events.append({"start_ms": t_ms - d2, "end_ms": t_ms, "label": "wifi_cca_wait"})
            if uwb_shot_gap_ms > 0.0:
                g0 = t_ms
                t_ms += float(uwb_shot_gap_ms)
                events.append({"start_ms": g0, "end_ms": t_ms, "label": "uwb_gap"})

            res_s = run_full_stack_case(
                cfg=replace(cfg_attempt, seed=int(seed + 1009 * attempt + 17 * shot_attempts)),
                wifi_on=wifi_on_fullstack,
                wifi_channel_override=wifi_ch,
                baseline_sanity_mode=False,
                case_tag=f"lat_{wifi_mode}_d{int(distance_m)}_u{uwb_channel}_t{trial_idx}_a{attempt}_s{shot_attempts+1}",
                n_trials=1,
                debug_first_trial=False,
                save_psd=False,
                skip_nb_control=True,
            )
            last_res = res_s
            shot_attempts += 1
            us = t_ms
            if not np.isfinite(t_uwb_tx_start_ms):
                t_uwb_tx_start_ms = float(us)
            t_ms += uwb_bd["t_uwb_airtime_ms"]
            t_uwb_rx_done_ms = float(t_ms)
            uwb_ranging_ms_acc += uwb_bd["t_uwb_airtime_ms"]
            events.append({"start_ms": us, "end_ms": t_ms, "label": "uwb_shot"})
            shot_ok = float(res_s.get("fer", 1.0)) <= 0.0 and float(res_s.get("ranging_fail_rate", 1.0)) <= 0.0
            if shot_ok and wifi_model == "layout" and spatial_model is not None:
                p_int_uwb_dbm = float(
                    spatial_model.estimate_interference_avg_dbm(
                        t_start_ms=us,
                        t_end_ms=t_ms,
                        rx_xy=(float(distance_m), 0.0),
                        rx_center_hz=float(uwb_fc_hz),
                        rx_bw_hz=499.2e6,
                        n_samples=7,
                    )
                )
                snr_eff_db = float(uwb_sig_ref_dbm - p_int_uwb_dbm)
                if snr_eff_db < 3.0:
                    shot_ok = False
            if shot_ok:
                shot_successes += 1
                bs = float(res_s.get("ranging_bias_m", float("nan")))
                if np.isfinite(bs):
                    shot_range_estimates_m.append(float(distance_m + bs))
            else:
                retries_uwb += 1
                last_attempt_stage = "ranging"

        # Optional NBA MMS report phase (after ranging exchange for this block).
        if control_ok and bool(enable_report_phase_model):
            report_nb_channel = int(selected_nb_channel_for_phase(nb_switch_cfg, "report", int(ranging_block_index)))
            report_channels.append(int(report_nb_channel))
            nb_center_hz_report = nb_center_base_hz + float(report_nb_channel) * float(nb_switch_cfg.nb_channel_spacing_mhz) * 1e6
            if not np.isfinite(t_report_start_ms):
                t_report_start_ms = float(t_ms)
            events.append(
                {
                    "start_ms": float(t_ms),
                    "end_ms": float(t_ms),
                    "label": "report_rx_enable",
                    "phase": "report",
                    "nb_ch": int(report_nb_channel),
                    "required_r2i": bool(report_required_r2i),
                    "required_i2r": bool(report_required_i2r),
                }
            )

            # Reuse NB timing scale from control frames for report periods.
            slot_ms = float(max((nb_bd["t_nb_resp_ms"] if nb_bd is not None else 0.5), 1e-3))

            def _sense_report_inband_dbm(t_probe_ms: float) -> float:
                off = float(wifi_offset_mhz or 0.0) - float(report_nb_channel) * float(nb_switch_cfg.nb_channel_spacing_mhz)
                if wifi_mode != "dense":
                    return -200.0
                if spatial_model is not None:
                    if wifi_model == "layout":
                        return float(
                            spatial_model.estimate_interference_dbm(
                                t_ms=t_probe_ms,
                                rx_xy=(0.0, 0.0),
                                rx_center_hz=float(nb_center_hz_report),
                                rx_bw_hz=2.0e6,
                            )
                        )
                    p_wide = float(spatial_model.estimate_interference_dbm(t_probe_ms, rx_xy=(0.0, 0.0)))
                    return float(p_wide - _nb_selectivity_atten_db(off, nb_bw_mhz=2.0))
                return _occupancy_interference_dbm(
                    wifi_density=float(wifi_density),
                    wifi_offset_mhz=off,
                    distance_m=float(distance_m),
                )

            def _report_rx_success(t0_ms: float, t1_ms: float) -> bool:
                if wifi_mode == "off":
                    return True
                tm = 0.5 * (float(t0_ms) + float(t1_ms))
                p_int_dbm = float(_sense_report_inband_dbm(tm))
                # Map interference level to failure probability around ED threshold.
                # More interference than threshold => sharply higher fail probability.
                margin_db = float(ssbd_cfg.phy_cca_ed_threshold_dbm - p_int_dbm)
                p_fail = float(1.0 / (1.0 + np.exp(margin_db / 2.0)))
                p_fail = float(np.clip(p_fail, 0.01, 0.99))
                return bool(rng.random() >= p_fail)

            def _run_one_report_period(*, required: bool, label_prefix: str, nslots: int) -> tuple[bool, str]:
                nonlocal t_ms, report_phase_ms, nb_report_txrx_ms, nb_ssbd_total_deferral_ms, nb_ssbd_total_cca_ms
                nonlocal nb_ssbd_cca_count, nb_ssbd_nb_count, nb_ssbd_bf_final, prev_ssbd_terminating_bf, last_ssbd_reason
                nonlocal nb_ssbd_busy_count, nb_lbt_busy_events, nb_lbt_wait_ms, nb_backoff_slots, waiting_ms_acc
                nonlocal wifi_cca_busy_events, wifi_cca_wait_ms, wifi_backoff_slots

                period_ms = float(max(0, int(nslots))) * float(slot_ms)
                if period_ms <= 0.0:
                    return (False, "slot_disabled")
                p0 = float(t_ms)
                p1 = float(t_ms + period_ms)
                report_phase_ms += float(period_ms)
                t_ms = p1
                if not required:
                    events.append(
                        {
                            "start_ms": p0,
                            "end_ms": p1,
                            "label": f"{label_prefix}_idle",
                            "phase": "report",
                            "nb_ch": int(report_nb_channel),
                        }
                    )
                    return (False, "not_required")

                prev_bf_before_rpt = int(prev_ssbd_terminating_bf)
                ssbd_rpt = run_ssbd_access(
                    t_start_ms=float(p0),
                    sense_inband_dbm_fn=_sense_report_inband_dbm,
                    cfg=ssbd_cfg,
                    rng=rng,
                    is_retransmission_attempt=False,
                    prev_terminating_bf=int(prev_bf_before_rpt),
                    debug=False,
                )
                nb_ssbd_total_deferral_ms += float(ssbd_rpt.total_deferral_ms)
                nb_ssbd_total_cca_ms += float(ssbd_rpt.total_cca_ms)
                nb_ssbd_cca_count += int(ssbd_rpt.cca_count)
                nb_ssbd_nb_count = max(nb_ssbd_nb_count, int(ssbd_rpt.nb_count))
                nb_ssbd_bf_final = int(ssbd_rpt.bf_final)
                prev_ssbd_terminating_bf = int(ssbd_rpt.bf_final)
                last_ssbd_reason = str(ssbd_rpt.reason)
                for s in ssbd_rpt.trace:
                    d_ms = float(s.get("deferral_ms", 0.0))
                    if str(s.get("decision", "")) == "busy":
                        nb_ssbd_busy_count += 1
                    if d_ms > 0:
                        nb_lbt_busy_events += 1
                        nb_lbt_wait_ms += d_ms
                        nb_backoff_slots += int(math.ceil(d_ms / max(nb_lbt_slot_ms, 1e-9)))
                        waiting_ms_acc += d_ms
                        wifi_cca_busy_events += 1
                        wifi_cca_wait_ms += d_ms
                        wifi_backoff_slots += int(math.ceil(d_ms / max(nb_lbt_slot_ms, 1e-9)))
                if not ssbd_rpt.access_granted:
                    events.append(
                        {
                            "start_ms": p0,
                            "end_ms": p1,
                            "label": f"{label_prefix}_fail_access",
                            "phase": "report",
                            "nb_ch": int(report_nb_channel),
                        }
                    )
                    return (True, "nb_report_fail_channel_access")

                # Place TX/RX attempt in-period; keep deterministic duration inside slot.
                tx_dur = float(min(max(slot_ms * 0.6, 1e-3), period_ms))
                tx0 = float(p0 + max(0.0, period_ms - tx_dur) * 0.5)
                tx1 = float(tx0 + tx_dur)
                nb_report_txrx_ms += float(tx_dur)
                rx_ok = bool(_report_rx_success(tx0, tx1))
                events.append(
                    {
                        "start_ms": tx0,
                        "end_ms": tx1,
                        "label": f"{label_prefix}_tx",
                        "phase": "report",
                        "nb_ch": int(report_nb_channel),
                        "rx_ok": bool(rx_ok),
                    }
                )
                events.append(
                    {
                        "start_ms": tx1,
                        "end_ms": tx1,
                        "label": f"{label_prefix}_rx_{'success' if rx_ok else 'fail'}",
                        "phase": "report",
                        "nb_ch": int(report_nb_channel),
                    }
                )
                return (True, "none" if rx_ok else "nb_report_fail_interference")

            # 1st reporting period: responder -> initiator
            if int(mms1st_report_nslots) > 0:
                tx1, reason1 = _run_one_report_period(required=bool(report_required_r2i), label_prefix="report1_r2i", nslots=int(mms1st_report_nslots))
                report_tx_attempted_r2i = bool(tx1)
                report_rx_ok_r2i = bool(reason1 == "none")
                if tx1 and not np.isfinite(t_report1_tx_ms):
                    # Recover from event log nearest report1 tx.
                    for ev in reversed(events):
                        if ev.get("label") == "report1_r2i_tx":
                            t_report1_tx_ms = float(ev.get("start_ms", float("nan")))
                            t_report1_rx_done_ms = float(ev.get("end_ms", float("nan")))
                            break
                if reason1 not in ("none", "not_required", "slot_disabled") and report_fail_reason == "none":
                    report_fail_reason = str(reason1)

            # 2nd reporting period: initiator -> responder (independent of report1 success)
            if int(mms2nd_report_nslots) > 0:
                tx2, reason2 = _run_one_report_period(required=bool(report_required_i2r), label_prefix="report2_i2r", nslots=int(mms2nd_report_nslots))
                report_tx_attempted_i2r = bool(tx2)
                report_rx_ok_i2r = bool(reason2 == "none")
                if tx2 and not np.isfinite(t_report2_tx_ms):
                    for ev in reversed(events):
                        if ev.get("label") == "report2_i2r_tx":
                            t_report2_tx_ms = float(ev.get("start_ms", float("nan")))
                            t_report2_rx_done_ms = float(ev.get("end_ms", float("nan")))
                            break
                if reason2 not in ("none", "not_required", "slot_disabled") and report_fail_reason == "none":
                    report_fail_reason = str(reason2)

            missing_required = (
                (bool(report_required_r2i) and not bool(report_rx_ok_r2i))
                or (bool(report_required_i2r) and not bool(report_rx_ok_i2r))
            )
            if missing_required and report_fail_reason == "none":
                report_fail_reason = "nb_report_missing_timeout"

        # Aggregate result placeholder (current PHY returns per-shot success; aggregation selector is tracked in metadata).
        _ = aggregation

        report_ok_for_success = True
        if bool(enable_report_phase_model) and not bool(assume_oob_report_on_missing):
            if bool(report_required_r2i) and not bool(report_rx_ok_r2i):
                report_ok_for_success = False
            if bool(report_required_i2r) and not bool(report_rx_ok_i2r):
                report_ok_for_success = False

        if control_ok and shot_successes >= max(1, int(require_k_successes)) and report_ok_for_success:
            success = True
            latency_to_success_ms = t_ms
            last_attempt_root_reason = "none"
            last_attempt_stage = "none"
            break

        if control_ok and shot_successes < max(1, int(require_k_successes)):
            snr_last = float(last_res.get("snr_db_avg", float("nan"))) if last_res is not None else float("nan")
            if np.isfinite(snr_last) and snr_last < 0.0:
                last_attempt_root_reason = "uwb_frame_fail_snr"
                count_uwb_frame_fail_snr += 1
            else:
                last_attempt_root_reason = "uwb_frame_fail_interference"
                count_uwb_frame_fail_interference += 1
            last_attempt_stage = "ranging"
        elif control_ok and shot_successes >= max(1, int(require_k_successes)) and not report_ok_for_success:
            last_attempt_root_reason = str(report_fail_reason if report_fail_reason != "none" else "nb_report_missing_timeout")
            last_attempt_stage = "report"

        if not until_success:
            break

    range_result_m = float("nan")
    if shot_range_estimates_m:
        x = np.asarray(shot_range_estimates_m, dtype=float)
        if aggregation == "mean":
            range_result_m = float(np.mean(x))
        elif aggregation == "min":
            range_result_m = float(np.min(x))
        else:
            range_result_m = float(np.median(x))
    t_aggregation_done_ms = float(t_ms)
    if wifi_mode == "dense":
        if wifi_model == "spatial" and spatial_model is not None:
            wifi_tx_occupancy_time_ms = float(t_ms * np.clip(spatial_model.cfg.duty_cycle, 0.0, 1.0))
        else:
            wifi_tx_occupancy_time_ms = float(t_ms * np.clip(wifi_density, 0.0, 1.0))
    if wifi_mode == "dense" and spatial_model is not None:
        rd = spatial_model.realized_duty_per_ap(0.0, float(t_ms))
        if rd.size > 0:
            wifi_realized_duty_mean = float(np.mean(rd))
            wifi_realized_duty_min = float(np.min(rd))
            wifi_realized_duty_max = float(np.max(rd))
        wifi_busy_frac_nb = float(
            spatial_model.aggregate_busy_fraction(
                t_start_ms=0.0,
                t_end_ms=float(t_ms),
                rx_center_hz=float(nb_center_hz),
                rx_bw_hz=2.0e6,
            )
        )
        wifi_busy_frac_uwb = float(
            spatial_model.aggregate_busy_fraction(
                t_start_ms=0.0,
                t_end_ms=float(t_ms),
                rx_center_hz=float(uwb_fc_hz),
                rx_bw_hz=499.2e6,
            )
        )

    ctrl_to_ranging_gap_ms = 0.0
    if np.isfinite(conf_done_ms) and np.isfinite(t_uwb_tx_start_ms):
        ctrl_to_ranging_gap_ms = float(max(0.0, t_uwb_tx_start_ms - conf_done_ms))
    nb_ssbd_cca_ms = float(nb_ssbd_total_cca_ms)
    nb_ssbd_backoff_ms = float(nb_ssbd_total_deferral_ms)
    nb_ctrl_txrx_ms = float(nb_handshake_ms_acc)
    uwb_ranging_txrx_ms = float(uwb_ranging_ms_acc)
    total_session_ms = float(t_ms - t0_trial_start_ms)
    known_sum = (
        float(init_scan_wait_ms)
        + float(advertising_wait_ms)
        + float(nb_ssbd_cca_ms)
        + float(nb_ssbd_backoff_ms)
        + float(nb_ctrl_txrx_ms)
        + float(ctrl_to_ranging_gap_ms)
        + float(uwb_ranging_txrx_ms)
        + float(report_phase_ms)
    )
    misc_overhead_ms = float(max(0.0, total_session_ms - known_sum))

    row = {
        "scenario": f"{wifi_mode}_latency",
        "wifi_model": str(wifi_model),
        "enable_nb_channel_switching": int(1 if (nb_switch_cfg.enable_switching and len(nb_switch_cfg.allow_list) > 1) else 0),
        "nb_channel_switching_field": int(nb_switch_cfg.channel_switching_field),
        "mac_mms_prng_seed": int(nb_switch_cfg.mms_prng_seed),
        "mmsNbInitChannel": int(nb_switch_cfg.mms_nb_init_channel),
        "mmsNbChannelAllowList_json": json.dumps(list(nb_switch_cfg.allow_list), separators=(",", ":")),
        "nb_phase_rule_init": "fixed_mmsNbInitChannel",
        "nb_phase_rule_ctrl_report": "allow_list_lowest_if_switching_disabled_else_prng",
        "wifi_mode": wifi_mode,
        "wifi_density": float(0.0 if wifi_mode == "off" else wifi_density),
        "distance_m": float(distance_m),
        "dist_m": float(distance_m),
        "uwb_channel": int(uwb_channel),
        "uwb_ch": int(uwb_channel),
        "nb_center_hz": float(nb_center_hz),
        "uwb_center_hz": float(uwb_fc_hz),
        "uwb_sig_ref_dbm_layout": float(uwb_sig_ref_dbm),
        "wifi_center_hz": float(wifi_fc_hz),
        "wifi_offset_mhz_req": float(wifi_offset_mhz) if wifi_offset_mhz is not None else float("nan"),
        "wifi_offset_mhz": float(wifi_offset_mhz) if wifi_offset_mhz is not None else float("nan"),
        "n_ap": int(spatial_model.cfg.n_ap) if spatial_model is not None else 0,
        "area_size_m": float(spatial_model.cfg.area_size_m) if spatial_model is not None else float("nan"),
        "latency_to_uwb_success_ms": None if not np.isfinite(latency_to_success_ms) else float(latency_to_success_ms),
        "latency_to_conf_done_ms": float(conf_done_ms),
        "latency_to_success_ms": None if not np.isfinite(latency_to_success_ms) else float(latency_to_success_ms),
        "attempts": int(attempts_used),
        "attempts_used": int(attempts_used),
        "retries_nb": int(retries_nb),
        "retries_uwb": int(retries_uwb),
        "nb_lbt_busy_events": int(nb_lbt_busy_events),
        "nb_lbt_wait_ms": float(nb_lbt_wait_ms),
        "nb_backoff_slots": int(nb_backoff_slots),
        "nb_tx_attempts": int(attempts_used),
        "nb_access_fail": bool(nb_access_fail),
        "nb_ssbd_total_deferral_ms": float(nb_ssbd_total_deferral_ms),
        "nb_ssbd_total_cca_ms": float(nb_ssbd_total_cca_ms),
        "nb_ssbd_cca_count": int(nb_ssbd_cca_count),
        "nb_ssbd_busy_count": int(nb_ssbd_busy_count),
        "nb_ssbd_nb_count": int(nb_ssbd_nb_count),
        "nb_ssbd_bf_final": int(nb_ssbd_bf_final),
        "nb_ssbd_last_reason": str(last_ssbd_reason),
        "wifi_cca_busy_events": int(wifi_cca_busy_events),
        "wifi_cca_wait_ms": float(wifi_cca_wait_ms),
        "wifi_backoff_slots": int(wifi_backoff_slots),
        "wifi_tx_occupancy_time": float(wifi_tx_occupancy_time_ms),
        "wifi_busy_frac_nb": float(wifi_busy_frac_nb),
        "wifi_busy_frac_uwb": float(wifi_busy_frac_uwb),
        "wifi_realized_duty_mean": float(wifi_realized_duty_mean),
        "wifi_realized_duty_min": float(wifi_realized_duty_min),
        "wifi_realized_duty_max": float(wifi_realized_duty_max),
        "nb_cca_busy_rate": float(nb_ssbd_busy_count / max(nb_ssbd_cca_count, 1)),
        "nb_handshake_ms": float(nb_handshake_ms_acc),
        "uwb_ranging_ms": float(uwb_ranging_ms_acc),
        "waiting_ms": float(waiting_ms_acc),
        "backoff_ms": float(waiting_ms_acc),
        "time_spent_ms": float(t_ms),
        "success": bool(success),
        "seed": int(seed),
        "trial_index": int(trial_idx),
        "ber": float(last_res.get("ber", float("nan"))) if last_res is not None else float("nan"),
        "fer": float(last_res.get("fer", float("nan"))) if last_res is not None else float("nan"),
        "snr_db": float(last_res.get("snr_db_avg", float("nan"))) if last_res is not None else float("nan"),
        "ranging_fail": float(last_res.get("ranging_fail_rate", float("nan"))) if last_res is not None else float("nan"),
        "first_path_index": float(last_res.get("first_path_index_mean", float("nan"))) if last_res is not None else float("nan"),
        "peak_index": float(last_res.get("peak_index_mean", float("nan"))) if last_res is not None else float("nan"),
        "detection_threshold_abs": float(last_res.get("detection_threshold_abs_mean", float("nan"))) if last_res is not None else float("nan"),
        "estimated_tof_ns": float(last_res.get("estimated_tof_ns_mean", float("nan"))) if last_res is not None else float("nan"),
        "sample_period_ns": float(last_res.get("sample_period_ns", float("nan"))) if last_res is not None else float("nan"),
        "applied_calibration_offset_ns": float(last_res.get("applied_calibration_offset_ns", float("nan"))) if last_res is not None else float("nan"),
        "first_path_thr_db": float(last_res.get("first_path_thr_db_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_peak_frac": float(last_res.get("first_path_peak_frac_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_noise_floor": float(last_res.get("first_path_noise_floor_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_thr_noise_abs": float(last_res.get("first_path_thr_noise_abs_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_thr_peak_abs": float(last_res.get("first_path_thr_peak_abs_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_snr_corr_db": float(last_res.get("first_path_snr_corr_db_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_peak_ratio_db": float(last_res.get("first_path_peak_ratio_db_mean", float("nan"))) if last_res is not None else float("nan"),
        "noise_win_start": float(last_res.get("noise_win_start_mean", float("nan"))) if last_res is not None else float("nan"),
        "noise_win_end": float(last_res.get("noise_win_end_mean", float("nan"))) if last_res is not None else float("nan"),
        "first_path_fallback_rate": float(last_res.get("first_path_fallback_rate", float("nan"))) if last_res is not None else float("nan"),
        "a2b1_rstu": float(last_res.get("a2b1_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "b2a1_rstu": float(last_res.get("b2a1_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "a2b2_rstu": float(last_res.get("a2b2_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "b2a2_rstu": float(last_res.get("b2a2_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "ds_ra_rstu": float(last_res.get("ds_ra_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "ds_rb_rstu": float(last_res.get("ds_rb_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "ds_da_rstu": float(last_res.get("ds_da_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "ds_db_rstu": float(last_res.get("ds_db_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "ds_tof_rstu": float(last_res.get("ds_tof_rstu_mean", float("nan"))) if last_res is not None else float("nan"),
        "control_ok_last": bool(last_res.get("control_ok", False)) if last_res is not None else False,
        "report_enabled": bool(report_enabled),
        "report_required_responder_to_initiator": bool(report_required_r2i),
        "report_required_initiator_to_responder": bool(report_required_i2r),
        "report_tx_attempted_R2I": bool(report_tx_attempted_r2i),
        "report_rx_ok_R2I": bool(report_rx_ok_r2i),
        "report_tx_attempted_I2R": bool(report_tx_attempted_i2r),
        "report_rx_ok_I2R": bool(report_rx_ok_i2r),
        "t_report_start_ms": float(t_report_start_ms),
        "t_report1_tx_ms": float(t_report1_tx_ms),
        "t_report1_rx_done_ms": float(t_report1_rx_done_ms),
        "t_report2_tx_ms": float(t_report2_tx_ms),
        "t_report2_rx_done_ms": float(t_report2_rx_done_ms),
        "nb_report_txrx_ms": float(nb_report_txrx_ms),
        "uwb_shots_per_session": int(uwb_shots_per_session),
        "require_k_successes": int(require_k_successes),
        "aggregation": str(aggregation),
        "range_result_m": float(range_result_m),
        "t_nb_adv_ms": float(nb_bd_last["t_nb_adv_ms"]) if nb_bd_last else float("nan"),
        "t_nb_poll_ms": float(nb_bd_last["t_nb_poll_ms"]) if nb_bd_last else float("nan"),
        "t_nb_resp_ms": float(nb_bd_last["t_nb_resp_ms"]) if nb_bd_last else float("nan"),
        "t_nb_conf_ms": float(nb_bd_last["t_nb_conf_ms"]) if nb_bd_last else float("nan"),
        "t_nb_ifs_total_ms": float(nb_bd_last["t_nb_ifs_total_ms"]) if nb_bd_last else float("nan"),
        "t_uwb_airtime_ms": float(uwb_bd["t_uwb_airtime_ms"]),
        "t_uwb_reply_delay_ms": float(uwb_bd["t_uwb_reply_delay_ms"]),
        "event_trace_json": json.dumps(events, separators=(",", ":")),
        "ssbd_attempts_json": json.dumps(ssbd_attempt_summaries, separators=(",", ":")),
        "nb_selected_channel": int(selected_channels[-1]) if selected_channels else int(selected_nb_channel_for_block(nb_switch_cfg, 0)),
        "ranging_block_index": int(ranging_block_index if attempts_used > 0 else 0),
        "nb_selected_channel_seq_json": json.dumps(selected_channels, separators=(",", ":")),
        "nb_channel_seq_first8_json": json.dumps(selected_channels[:8], separators=(",", ":")),
        "nb_ch_init": int(init_channels[0]) if init_channels else int(selected_nb_channel_for_block(nb_switch_cfg, 0)),
        "nb_ch_init_seq_first8_json": json.dumps(init_channels[:8], separators=(",", ":")),
        "nb_ch_ctrl_seq_first8_json": json.dumps(ctrl_channels[:8], separators=(",", ":")),
        "nb_ch_report_seq_first8_json": json.dumps(report_channels[:8], separators=(",", ":")),
        "nb_ch_all_seq_first8_json": json.dumps(selected_channels[:8], separators=(",", ":")),
        "nb_ch_unique_count_init": int(len(set(init_channels))) if init_channels else 0,
        "nb_ch_unique_count_ctrl": int(len(set(ctrl_channels))) if ctrl_channels else 0,
        "nb_ch_unique_count_report": int(len(set(report_channels))) if report_channels else 0,
        "ssbd_nb": int(nb_ssbd_nb_count),
        "ssbd_bf": int(nb_ssbd_bf_final),
        "sim_runtime_s": float(time.perf_counter() - t_run0),
        "terminal_fail_reason": str(last_attempt_root_reason if not success else "none"),
        "last_attempt_stage": str(last_attempt_stage if not success else "none"),
        "last_attempt_nb_channel": int(selected_channels[-1]) if selected_channels else int(selected_nb_channel_for_block(nb_switch_cfg, 0)),
        "last_attempt_uwb_channel": int(uwb_channel),
        "count_nb_ssbd_timeout": int(count_nb_ssbd_timeout),
        "count_nb_control_decode_fail": int(count_nb_control_decode_fail),
        "count_uwb_frame_fail_snr": int(count_uwb_frame_fail_snr),
        "count_uwb_frame_fail_interference": int(count_uwb_frame_fail_interference),
        "count_cca_busy": int(nb_ssbd_busy_count),
        "nb_channel_match": bool(nb_channel_match),
        "nb_channel_switching_enabled": bool(nb_switch_cfg.enable_switching and len(nb_switch_cfg.allow_list) > 1),
        "nb_allow_list_len": int(len(nb_switch_cfg.allow_list)),
        "nb_allow_list_min": int(min(nb_switch_cfg.allow_list)) if len(nb_switch_cfg.allow_list) > 0 else -1,
        # Time breakdown timestamps (simulation-time, ms)
        "t0_trial_start_ms": float(t0_trial_start_ms),
        "t_init_done_ms": float(t_init_done_ms),
        "t_scan_start_ms": float(t_scan_start_ms),
        "t_adv_start_ms": float(t_adv_start_ms),
        "t_adv_rx_detected_ms": float(t_adv_rx_detected_ms),
        "t_cca_start_ms": float(t_cca_start_ms),
        "t_cca_done_ms": float(t_cca_done_ms),
        "t_backoff_done_ms": float(t_backoff_done_ms),
        "t_uwb_tx_start_ms": float(t_uwb_tx_start_ms),
        "t_uwb_rx_done_ms": float(t_uwb_rx_done_ms),
        "t_toa_est_start_ms": float(t_toa_est_start_ms),
        "t_toa_est_done_ms": float(t_toa_est_done_ms),
        "t_aggregation_done_ms": float(t_aggregation_done_ms),
        "t_trial_end_ms": float(t_ms),
        # Time breakdown deltas (ms)
        "lat_total_ms": float(t_ms - t0_trial_start_ms),
        "lat_adv_scan_wait_ms": float(
            (t_adv_rx_detected_ms - min(t_scan_start_ms, t_adv_start_ms))
            if np.isfinite(t_adv_rx_detected_ms) and np.isfinite(t_scan_start_ms) and np.isfinite(t_adv_start_ms)
            else float("nan")
        ),
        "lat_lbt_cca_ms": float(nb_ssbd_total_cca_ms),
        "lat_backoff_ms": float(nb_ssbd_total_deferral_ms),
        "lat_uwb_exchange_ms": float(uwb_ranging_ms_acc),
        "lat_processing_ms": float("nan"),
        # Component breakdown (A..I)
        "init_scan_wait_ms": float(init_scan_wait_ms),
        "advertising_wait_ms": float(advertising_wait_ms),
        "nb_ssbd_cca_ms": float(nb_ssbd_cca_ms),
        "nb_ssbd_backoff_ms": float(nb_ssbd_backoff_ms),
        "nb_ctrl_txrx_ms": float(nb_ctrl_txrx_ms),
        "ctrl_to_ranging_gap_ms": float(ctrl_to_ranging_gap_ms),
        "uwb_ranging_txrx_ms": float(uwb_ranging_txrx_ms),
        "report_phase_ms": float(report_phase_ms),
        "misc_overhead_ms": float(misc_overhead_ms),
        "total_session_ms": float(total_session_ms),
        "lat_breakdown_note": (
            "adv/scan and ToA-processing wall-clock are not explicitly modeled in _run_trial; "
            "NaN means unmodeled rather than zero."
        ),
    }
    if abs(float(row["total_session_ms"]) - float(row["time_spent_ms"])) > 1e-6:
        raise RuntimeError(
            f"latency accounting mismatch: total_session_ms={row['total_session_ms']}, time_spent_ms={row['time_spent_ms']}"
        )
    # fail reason classification for latency experiments
    if row["success"]:
        row["fail_reason"] = "none"
    elif str(row.get("terminal_fail_reason", "none")).startswith("nb_report_"):
        row["fail_reason"] = str(row["terminal_fail_reason"])
    elif t_ms > max_trial_ms:
        row["fail_reason"] = "time_budget_exceeded"
    elif bool(row["nb_access_fail"]) and str(row.get("nb_ssbd_last_reason", "")) == "maxbackoffs_fail":
        row["fail_reason"] = "nb_ssbd_timeout"
    elif attempts_used >= max_attempts and until_success:
        row["fail_reason"] = "max_attempts_exceeded"
        if str(row.get("terminal_fail_reason", "none")).startswith("nb_report_"):
            row["fail_reason"] = str(row["terminal_fail_reason"])
    elif not bool(row["control_ok_last"]):
        row["fail_reason"] = "nb_control_decode_fail"
    else:
        snr = float(row["snr_db"]) if np.isfinite(row["snr_db"]) else -999.0
        row["fail_reason"] = "uwb_frame_fail_snr" if snr < 0.0 else "uwb_frame_fail_interference"
    return row


def _print_summary(rows: list[dict]) -> None:
    print("\n=== Latency Summary ===")

    def _off_key(v: float) -> str:
        return "nan" if not np.isfinite(v) else f"{v:.3f}"

    keys = sorted(
        set(
            (
                r["wifi_mode"],
                r["distance_m"],
                r["uwb_channel"],
                _off_key(float(r["wifi_offset_mhz_req"])),
            )
            for r in rows
        )
    )
    print("mode | dist(m) | uwb_ch | wifi_off_mhz | succ_rate | succ_lat mean/median/p95 [ms] | fail_time median/p95 [ms]")
    for k in keys:
        subset = [
            r for r in rows
            if (r["wifi_mode"], r["distance_m"], r["uwb_channel"], _off_key(float(r["wifi_offset_mhz_req"]))) == k
        ]
        succ = [1.0 if r["success"] else 0.0 for r in subset]
        succ_rate = float(np.mean(succ)) if succ else float("nan")
        succ_lat = [float(r["latency_to_success_ms"]) for r in subset if r["success"] and np.isfinite(r["latency_to_success_ms"])]
        fail_t = [float(r["time_spent_ms"]) for r in subset if not r["success"] and np.isfinite(r["time_spent_ms"])]
        mean_v, med_v, p95_v = _summary_stats(succ_lat)
        _, fail_med, fail_p95 = _summary_stats(fail_t)
        off_txt = "nan" if k[3] == "nan" else f"{float(k[3]):.1f}"
        print(
            f"{k[0]:>5s} | {k[1]:>7.1f} | {k[2]:>6d} | {off_txt:>11s} | {succ_rate:>9.3f} | "
            f"{mean_v:>7.3f}/{med_v:>7.3f}/{p95_v:>7.3f} | {fail_med:>7.3f}/{fail_p95:>7.3f}"
        )
    # fail reason distribution
    fr: dict[str, int] = {}
    for r in rows:
        if not r["success"]:
            key = str(r.get("fail_reason", "unknown"))
            fr[key] = fr.get(key, 0) + 1
    if fr:
        print("\nfail_reason counts:")
        for k, v in sorted(fr.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {k}: {v}")


def _print_impact_report(rows: list[dict], vary_param: str, baseline_value: float | int) -> None:
    if not rows:
        return

    def _group(vv: str) -> list[dict]:
        return [r for r in rows if str(r.get("vary_value", "")) == vv]

    base_key = str(baseline_value)
    base_rows = _group(base_key)
    if not base_rows:
        print("\n[impact] baseline group missing; skip impact report")
        return

    def _metrics(sub: list[dict]) -> tuple[float, float, float, float]:
        succ_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in sub])) if sub else float("nan")
        succ_lat = [float(r["latency_to_success_ms"]) for r in sub if r["success"] and np.isfinite(r["latency_to_success_ms"])]
        _, med, p95 = _summary_stats(succ_lat)
        fail_t = [float(r["time_spent_ms"]) for r in sub if not r["success"] and np.isfinite(r["time_spent_ms"])]
        _, fail_med, _ = _summary_stats(fail_t)
        return succ_rate, med, p95, fail_med

    b_sr, b_med, b_p95, b_fail_med = _metrics(base_rows)
    vals = sorted(set(str(r.get("vary_value", "")) for r in rows))
    table = []
    for v in vals:
        sub = _group(v)
        sr, med, p95, fail_med = _metrics(sub)
        table.append(
            {
                "v": v,
                "d_succ": sr - b_sr,
                "d_med": med - b_med,
                "d_p95": p95 - b_p95,
                "d_fail_med": fail_med - b_fail_med,
            }
        )

    table_sorted = sorted(table, key=lambda x: (x["d_succ"], -x["d_med"]))
    print(f"\n=== One-Factor Impact Report (vary={vary_param}, baseline={baseline_value}) ===")
    print("vary_value | delta_success_rate | delta_median_latency_ms | delta_p95_latency_ms | delta_fail_median_time_ms")
    for r in table_sorted:
        print(
            f"{r['v']:>10s} | {r['d_succ']:>+18.3f} | {r['d_med']:>+23.3f} | "
            f"{r['d_p95']:>+20.3f} | {r['d_fail_med']:>+25.3f}"
        )


def _print_timing_breakdown(cfg: FullStackConfig, row_example: dict) -> None:
    nb_bd = _nb_timing_breakdown_ms(cfg, nb_poll_attempts=1)
    uwb_bd = _uwb_timing_breakdown_ms()
    t_ssbd_def = float(row_example.get("nb_ssbd_total_deferral_ms", 0.0) or 0.0)
    t_ssbd_cca = float(row_example.get("nb_ssbd_total_cca_ms", 0.0) or 0.0)
    sum_ms = nb_bd["nb_total_ms"] + uwb_bd["t_uwb_airtime_ms"] + t_ssbd_def + t_ssbd_cca
    measured = float(row_example.get("latency_to_success_ms", float("nan")))
    print("\n=== Timing Breakdown (single attempt) ===")
    print(f"t_nb_adv={nb_bd['t_nb_adv_ms']:.6f} ms")
    print(f"t_nb_poll={nb_bd['t_nb_poll_ms']:.6f} ms")
    print(f"t_nb_resp={nb_bd['t_nb_resp_ms']:.6f} ms")
    print(f"t_nb_conf={nb_bd['t_nb_conf_ms']:.6f} ms")
    print(f"t_nb_ifs_total={nb_bd['t_nb_ifs_total_ms']:.6f} ms")
    print(f"t_nb_turnaround={nb_bd['t_nb_turnaround_ms']:.6f} ms")
    print(f"t_nb_ssbd_cca={t_ssbd_cca:.6f} ms")
    print(f"t_nb_ssbd_deferral={t_ssbd_def:.6f} ms")
    print(f"t_uwb_airtime={uwb_bd['t_uwb_airtime_ms']:.6f} ms")
    print(f"t_uwb_reply_delay={uwb_bd['t_uwb_reply_delay_ms']:.6f} ms")
    print(f"sum={sum_ms:.6f} ms")
    if np.isfinite(measured):
        print(f"measured_latency_to_uwb_success_ms={measured:.6f} ms")
        print(f"delta(sum-measured)={sum_ms - measured:+.6e} ms")
    else:
        print("measured_latency_to_uwb_success_ms=N/A")

    print("\n[parameter source]")
    print(f"NB source: {nb_bd['params_source']}")
    print(f"UWB source: {uwb_bd['uwb_params_source']}")
    print("UWB symbol/reply scheduling uses RSTU_S and fragment schedule from UWB/mms_uwb_packet_mode.py")
    attempts_raw = row_example.get("ssbd_attempts_json", "[]")
    try:
        attempts = json.loads(attempts_raw)
    except Exception:
        attempts = []
    if attempts:
        print("\n[ssbd-attempt-breakdown]")
        for a in attempts:
            print(
                f"attempt={a.get('attempt')} cca_count={a.get('cca_count')} "
                f"total_deferral_ms={float(a.get('total_deferral_ms', 0.0)):.6f} "
                f"final_NB={a.get('final_NB')} final_BF={a.get('final_BF')} "
                f"bf_init={a.get('bf_init')} terminated_by={a.get('terminated_by')}"
            )


def _maybe_plot(rows: list[dict], out_csv: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    out_dir = out_csv.parent
    for mode in sorted(set(r["wifi_mode"] for r in rows)):
        ds = sorted(set(float(r["distance_m"]) for r in rows if r["wifi_mode"] == mode))
        ys = []
        for d in ds:
            lat = [
                float(r["latency_to_success_ms"])
                for r in rows
                if r["wifi_mode"] == mode and float(r["distance_m"]) == d and r["success"] and np.isfinite(r["latency_to_success_ms"])
            ]
            ys.append(np.mean(lat) if lat else np.nan)
        plt.figure(figsize=(7, 4))
        plt.plot(ds, ys, marker="o")
        plt.title(f"Latency vs Distance ({mode})")
        plt.xlabel("Distance [m]")
        plt.ylabel("Latency to UWB Success [ms]")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{out_csv.stem}_{mode}_latency_vs_distance.png", dpi=150)
        plt.close()


def _maybe_plot_density_study(rows: list[dict], out_csv: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return

    df = pd.DataFrame(rows)
    if df.empty or "wifi_density" not in df.columns:
        return

    dfd = df[(df["wifi_mode"] == "dense") & (df["wifi_density"].notna())].copy()
    if dfd.empty:
        return

    # Per-point debug to prevent empty-plot confusion.
    print("\n[density-study-debug]")
    modes = sorted(dfd["wifi_model"].dropna().unique()) if "wifi_model" in dfd.columns else ["dense"]
    if not modes:
        modes = ["dense"]
    for m in modes:
        dm = dfd[dfd["wifi_model"] == m] if "wifi_model" in dfd.columns else dfd
        for den in sorted(dm["wifi_density"].dropna().unique()):
            g = dm[dm["wifi_density"] == den]
            total = int(len(g))
            succ = int(g["success"].sum()) if "success" in g.columns else 0
            fail = int(total - succ)
            lat_all = g["time_spent_ms"] if "time_spent_ms" in g.columns else g["latency_to_success_ms"]
            mean_all = float(np.nanmean(lat_all.to_numpy(dtype=float))) if total > 0 else float("nan")
            lat_succ = g[g["success"] == True]["latency_to_success_ms"] if "latency_to_success_ms" in g.columns else g[g["success"] == True]["time_spent_ms"]
            mean_succ = float(np.nanmean(lat_succ.to_numpy(dtype=float))) if len(lat_succ) > 0 else float("nan")
            print(
                f"mode={m:10s} density={float(den):.3f} total={total:4d} success={succ:4d} fail={fail:4d} "
                f"mean_latency_all={mean_all:8.3f} mean_latency_success={mean_succ:8.3f}"
            )
            if succ == 0:
                print(
                    f"  [WARN] success_count=0 for density={float(den):.3f}, mode={m}. "
                    "추천: tx_power↑, SIR_threshold↓, retries↑, density range↓"
                )

    # Aggregate
    grp = (
        dfd.groupby(["wifi_model", "wifi_density"], dropna=False)
        .agg(
            n_trials=("success", "count"),
            success_count=("success", "sum"),
            success_prob=("success", "mean"),
            latency_all_mean_ms=("time_spent_ms", "mean"),
            latency_all_median_ms=("time_spent_ms", "median"),
            latency_all_p95_ms=("time_spent_ms", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 95))),
        )
        .reset_index()
    )

    # Success-only latency metrics.
    ok = dfd[(dfd["success"] == True) & (dfd["latency_to_success_ms"].notna())].copy()
    if not ok.empty:
        okg = (
            ok.groupby(["wifi_model", "wifi_density"], dropna=False)["latency_to_success_ms"]
            .agg(
                latency_success_mean_ms="mean",
                latency_success_median_ms="median",
                latency_success_p95_ms=lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 95)),
            )
            .reset_index()
        )
        grp = grp.merge(okg, on=["wifi_model", "wifi_density"], how="left")
    else:
        grp["latency_success_mean_ms"] = np.nan
        grp["latency_success_median_ms"] = np.nan
        grp["latency_success_p95_ms"] = np.nan

    out_dir = out_csv.parent
    grp.to_csv(out_dir / f"{out_csv.stem}_density_study_summary.csv", index=False)

    # Plot 1: success probability vs density
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for m in sorted(grp["wifi_model"].dropna().unique()):
        g = grp[grp["wifi_model"] == m].sort_values("wifi_density")
        ax.plot(g["wifi_density"], g["success_prob"], marker="o", label=str(m))
    ax.set_title("Success Probability vs Wi-Fi Density")
    ax.set_xlabel("Wi-Fi density (airtime duty scale)")
    ax.set_ylabel("Success probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_csv.stem}_plot1_success_prob_vs_density.png", dpi=150)
    plt.close(fig)

    # Plot 2: latency vs density
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for m in sorted(grp["wifi_model"].dropna().unique()):
        g = grp[grp["wifi_model"] == m].sort_values("wifi_density")
        ax.plot(g["wifi_density"], g["latency_all_mean_ms"], marker="o", linestyle="-", label=f"{m} all_mean")
        ax.plot(g["wifi_density"], g["latency_success_median_ms"], marker="x", linestyle="--", label=f"{m} succ_median")
        ax.plot(g["wifi_density"], g["latency_success_p95_ms"], marker="^", linestyle=":", label=f"{m} succ_p95")
    ax.set_title("Latency vs Wi-Fi Density")
    ax.set_xlabel("Wi-Fi density (airtime duty scale)")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_csv.stem}_plot2_latency_vs_density.png", dpi=150)
    plt.close(fig)

    # Plot 3: CDF for selected densities
    dens_all = sorted(dfd["wifi_density"].dropna().unique())
    if dens_all:
        pick = [dens_all[0], dens_all[len(dens_all) // 2], dens_all[-1]]
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharey=True)
        for ax, den in zip(axes, pick):
            for m in sorted(dfd["wifi_model"].dropna().unique()):
                x = dfd[(dfd["wifi_model"] == m) & (dfd["wifi_density"] == den)]["time_spent_ms"].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                x = np.sort(x)
                y = np.arange(1, x.size + 1) / x.size
                ax.plot(x, y, label=str(m))
            ax.set_title(f"CDF @ density={float(den):.3f}")
            ax.set_xlabel("Latency [ms]")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("CDF")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{out_csv.stem}_plot3_latency_cdf_selected_density.png", dpi=150)
        plt.close(fig)

    print(f"Saved density-study plots/summary near {out_csv}.")


def _maybe_dump_timeline(rows: list[dict], out_csv: Path, n: int, seed: int) -> None:
    if n <= 0:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    cand = [r for r in rows if isinstance(r.get("event_trace_json"), str) and r.get("event_trace_json")]
    if not cand:
        return
    rng = random.Random(int(seed))
    rng.shuffle(cand)
    out_dir = out_csv.parent
    cmap = {
        "nb_lbt_wait": "tab:orange",
        "nb_lbt_wait_spatial": "tab:brown",
        "nb_ssbd_deferral": "tab:brown",
        "nb_control_tx": "tab:blue",
        "wifi_cca_wait": "tab:purple",
        "uwb_gap": "tab:gray",
        "uwb_shot": "tab:green",
    }
    for i, row in enumerate(cand[: int(n)]):
        try:
            ev = json.loads(str(row.get("event_trace_json", "[]")))
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for e in ev:
            s = float(e.get("start_ms", 0.0))
            e_ms = float(e.get("end_ms", s))
            lbl = str(e.get("label", "evt"))
            ax.broken_barh([(s, max(0.0, e_ms - s))], (0, 8), facecolors=cmap.get(lbl, "tab:red"))
        ax.set_title(f"Timeline #{i} success={row.get('success')}")
        ax.set_xlabel("Time [ms]")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{out_csv.stem}_timeline_{i:03d}.png", dpi=150)
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MMS latency sweep for NB init/control + UWB ranging.")
    p.add_argument("--wifi-mode", choices=["off", "dense", "both"], default="off")
    p.add_argument("--wifi-model", choices=["occupancy", "spatial", "layout"], default="occupancy")
    p.add_argument("--wifi-layout-file", type=str, default=None, help="JSON/YAML layout for wifi-model=layout")
    p.add_argument("--wifi-density", type=float, default=0.75, help="Dense Wi-Fi occupancy duty cycle in [0,1].")
    p.add_argument("--wifi-density-list", type=str, default=None)
    p.add_argument("--area-size-m", type=float, default=100.0)
    p.add_argument("--n-ap", type=int, default=20)
    p.add_argument("--n-sta-per-ap", type=int, default=0)
    p.add_argument("--ap-tx-power-dbm", type=float, default=20.0)
    p.add_argument("--dist-start", type=float, default=5.0)
    p.add_argument("--dist-stop", type=float, default=25.0)
    p.add_argument("--dist-step", type=float, default=5.0)
    p.add_argument("--dist-list", type=str, default=None)
    p.add_argument("--uwb-channels", type=str, default="5,7,8,10")
    p.add_argument("--nb-center-ghz", type=float, default=6.4890)
    p.add_argument("--wifi-offsets-mhz", type=str, default="[-80,-40,0,40,80]")
    p.add_argument("--enable-nb-channel-switching", type=int, choices=[0, 1], default=0)
    p.add_argument("--nb-allow-list", type=str, default=None)
    p.add_argument("--nb-channel-start", type=int, default=0)
    p.add_argument("--nb-channel-step", type=int, default=1)
    p.add_argument("--nb-channel-step-code", type=int, default=None, help="Table-11 style step code (0..7 -> 1,2,4,8,...)")
    p.add_argument("--nb-channel-bitmask-hex", type=str, default=None)
    p.add_argument("--nb-channel-switching-field", type=int, choices=[0, 1], default=1)
    p.add_argument("--mms-prng-seed", type=int, default=0)
    p.add_argument("--nb-init-channel", type=int, default=2, help="mmsNbInitChannel used for init phase (default 2).")
    p.add_argument("--nb-channel-spacing-mhz", type=float, default=5.0)
    p.add_argument("--dump-nb-channel-seq", type=int, default=0)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260301)
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--until-success", action="store_true")
    p.add_argument("--max-trial-ms", type=float, default=2000.0)
    p.add_argument("--uwb-shots-per-session", type=int, default=1)
    p.add_argument("--aggregation", choices=["mean", "median", "min"], default="median")
    p.add_argument("--require-k-successes", type=int, default=None)
    p.add_argument("--uwb-shot-gap-ms", type=float, default=0.2)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-density-study", action="store_true")
    p.add_argument("--cache-toa-calibration", action="store_true", default=True)
    p.add_argument("--no-cache-toa-calibration", dest="cache_toa_calibration", action="store_false")
    p.add_argument("--timing-log", action="store_true", help="Print per-trial runtime timing summary.")
    p.add_argument("--dump-timeline", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--pilot-trials", type=int, default=0)
    p.add_argument("--use-standard-defaults", action="store_true")
    p.add_argument("--nb-phy-cca-duration-ms", type=float, default=None)
    p.add_argument("--nb-phy-cca-ed-threshold-dbm", type=float, default=None)
    p.add_argument("--phy-cca-duration-ms", type=float, default=None, help="Alias of --nb-phy-cca-duration-ms")
    p.add_argument("--phy-cca-ed-threshold-dbm", type=float, default=None, help="Alias of --nb-phy-cca-ed-threshold-dbm")
    p.add_argument("--ssbd-cca-mode", type=int, choices=[1, 3], default=1)
    # Preferred SSBD CLI knobs (Table 67 defaults).
    p.add_argument("--ssbd-tx-on-end", type=int, choices=[0, 1], default=1)
    p.add_argument("--ssbd-persistence", type=int, choices=[0, 1], default=0)
    p.add_argument("--ssbd-max-backoffs", type=int, default=5)
    p.add_argument("--ssbd-max-bf", type=int, default=5)
    p.add_argument("--ssbd-min-bf", type=int, default=1)
    p.add_argument("--ssbd-unit-backoff-us", type=float, default=1.0)
    p.add_argument("--ssbd-unit-backoff-period-us", type=float, default=None, help="Alias of --ssbd-unit-backoff-us")
    # Legacy aliases (kept for compatibility)
    p.add_argument("--nb-ssbd-unit-backoff-ms", type=float, default=None)
    p.add_argument("--nb-ssbd-min-bf", type=int, default=None)
    p.add_argument("--nb-ssbd-max-bf", type=int, default=None)
    p.add_argument("--nb-ssbd-max-backoffs", type=int, default=None)
    p.add_argument("--nb-ssbd-persistence", type=float, default=None)
    p.add_argument("--nb-ssbd-tx-on-end", action="store_true")
    p.add_argument("--ssbd-debug", action="store_true")
    p.add_argument("--print-ssbd-trace", action="store_true")

    p.add_argument("--sweep-mode", choices=["grid", "one_factor"], default="grid")
    p.add_argument("--baseline-dist", type=float, default=20.0)
    p.add_argument("--baseline-uwb-ch", type=int, default=5)
    p.add_argument("--baseline-wifi-offset-mhz", type=float, default=0.0)
    p.add_argument("--baseline-wifi-density", type=float, default=0.75)
    p.add_argument("--vary", choices=["dist", "uwb_ch", "wifi_offset_mhz", "wifi_density"], default=None)

    p.add_argument("--paired-trials", action="store_true")
    p.add_argument("--early-stop-zero-success", action="store_true")
    p.add_argument("--early-stop-min-trials", type=int, default=5)
    p.add_argument("--early-stop-max-sr", type=float, default=0.05)

    p.add_argument("--print-timing-breakdown", action="store_true")
    p.add_argument("--enable-report-phase-model", type=int, choices=[0, 1], default=0)
    p.add_argument("--initiator-report-request", type=int, choices=[0, 1], default=0)
    p.add_argument("--responder-report-request", type=int, choices=[0, 1], default=0)
    p.add_argument("--mms1stReportNSlots", type=int, default=1)
    p.add_argument("--mms2ndReportNSlots", type=int, default=0)
    p.add_argument("--assume-oob-report-on-missing", type=int, choices=[0, 1], default=1)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    std_params = get_default_params("802154ab" if args.use_standard_defaults else "repo_defaults")

    distances = _distances_from_args(args)
    uwb_channels = _parse_int_list(args.uwb_channels)
    if not uwb_channels:
        raise ValueError("No UWB channels provided")
    wifi_offsets = _parse_float_list(args.wifi_offsets_mhz)
    if not wifi_offsets:
        wifi_offsets = [0.0]
    explicit_allow = _parse_int_list(args.nb_allow_list) if args.nb_allow_list else None
    nb_allow_list = build_nb_channel_allow_list(
        explicit_allow_list=explicit_allow,
        nb_channel_start=int(args.nb_channel_start),
        nb_channel_step=int(args.nb_channel_step),
        nb_channel_step_code=args.nb_channel_step_code,
        nb_channel_bitmask_hex=args.nb_channel_bitmask_hex,
    )
    nb_switch_cfg = NbChannelSwitchConfig(
        enable_switching=bool(int(args.enable_nb_channel_switching)),
        allow_list=tuple(int(v) for v in nb_allow_list),
        mms_prng_seed=int(args.mms_prng_seed) & 0xFF,
        channel_switching_field=int(args.nb_channel_switching_field),
        nb_channel_spacing_mhz=float(args.nb_channel_spacing_mhz),
        mms_nb_init_channel=int(args.nb_init_channel),
    )
    wifi_density_list = _parse_float_list(args.wifi_density_list) if args.wifi_density_list else [float(args.wifi_density)]

    if args.wifi_mode == "both":
        modes = ["off", "dense"]
    else:
        modes = [args.wifi_mode]

    if args.trials is None:
        trials = 10 if args.sweep_mode == "one_factor" else 20
    else:
        trials = int(args.trials)

    require_k = int(args.require_k_successes) if args.require_k_successes is not None else int(std_params.uwb_require_k_successes)
    require_k = max(1, min(require_k, int(args.uwb_shots_per_session)))
    unit_us = float(args.ssbd_unit_backoff_us)
    if args.ssbd_unit_backoff_period_us is not None:
        unit_us = float(args.ssbd_unit_backoff_period_us)
    ssbd_unit_backoff_ms = unit_us / 1000.0
    if args.nb_ssbd_unit_backoff_ms is not None:
        ssbd_unit_backoff_ms = float(args.nb_ssbd_unit_backoff_ms)
    ssbd_min_bf = int(args.ssbd_min_bf if args.nb_ssbd_min_bf is None else args.nb_ssbd_min_bf)
    ssbd_max_bf = int(args.ssbd_max_bf if args.nb_ssbd_max_bf is None else args.nb_ssbd_max_bf)
    ssbd_max_backoffs = int(args.ssbd_max_backoffs if args.nb_ssbd_max_backoffs is None else args.nb_ssbd_max_backoffs)
    ssbd_tx_on_end = bool(int(args.ssbd_tx_on_end))
    if args.nb_ssbd_tx_on_end:
        ssbd_tx_on_end = True
    ssbd_persistence = bool(int(args.ssbd_persistence))
    if args.nb_ssbd_persistence is not None:
        ssbd_persistence = bool(int(args.nb_ssbd_persistence))
    cca_dur_arg = args.nb_phy_cca_duration_ms
    if args.phy_cca_duration_ms is not None:
        cca_dur_arg = args.phy_cca_duration_ms
    cca_ed_arg = args.nb_phy_cca_ed_threshold_dbm
    if args.phy_cca_ed_threshold_dbm is not None:
        cca_ed_arg = args.phy_cca_ed_threshold_dbm
    ssbd_cfg = SsbdConfig(
        phy_cca_duration_ms=float(
            cca_dur_arg if cca_dur_arg is not None else std_params.nb_phy_cca_duration_ms
        ),
        phy_cca_ed_threshold_dbm=float(
            cca_ed_arg
            if cca_ed_arg is not None
            else std_params.nb_phy_cca_ed_threshold_dbm
        ),
        cca_mode=int(args.ssbd_cca_mode),
        mac_ssbd_unit_backoff_ms=float(ssbd_unit_backoff_ms),
        mac_ssbd_min_bf=int(ssbd_min_bf),
        mac_ssbd_max_bf=int(ssbd_max_bf),
        mac_ssbd_max_backoffs=int(ssbd_max_backoffs),
        mac_ssbd_tx_on_end=bool(ssbd_tx_on_end),
        mac_ssbd_persistence=bool(ssbd_persistence),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_default = Path("simulation/mms/results") / f"latency_sweep_{stamp}.csv"
    out_csv = Path(args.output) if args.output else out_default
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg_base = FullStackConfig(
        distance_m=20.0,
        nb_channel=1,
        uwb_channel=5,
        wifi_channel=108,
        wifi_bw_mhz=160,
        wifi_standard="wifi7",
        nb_center_override_hz=float(args.nb_center_ghz) * 1e9,
        nb_eirp_dbw=-16.0,
        wifi_tx_power_dbw=-20.0 if args.wifi_mode == "off" else -10.0,
        nf_db=6.0,
        seed=int(args.seed),
    )
    if args.cache_toa_calibration:
        t_cal0 = time.perf_counter()
        cal_samp = _one_time_toa_calibration_samples(cfg_base)
        cfg_base = replace(cfg_base, toa_calibration_override=float(cal_samp))
        print(
            f"[perf] one-time ToA calibration cached: samples={cal_samp:.6f} "
            f"(elapsed={time.perf_counter() - t_cal0:.3f}s)"
        )
    layout_cfg = None
    if args.wifi_model == "layout":
        if not args.wifi_layout_file:
            raise ValueError("--wifi-model layout requires --wifi-layout-file")
        layout_cfg = load_wifi_layout(args.wifi_layout_file)
    if args.use_standard_defaults:
        print(f"[standard] source={std_params.source} note={std_params.note}")
        print(f"[standard] section_hint={std_params.section_hint}")
        print(
            "[standard] "
            f"nb_ifs_ms={std_params.nb_ifs_ms}, nb_lbt_cca_slots={std_params.nb_lbt_cca_slots}, "
            f"nb_lbt_slot_ms={std_params.nb_lbt_slot_ms}, nb_retry_limit={std_params.nb_retry_limit}, "
            f"nb_phy_cca_duration_ms={std_params.nb_phy_cca_duration_ms}, "
            f"nb_phy_cca_ed_threshold_dbm={std_params.nb_phy_cca_ed_threshold_dbm}, "
            f"nb_ssbd_unit_backoff_ms={std_params.nb_ssbd_unit_backoff_ms}, "
            f"nb_ssbd_min_bf={std_params.nb_ssbd_min_bf}, nb_ssbd_max_bf={std_params.nb_ssbd_max_bf}, "
            f"nb_ssbd_max_backoffs={std_params.nb_ssbd_max_backoffs}, nb_ssbd_persistence={std_params.nb_ssbd_persistence}, "
            f"uwb_shots_per_session={std_params.uwb_shots_per_session}, uwb_require_k_successes={std_params.uwb_require_k_successes}, "
            f"uwb_reply_delay_ms={std_params.uwb_reply_delay_ms}"
        )
    print(
        "[ssbd-config] "
        f"macSsbdTxOnEnd={int(ssbd_cfg.mac_ssbd_tx_on_end)} "
        f"macSsbdPersistence={int(ssbd_cfg.mac_ssbd_persistence)} "
        f"macSsbdMaxBackoffs={ssbd_cfg.mac_ssbd_max_backoffs} "
        f"macSsbdMinBf={ssbd_cfg.mac_ssbd_min_bf} "
        f"macSsbdMaxBf={ssbd_cfg.mac_ssbd_max_bf} "
        f"macSsbdUnitBackoffPeriod_us={ssbd_cfg.mac_ssbd_unit_backoff_ms*1000.0:.3f} "
        f"phyCcaDuration_ms={ssbd_cfg.phy_cca_duration_ms:.6f} "
        f"phyCcaEdThreshold_dBm={ssbd_cfg.phy_cca_ed_threshold_dbm:.2f} "
        f"cca_mode={ssbd_cfg.cca_mode}"
    )
    print(
        "[nb-switch-config] "
        f"enable={int(1 if (nb_switch_cfg.enable_switching and len(nb_switch_cfg.allow_list)>1) else 0)} "
        f"channel_switching_field={int(nb_switch_cfg.channel_switching_field)} "
        f"allow_list_len={len(nb_switch_cfg.allow_list)} seed={nb_switch_cfg.mms_prng_seed} "
        f"allow_min={min(nb_switch_cfg.allow_list)} allow_max={max(nb_switch_cfg.allow_list)} "
        f"spacing_mhz={nb_switch_cfg.nb_channel_spacing_mhz:.3f}"
    )
    if int(args.dump_nb_channel_seq) > 0:
        n_dump = int(args.dump_nb_channel_seq)
        seq_i = [
            int(selected_nb_channel_for_block(nb_switch_cfg, i))
            for i in range(n_dump)
        ]
        seq_r = [
            int(selected_nb_channel_for_block(nb_switch_cfg, i))
            for i in range(n_dump)
        ]
        if seq_i != seq_r:
            raise RuntimeError("NB channel sequence mismatch initiator/responder")
        print(f"[nb-switch-seq] first_{n_dump}={seq_i}")

    scenarios: list[dict] = []
    if args.sweep_mode == "grid":
        for mode in modes:
            offs = [None] if mode == "off" else wifi_offsets
            dens = [0.0] if mode == "off" else wifi_density_list
            for d in distances:
                for ch in uwb_channels:
                    for off in offs:
                        for den in dens:
                            scenarios.append(
                                {
                                    "sweep_mode": "grid",
                                    "vary_param": "",
                                    "vary_value": "",
                                    "wifi_mode": mode,
                                    "distance_m": float(d),
                                    "uwb_channel": int(ch),
                                    "wifi_offset_mhz": None if off is None else float(off),
                                    "wifi_density": float(den),
                                    "baseline_dist": float(args.baseline_dist),
                                    "baseline_uwb_ch": int(args.baseline_uwb_ch),
                                    "baseline_wifi_offset_mhz": float(args.baseline_wifi_offset_mhz),
                                    "baseline_wifi_density": float(args.baseline_wifi_density),
                                }
                            )
    else:
        if args.vary is None:
            raise ValueError("--one_factor mode requires --vary")
        mode = modes[0]
        base = {
            "distance_m": float(args.baseline_dist),
            "uwb_channel": int(args.baseline_uwb_ch),
            "wifi_offset_mhz": float(args.baseline_wifi_offset_mhz),
            "wifi_density": float(args.baseline_wifi_density),
        }
        if args.vary == "dist":
            vals = distances
        elif args.vary == "uwb_ch":
            vals = [int(v) for v in uwb_channels]
        elif args.vary == "wifi_offset_mhz":
            vals = wifi_offsets
        else:
            vals = wifi_density_list

        for v in vals:
            cfgv = dict(base)
            key = {
                "dist": "distance_m",
                "uwb_ch": "uwb_channel",
                "wifi_offset_mhz": "wifi_offset_mhz",
                "wifi_density": "wifi_density",
            }[args.vary]
            cfgv[key] = float(v) if key != "uwb_channel" else int(v)
            scenarios.append(
                {
                    "sweep_mode": "one_factor",
                    "vary_param": args.vary,
                    "vary_value": str(v),
                    "wifi_mode": mode,
                    "distance_m": float(cfgv["distance_m"]),
                    "uwb_channel": int(cfgv["uwb_channel"]),
                    "wifi_offset_mhz": None if mode == "off" else float(cfgv["wifi_offset_mhz"]),
                    "wifi_density": float(0.0 if mode == "off" else cfgv["wifi_density"]),
                    "baseline_dist": float(base["distance_m"]),
                    "baseline_uwb_ch": int(base["uwb_channel"]),
                    "baseline_wifi_offset_mhz": float(base["wifi_offset_mhz"]),
                    "baseline_wifi_density": float(base["wifi_density"]),
                }
            )

    rows: list[dict] = []
    total = len(scenarios) * trials
    done = 0
    progress_every = int(args.progress_every) if args.progress_every > 0 else max(1, total // 20)

    for s_idx, sc in enumerate(scenarios):
        succ_count = 0
        max_trials_for_s = trials
        # optional pilot gating to skip obviously-zero-success conditions cheaply
        pilot_trials = int(args.pilot_trials)
        if pilot_trials > 0:
            pilot_succ = 0
            for t0 in range(min(pilot_trials, trials)):
                seed0 = int(args.seed + t0) if args.paired_trials else int(args.seed + 700000 * (s_idx + 1) + t0)
                sp0 = None
                if sc["wifi_mode"] == "dense":
                    if args.wifi_model == "spatial":
                        sp0 = WiFiSpatialModel(
                            WiFiSpatialConfig(
                                area_size_m=float(args.area_size_m),
                                n_ap=int(args.n_ap),
                                n_sta_per_ap=int(args.n_sta_per_ap),
                                ap_tx_power_dbm=float(args.ap_tx_power_dbm),
                                duty_cycle=float(sc["wifi_density"]),
                                seed=seed0 + 77,
                            )
                        )
                    elif args.wifi_model == "layout":
                        lc = _scaled_layout_cfg(
                            layout_cfg,
                            seed=seed0 + 77,
                            duty_scale=float(sc["wifi_density"]),
                        )
                        if lc is not None:
                            sp0 = WiFiSpatialModel(lc)
                prow = _run_trial(
                    cfg_base=cfg_base,
                    wifi_mode=str(sc["wifi_mode"]),
                    wifi_density=float(sc["wifi_density"]),
                    distance_m=float(sc["distance_m"]),
                    uwb_channel=int(sc["uwb_channel"]),
                    wifi_offset_mhz=sc["wifi_offset_mhz"],
                    trial_idx=t0,
                    seed=seed0,
                    max_attempts=min(int(args.max_attempts), int(std_params.nb_retry_limit)),
                    until_success=bool(args.until_success),
                    max_trial_ms=float(args.max_trial_ms),
                    uwb_shots_per_session=int(args.uwb_shots_per_session),
                    require_k_successes=int(require_k),
                    aggregation=str(args.aggregation),
                    uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
                    wifi_model=str(args.wifi_model),
                    spatial_model=sp0,
                    nb_lbt_slot_ms=float(std_params.nb_lbt_slot_ms),
                    nb_lbt_cca_slots=int(std_params.nb_lbt_cca_slots),
                    ssbd_cfg=ssbd_cfg,
                    ssbd_debug=bool(args.ssbd_debug),
                    print_ssbd_trace=bool(args.print_ssbd_trace),
                    nb_switch_cfg=nb_switch_cfg,
                    enable_report_phase_model=bool(int(args.enable_report_phase_model)),
                    initiator_report_request=bool(int(args.initiator_report_request)),
                    responder_report_request=bool(int(args.responder_report_request)),
                    mms1st_report_nslots=int(args.mms1stReportNSlots),
                    mms2nd_report_nslots=int(args.mms2ndReportNSlots),
                    assume_oob_report_on_missing=bool(int(args.assume_oob_report_on_missing)),
                )
                pilot_succ += 1 if prow["success"] else 0
            if pilot_succ == 0 and trials > pilot_trials:
                print(f"[pilot-skip] scenario#{s_idx} predicted zero success from {pilot_trials} pilot trials; skipping rest")
                max_trials_for_s = pilot_trials
        for t in range(trials):
            if t >= max_trials_for_s:
                break
            # paired trials: same trial index gets same seed across vary values.
            if args.paired_trials:
                trial_seed = int(args.seed + t)
            else:
                trial_seed = int(args.seed + 100000 * (s_idx + 1) + t)
            spatial_model = None
            if sc["wifi_mode"] == "dense":
                if args.wifi_model == "spatial":
                    spatial_model = WiFiSpatialModel(
                        WiFiSpatialConfig(
                            area_size_m=float(args.area_size_m),
                            n_ap=int(args.n_ap),
                            n_sta_per_ap=int(args.n_sta_per_ap),
                            ap_tx_power_dbm=float(args.ap_tx_power_dbm),
                            duty_cycle=float(sc["wifi_density"]),
                            seed=trial_seed + 77,
                        )
                    )
                elif args.wifi_model == "layout":
                    lc = _scaled_layout_cfg(
                        layout_cfg,
                        seed=trial_seed + 77,
                        duty_scale=float(sc["wifi_density"]),
                    )
                    if lc is not None:
                        spatial_model = WiFiSpatialModel(lc)

            row = _run_trial(
                cfg_base=cfg_base,
                wifi_mode=str(sc["wifi_mode"]),
                wifi_density=float(sc["wifi_density"]),
                distance_m=float(sc["distance_m"]),
                uwb_channel=int(sc["uwb_channel"]),
                wifi_offset_mhz=sc["wifi_offset_mhz"],
                trial_idx=t,
                seed=trial_seed,
                max_attempts=min(int(args.max_attempts), int(std_params.nb_retry_limit)),
                until_success=bool(args.until_success),
                max_trial_ms=float(args.max_trial_ms),
                uwb_shots_per_session=int(args.uwb_shots_per_session or std_params.uwb_shots_per_session),
                require_k_successes=int(require_k),
                aggregation=str(args.aggregation),
                uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
                wifi_model=str(args.wifi_model),
                spatial_model=spatial_model,
                nb_lbt_slot_ms=float(std_params.nb_lbt_slot_ms),
                nb_lbt_cca_slots=int(std_params.nb_lbt_cca_slots),
                ssbd_cfg=ssbd_cfg,
                ssbd_debug=bool(args.ssbd_debug),
                print_ssbd_trace=bool(args.print_ssbd_trace),
                nb_switch_cfg=nb_switch_cfg,
                enable_report_phase_model=bool(int(args.enable_report_phase_model)),
                initiator_report_request=bool(int(args.initiator_report_request)),
                responder_report_request=bool(int(args.responder_report_request)),
                mms1st_report_nslots=int(args.mms1stReportNSlots),
                mms2nd_report_nslots=int(args.mms2ndReportNSlots),
                assume_oob_report_on_missing=bool(int(args.assume_oob_report_on_missing)),
            )
            if args.timing_log:
                print(
                    f"[timing] scenario={sc['wifi_mode']} d={sc['distance_m']:.1f} "
                    f"u{sc['uwb_channel']} den={sc['wifi_density']:.3f} "
                    f"trial={t+1}/{trials} runtime={float(row.get('sim_runtime_s', float('nan'))):.3f}s "
                    f"success={bool(row['success'])} attempts={int(row['attempts_used'])}"
                )
            row.update(
                {
                    "sweep_mode": sc["sweep_mode"],
                    "vary_param": sc["vary_param"],
                    "vary_value": sc["vary_value"],
                    "baseline_dist": sc["baseline_dist"],
                    "baseline_uwb_ch": sc["baseline_uwb_ch"],
                    "baseline_wifi_offset_mhz": sc["baseline_wifi_offset_mhz"],
                    "baseline_wifi_density": sc["baseline_wifi_density"],
                }
            )
            rows.append(row)
            done += 1
            succ_count += 1 if row["success"] else 0

            if done == 1 or done % progress_every == 0 or done == total:
                print(
                    f"[{done:5d}/{total:5d}] mode={sc['wifi_mode']} d={sc['distance_m']:.1f}m "
                    f"uwb_ch={sc['uwb_channel']} off={sc['wifi_offset_mhz']} trial={t} "
                    f"success={row['success']} lat={row['latency_to_success_ms']}"
                )

            # Conservative early stop for zero-success groups in one-factor.
            if (
                args.sweep_mode == "one_factor"
                and args.early_stop_zero_success
                and (t + 1) >= int(args.early_stop_min_trials)
                and succ_count == 0
            ):
                ub95 = 3.0 / float(t + 1)  # rule-of-three upper bound
                if ub95 <= float(args.early_stop_max_sr):
                    max_trials_for_s = t + 1
                    break

        if max_trials_for_s < trials:
            # Fill skipped trials with timeout/censored rows.
            for t2 in range(max_trials_for_s, trials):
                filler_seed = int(args.seed + t2) if args.paired_trials else int(args.seed + 100000 * (s_idx + 1) + t2)
                rows.append(
                    {
                        "scenario": f"{sc['wifi_mode']}_latency",
                        "wifi_model": str(args.wifi_model),
                        "enable_nb_channel_switching": int(
                            1 if (nb_switch_cfg.enable_switching and len(nb_switch_cfg.allow_list) > 1) else 0
                        ),
                        "nb_channel_switching_field": int(nb_switch_cfg.channel_switching_field),
                        "mac_mms_prng_seed": int(nb_switch_cfg.mms_prng_seed),
                        "mmsNbChannelAllowList_json": json.dumps(list(nb_switch_cfg.allow_list), separators=(",", ":")),
                        "wifi_mode": sc["wifi_mode"],
                        "wifi_density": sc["wifi_density"],
                        "distance_m": sc["distance_m"],
                        "dist_m": sc["distance_m"],
                        "uwb_channel": sc["uwb_channel"],
                        "uwb_ch": sc["uwb_channel"],
                        "nb_center_hz": float(cfg_base.nb_center_override_hz),
                        "uwb_center_hz": float(uwb_center_freq_hz(int(sc["uwb_channel"]))),
                        "wifi_center_hz": float("nan"),
                        "wifi_offset_mhz_req": float(sc["wifi_offset_mhz"]) if sc["wifi_offset_mhz"] is not None else float("nan"),
                        "wifi_offset_mhz": float(sc["wifi_offset_mhz"]) if sc["wifi_offset_mhz"] is not None else float("nan"),
                        "n_ap": int(args.n_ap),
                        "area_size_m": float(args.area_size_m),
                        "latency_to_uwb_success_ms": float("nan"),
                        "latency_to_conf_done_ms": float("nan"),
                        "latency_to_success_ms": None,
                        "attempts": 0,
                        "attempts_used": 0,
                        "retries_nb": 0,
                        "retries_uwb": 0,
                        "nb_handshake_ms": 0.0,
                        "uwb_ranging_ms": 0.0,
                        "waiting_ms": 0.0,
                        "backoff_ms": 0.0,
                        "nb_lbt_busy_events": 0,
                        "nb_lbt_wait_ms": 0.0,
                        "nb_backoff_slots": 0,
                        "nb_tx_attempts": 0,
                        "nb_access_fail": False,
                        "nb_ssbd_total_deferral_ms": 0.0,
                        "nb_ssbd_total_cca_ms": 0.0,
                        "nb_ssbd_cca_count": 0,
                        "nb_ssbd_busy_count": 0,
                        "nb_ssbd_nb_count": 0,
                        "nb_ssbd_bf_final": 0,
                        "nb_ssbd_last_reason": "pilot_skipped",
                        "wifi_cca_busy_events": 0,
                        "wifi_cca_wait_ms": 0.0,
                        "wifi_backoff_slots": 0,
                        "wifi_tx_occupancy_time": 0.0,
                        "wifi_busy_frac_nb": float("nan"),
                        "wifi_busy_frac_uwb": float("nan"),
                        "wifi_realized_duty_mean": float("nan"),
                        "wifi_realized_duty_min": float("nan"),
                        "wifi_realized_duty_max": float("nan"),
                        "nb_cca_busy_rate": float("nan"),
                        "time_spent_ms": float(args.max_trial_ms),
                        "success": False,
                        "seed": filler_seed,
                        "trial_index": t2,
                        "ber": float("nan"),
                        "fer": float("nan"),
                        "snr_db": float("nan"),
                        "ranging_fail": float("nan"),
                        "control_ok_last": False,
                        "uwb_shots_per_session": int(args.uwb_shots_per_session),
                        "require_k_successes": int(require_k),
                        "aggregation": str(args.aggregation),
                        "range_result_m": float("nan"),
                        "t_nb_adv_ms": float("nan"),
                        "t_nb_poll_ms": float("nan"),
                        "t_nb_resp_ms": float("nan"),
                        "t_nb_conf_ms": float("nan"),
                        "t_nb_ifs_total_ms": float("nan"),
                        "t_uwb_airtime_ms": float("nan"),
                        "t_uwb_reply_delay_ms": float("nan"),
                        "event_trace_json": "[]",
                        "ssbd_attempts_json": "[]",
                        "nb_selected_channel": int(selected_nb_channel_for_block(nb_switch_cfg, 0)),
                        "ranging_block_index": 0,
                        "nb_selected_channel_seq_json": "[]",
                        "nb_channel_seq_first8_json": "[]",
                        "ssbd_nb": 0,
                        "ssbd_bf": 0,
                        "sim_runtime_s": 0.0,
                        "terminal_fail_reason": "pilot_skipped",
                        "last_attempt_stage": "none",
                        "last_attempt_nb_channel": int(selected_nb_channel_for_block(nb_switch_cfg, 0)),
                        "last_attempt_uwb_channel": int(sc["uwb_channel"]),
                        "count_nb_ssbd_timeout": 0,
                        "count_nb_control_decode_fail": 0,
                        "count_uwb_frame_fail_snr": 0,
                        "count_uwb_frame_fail_interference": 0,
                        "count_cca_busy": 0,
                        "fail_reason": "pilot_skipped",
                        "sweep_mode": sc["sweep_mode"],
                        "vary_param": sc["vary_param"],
                        "vary_value": sc["vary_value"],
                        "baseline_dist": sc["baseline_dist"],
                        "baseline_uwb_ch": sc["baseline_uwb_ch"],
                        "baseline_wifi_offset_mhz": sc["baseline_wifi_offset_mhz"],
                        "baseline_wifi_density": sc["baseline_wifi_density"],
                    }
                )
                done += 1

    fieldnames = [
        "scenario",
        "wifi_model",
        "enable_nb_channel_switching",
        "nb_channel_switching_field",
        "mac_mms_prng_seed",
        "mmsNbInitChannel",
        "mmsNbChannelAllowList_json",
        "nb_phase_rule_init",
        "nb_phase_rule_ctrl_report",
        "sweep_mode",
        "vary_param",
        "vary_value",
        "wifi_mode",
        "wifi_density",
        "distance_m",
        "dist_m",
        "uwb_channel",
        "uwb_ch",
        "nb_center_hz",
        "uwb_center_hz",
        "uwb_sig_ref_dbm_layout",
        "wifi_center_hz",
        "wifi_offset_mhz_req",
        "wifi_offset_mhz",
        "n_ap",
        "area_size_m",
        "baseline_dist",
        "baseline_uwb_ch",
        "baseline_wifi_offset_mhz",
        "baseline_wifi_density",
        "latency_to_uwb_success_ms",
        "latency_to_conf_done_ms",
        "latency_to_success_ms",
        "attempts",
        "attempts_used",
        "success",
        "seed",
        "trial_index",
        "retries_nb",
        "retries_uwb",
        "nb_handshake_ms",
        "uwb_ranging_ms",
        "waiting_ms",
        "backoff_ms",
        "nb_lbt_busy_events",
        "nb_lbt_wait_ms",
        "nb_backoff_slots",
        "nb_tx_attempts",
        "nb_access_fail",
        "nb_ssbd_total_deferral_ms",
        "nb_ssbd_total_cca_ms",
        "nb_ssbd_cca_count",
        "nb_ssbd_busy_count",
        "nb_ssbd_nb_count",
        "nb_ssbd_bf_final",
        "nb_ssbd_last_reason",
        "nb_selected_channel",
        "ranging_block_index",
        "nb_selected_channel_seq_json",
        "nb_channel_seq_first8_json",
        "nb_ch_init",
        "nb_ch_init_seq_first8_json",
        "nb_ch_ctrl_seq_first8_json",
        "nb_ch_report_seq_first8_json",
        "nb_ch_all_seq_first8_json",
        "nb_ch_unique_count_init",
        "nb_ch_unique_count_ctrl",
        "nb_ch_unique_count_report",
        "ssbd_nb",
        "ssbd_bf",
        "sim_runtime_s",
        "terminal_fail_reason",
        "last_attempt_stage",
        "last_attempt_nb_channel",
        "last_attempt_uwb_channel",
        "count_nb_ssbd_timeout",
        "count_nb_control_decode_fail",
        "count_uwb_frame_fail_snr",
        "count_uwb_frame_fail_interference",
        "count_cca_busy",
        "nb_channel_match",
        "nb_channel_switching_enabled",
        "nb_allow_list_len",
        "nb_allow_list_min",
        "wifi_cca_busy_events",
        "wifi_cca_wait_ms",
        "wifi_backoff_slots",
        "wifi_tx_occupancy_time",
        "wifi_busy_frac_nb",
        "wifi_busy_frac_uwb",
        "wifi_realized_duty_mean",
        "wifi_realized_duty_min",
        "wifi_realized_duty_max",
        "nb_cca_busy_rate",
        "time_spent_ms",
        "uwb_shots_per_session",
        "require_k_successes",
        "aggregation",
        "range_result_m",
        "t_nb_adv_ms",
        "t_nb_poll_ms",
        "t_nb_resp_ms",
        "t_nb_conf_ms",
        "t_nb_ifs_total_ms",
        "t_uwb_airtime_ms",
        "t_uwb_reply_delay_ms",
        "event_trace_json",
        "ssbd_attempts_json",
        "fail_reason",
        "snr_db",
        "ber",
        "fer",
        "ranging_fail",
        "first_path_index",
        "peak_index",
        "detection_threshold_abs",
        "estimated_tof_ns",
        "sample_period_ns",
        "applied_calibration_offset_ns",
        "first_path_thr_db",
        "first_path_peak_frac",
        "first_path_noise_floor",
        "first_path_thr_noise_abs",
        "first_path_thr_peak_abs",
        "first_path_snr_corr_db",
        "first_path_peak_ratio_db",
        "noise_win_start",
        "noise_win_end",
        "first_path_fallback_rate",
        "a2b1_rstu",
        "b2a1_rstu",
        "a2b2_rstu",
        "b2a2_rstu",
        "ds_ra_rstu",
        "ds_rb_rstu",
        "ds_da_rstu",
        "ds_db_rstu",
        "ds_tof_rstu",
        "control_ok_last",
        "report_enabled",
        "report_required_responder_to_initiator",
        "report_required_initiator_to_responder",
        "report_tx_attempted_R2I",
        "report_rx_ok_R2I",
        "report_tx_attempted_I2R",
        "report_rx_ok_I2R",
        "t_report_start_ms",
        "t_report1_tx_ms",
        "t_report1_rx_done_ms",
        "t_report2_tx_ms",
        "t_report2_rx_done_ms",
        "nb_report_txrx_ms",
        "t0_trial_start_ms",
        "t_init_done_ms",
        "t_scan_start_ms",
        "t_adv_start_ms",
        "t_adv_rx_detected_ms",
        "t_cca_start_ms",
        "t_cca_done_ms",
        "t_backoff_done_ms",
        "t_uwb_tx_start_ms",
        "t_uwb_rx_done_ms",
        "t_toa_est_start_ms",
        "t_toa_est_done_ms",
        "t_aggregation_done_ms",
        "t_trial_end_ms",
        "lat_total_ms",
        "lat_adv_scan_wait_ms",
        "lat_lbt_cca_ms",
        "lat_backoff_ms",
        "lat_uwb_exchange_ms",
        "lat_processing_ms",
        "init_scan_wait_ms",
        "advertising_wait_ms",
        "nb_ssbd_cca_ms",
        "nb_ssbd_backoff_ms",
        "nb_ctrl_txrx_ms",
        "ctrl_to_ranging_gap_ms",
        "uwb_ranging_txrx_ms",
        "report_phase_ms",
        "misc_overhead_ms",
        "total_session_ms",
        "lat_breakdown_note",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved CSV: {out_csv}")
    _print_summary(rows)

    if args.sweep_mode == "one_factor" and args.vary is not None:
        baseline_value = {
            "dist": args.baseline_dist,
            "uwb_ch": args.baseline_uwb_ch,
            "wifi_offset_mhz": args.baseline_wifi_offset_mhz,
            "wifi_density": args.baseline_wifi_density,
        }[args.vary]
        _print_impact_report(rows, vary_param=args.vary, baseline_value=baseline_value)

    if args.plot:
        _maybe_plot(rows, out_csv)
        print("Saved plots (if matplotlib available).")
    if args.plot_density_study:
        _maybe_plot_density_study(rows, out_csv)
    if int(args.dump_timeline) > 0:
        _maybe_dump_timeline(rows, out_csv, n=int(args.dump_timeline), seed=int(args.seed))
        print(f"Saved up to {int(args.dump_timeline)} timeline PNG(s) near {out_csv}.")

    if args.print_timing_breakdown:
        # Run a deterministic, Wi-Fi-off, single-attempt reference to explain constant-latency components.
        ref_row = _run_trial(
            cfg_base=cfg_base,
            wifi_mode="off",
            wifi_model="occupancy",
            spatial_model=None,
            wifi_density=0.0,
            distance_m=float(args.baseline_dist),
            uwb_channel=int(args.baseline_uwb_ch),
            wifi_offset_mhz=None,
            trial_idx=0,
            seed=int(args.seed),
            max_attempts=1,
            until_success=False,
            max_trial_ms=float(args.max_trial_ms),
            nb_lbt_slot_ms=float(std_params.nb_lbt_slot_ms),
            nb_lbt_cca_slots=int(std_params.nb_lbt_cca_slots),
            ssbd_cfg=ssbd_cfg,
            ssbd_debug=False,
            print_ssbd_trace=False,
            nb_switch_cfg=nb_switch_cfg,
            uwb_shots_per_session=1,
            require_k_successes=1,
            aggregation="median",
            uwb_shot_gap_ms=float(args.uwb_shot_gap_ms),
            enable_report_phase_model=bool(int(args.enable_report_phase_model)),
            initiator_report_request=bool(int(args.initiator_report_request)),
            responder_report_request=bool(int(args.responder_report_request)),
            mms1st_report_nslots=int(args.mms1stReportNSlots),
            mms2nd_report_nslots=int(args.mms2ndReportNSlots),
            assume_oob_report_on_missing=bool(int(args.assume_oob_report_on_missing)),
        )
        _print_timing_breakdown(cfg_base, ref_row)


if __name__ == "__main__":
    main()
