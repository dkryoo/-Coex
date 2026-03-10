from __future__ import annotations

import argparse
import os
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    from Channel.Rician import apply_distance_rician_channel
    from Channel.Thermal_noise import add_thermal_noise_white
    from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
    from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Channel.Rician import apply_distance_rician_channel
    from Channel.Thermal_noise import add_thermal_noise_white
    from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
    from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
from packet.narrowband_compact_frames import (
    AdvConf,
    AdvPoll,
    AdvResp,
    CFID_ADV_CONF,
    CFID_ADV_RESP,
    SmidTlv,
    decode_adv_conf,
    decode_adv_poll,
    decode_adv_resp,
    encode_adv_conf,
    encode_adv_poll,
    encode_adv_resp,
)
from simulation.mms.performance import simulate_mms_performance


def wifi6_center_freq_hz(channel: int) -> float:
    # 6 GHz Wi-Fi center frequency formula (MHz): fc = 5950 + 5*ch
    return (5950.0 + 5.0 * float(channel)) * 1e6


def nb_center_freq_hz(channel: int, base_hz: float = 6489.0e6, spacing_hz: float = 2.0e6) -> float:
    # Assumed NB channelization (configurable): ch1=base, step=2 MHz.
    return float(base_hz + (int(channel) - 1) * spacing_hz)


def uwb_center_freq_hz(channel: int) -> float:
    # Common HRP-UWB channel centers (IEEE 802.15.4a style).
    ch_map = {
        5: 6489.6e6,
        7: 6489.6e6,
        8: 7488.0e6,
        9: 7987.2e6,
        10: 8486.4e6,
    }
    if channel not in ch_map:
        raise ValueError(f"Unsupported UWB channel {channel}. Supported: {list(ch_map.keys())}")
    return float(ch_map[channel])


def _load_wifi_tx_class():
    wifi_path = Path(__file__).resolve().parents[2] / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("wifi_tx_module_full_stack_demo", wifi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Wi-Fi TX module from {wifi_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WiFiOFDMTx


def _bytes_to_bits_lsb(data: bytes) -> np.ndarray:
    out = []
    for b in data:
        for k in range(8):
            out.append((b >> k) & 1)
    return np.asarray(out, dtype=int)


def _bits_lsb_to_bytes(bits: np.ndarray, n_bytes: int) -> bytes:
    bits = np.asarray(bits, dtype=int).flatten()
    need = n_bytes * 8
    if len(bits) < need:
        raise ValueError("Insufficient bits")
    bits = bits[:need]
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        v = 0
        for k in range(8):
            v |= (int(bits[i * 8 + k]) & 1) << k
        out[i] = v
    return bytes(out)


def _resample_complex_linear(x: np.ndarray, fs_in_hz: float, fs_out_hz: float) -> np.ndarray:
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("sample rates must be positive")
    if len(x) <= 1:
        return x.astype(np.complex128)
    t_in = np.arange(len(x), dtype=float) / fs_in_hz
    n_out = max(1, int(np.floor((len(x) - 1) * fs_out_hz / fs_in_hz)) + 1)
    t_out = np.arange(n_out, dtype=float) / fs_out_hz
    re = np.interp(t_out, t_in, np.real(x))
    im = np.interp(t_out, t_in, np.imag(x))
    return (re + 1j * im).astype(np.complex128)


def _spectral_overlap_ratio(fc_victim_hz: float, bw_victim_hz: float, fc_int_hz: float, bw_int_hz: float) -> float:
    v0, v1 = fc_victim_hz - bw_victim_hz / 2.0, fc_victim_hz + bw_victim_hz / 2.0
    i0, i1 = fc_int_hz - bw_int_hz / 2.0, fc_int_hz + bw_int_hz / 2.0
    ov = max(0.0, min(v1, i1) - max(v0, i0))
    return ov / max(bw_victim_hz, 1e-30)


def _build_wifi_interference_for_uwb(
    wifi_tx,
    cfg: FullStackConfig,
    wifi_fc_hz: float,
    length_samples_at_uwb_fs: int,
    uwb_fs_hz: float,
    seed: int,
) -> tuple[np.ndarray, float, float, dict]:
    dur_s = max(3e-4, length_samples_at_uwb_fs / uwb_fs_hz * 1.2)
    fs_wifi = cfg.wifi_bw_mhz * 1e6
    wf_wifi_tx, _ = wifi_tx.generate_for_target_rx_throughput(
        target_rx_throughput_mbps=800.0,
        duration_s=dur_s,
        channel_bw_mhz=cfg.wifi_bw_mhz,
        standard=cfg.wifi_standard,
        tx_power_dbw=cfg.wifi_tx_power_dbw,
        center_freq_hz=wifi_fc_hz,
    )
    wf_wifi_tx = wifi_tx.apply_tx_emission_mask(
        wf_wifi_tx,
        fs_hz=fs_wifi,
        channel_bw_hz=fs_wifi,
        aclr_db=cfg.wifi_aclr_db,
        seed=seed + 5,
    )

    wf_wifi_ch, _, _, _, _ = apply_distance_rician_channel(
        wf=wf_wifi_tx,
        fs_hz=fs_wifi,
        fc_hz=wifi_fc_hz,
        distance_m=cfg.wifi_to_uwb_rx_distance_m,
        pathloss_exp=2.0,
        delays_s=(0.0, 30e-9, 80e-9),
        powers_db=(0.0, -6.0, -10.0),
        k_factor_db=6.0,
        include_toa=True,
        seed=seed + 6000,
    )
    return wf_wifi_ch.astype(np.complex128), float(fs_wifi), float(wifi_fc_hz), {}


@dataclass
class FullStackConfig:
    distance_m: float = 20.0
    nb_channel: int = 1
    uwb_channel: int = 5
    wifi_channel: int = 37
    wifi_bw_mhz: int = 160
    wifi_standard: str = "wifi7"  # wifi6e or wifi7
    nb_center_override_hz: float | None = 6.4890e9
    nb_eirp_dbw: float = -16.0
    wifi_tx_power_dbw: float = -20.0
    wifi_to_uwb_rx_distance_m: float = 2.0
    wifi_aclr_db: float = 35.0
    rx_stop_db: float = 25.0
    enable_agc: bool = False
    agc_stage: str = "post_selectivity"
    agc_target_dbfs: float = -12.0
    agc_min_gain_db: float = -60.0
    agc_max_gain_db: float = 60.0
    toa_refine_method: str = "fft_upsample"
    corr_upsample: int = 8
    corr_win: int = 64
    first_path: bool = True
    first_path_thr_db: float = 13.0
    first_path_peak_frac: float | None = 0.16
    fp_use_adaptive_thr: bool = True
    fp_snr_switch_db: float = 12.0
    fp_thr_noise_cap_mult: float = 2.5
    fp_thr_min_floor_mult: float = 3.0
    first_path_search_back: int = 8
    first_path_persist: int = 3
    first_path_local_win: int = 8
    range_bias_correction_m: float = 0.0
    auto_bias_calibrate: bool = False
    auto_range_bias_calibrate: bool = False
    toa_calibration_override: float | None = None
    toa_calibration_use_runtime_channel: bool = False
    lna_p1db_dbm: float | None = None
    lna_max_gain_db: float = 0.0
    adc_clip_dbfs: float | None = None
    quant_bits: int | None = None
    wifi_oob_atten_db: float | None = None
    nf_db: float = 6.0
    temperature_k: float = 290.0
    distance_adaptive_multipath: bool = False
    channel_delays_s: tuple[float, ...] = (0.0, 4e-9, 8e-9)
    channel_powers_db: tuple[float, ...] = (0.0, -10.0, -16.0)
    channel_k_factor_db: float = 12.0
    channel_delay_jitter_std_ns: float = 0.0
    channel_power_jitter_std_db: float = 0.0
    seed: int = 1234


def _nb_phy_send_and_receive(
    tx_bytes: bytes,
    tx_nb: OQPSK_SF32_Tx,
    rx_nb: OQPSK_SF32_Rx,
    nb_fc_hz: float,
    cfg: FullStackConfig,
    wifi_on: bool,
    wifi_tx,
    wifi_fc_hz: float,
    nb_wifi_overlap: float,
    step_seed: int,
) -> bytes | None:
    tx_bits = _bytes_to_bits_lsb(tx_bytes)
    tx_wf, _, _ = tx_nb.build_tx_waveform(
        psdu_bits=tx_bits,
        tx_eirp_db=cfg.nb_eirp_dbw,
        regulatory_profile="unlicensed_6g_lpi_ap",
    )
    nb_ch, _, _, _, _ = apply_distance_rician_channel(
        wf=tx_wf,
        fs_hz=tx_nb.fs,
        fc_hz=nb_fc_hz,
        distance_m=cfg.distance_m,
        pathloss_exp=2.0,
        delays_s=(0.0, 50e-9, 120e-9),
        powers_db=(0.0, -6.0, -10.0),
        k_factor_db=8.0,
        include_toa=True,
        seed=step_seed + 10,
    )
    rx_mix = np.concatenate([np.zeros(80, dtype=np.complex128), nb_ch])

    if wifi_on:
        dur_s = max(0.001, len(rx_mix) / tx_nb.fs)
        wf_wifi, _ = wifi_tx.generate_for_target_rx_throughput(
            target_rx_throughput_mbps=400.0,
            duration_s=dur_s,
            channel_bw_mhz=cfg.wifi_bw_mhz,
            standard=cfg.wifi_standard,
            tx_power_dbw=cfg.wifi_tx_power_dbw,
            center_freq_hz=wifi_fc_hz,
        )
        wf_wifi_rs = _resample_complex_linear(wf_wifi, fs_in_hz=cfg.wifi_bw_mhz * 1e6, fs_out_hz=tx_nb.fs)
        if len(wf_wifi_rs) < len(rx_mix):
            wf_wifi_rs = np.pad(wf_wifi_rs, (0, len(rx_mix) - len(wf_wifi_rs)))
        wf_wifi_rs = wf_wifi_rs[: len(rx_mix)]
        t = np.arange(len(wf_wifi_rs), dtype=float) / tx_nb.fs
        f_off = wifi_fc_hz - nb_fc_hz
        wf_wifi_shift = wf_wifi_rs * np.exp(1j * 2.0 * np.pi * f_off * t)
        # In-band overlap dominates; keep small leakage floor for out-of-band coupling.
        leakage_floor = 10.0 ** (-35.0 / 10.0)
        nb_intf_power_scale = max(float(nb_wifi_overlap), leakage_floor)
        wf_wifi_shift = wf_wifi_shift * np.sqrt(nb_intf_power_scale)
        wf_wifi_ch, _, _, _, _ = apply_distance_rician_channel(
            wf=wf_wifi_shift,
            fs_hz=tx_nb.fs,
            fc_hz=wifi_fc_hz,
            distance_m=cfg.distance_m,
            pathloss_exp=2.0,
            delays_s=(0.0, 30e-9, 80e-9),
            powers_db=(0.0, -6.0, -10.0),
            k_factor_db=6.0,
            include_toa=True,
            seed=step_seed + 20,
        )
        wf_wifi_ch = np.concatenate([np.zeros(80, dtype=np.complex128), wf_wifi_ch])
        if len(wf_wifi_ch) < len(rx_mix):
            wf_wifi_ch = np.pad(wf_wifi_ch, (0, len(rx_mix) - len(wf_wifi_ch)))
        rx_mix = rx_mix + wf_wifi_ch[: len(rx_mix)]

    rx_wf = add_thermal_noise_white(
        wf=rx_mix,
        fs_hz=tx_nb.fs,
        nf_db=cfg.nf_db,
        temperature_k=cfg.temperature_k,
        seed=step_seed + 30,
    )
    try:
        rx_bits, _, _ = rx_nb.decode(rx_wf, tx_fir=None, verbose=False)
        return _bits_lsb_to_bytes(rx_bits, n_bytes=len(tx_bytes))
    except Exception:
        return None


def run_full_stack_case(
    cfg: FullStackConfig,
    wifi_on: bool,
    wifi_channel_override: int | None = None,
    rx_selectivity_override: dict | None = None,
    baseline_sanity_mode: bool = False,
    case_tag: str = "case",
    n_trials: int = 200,
    debug_first_trial: bool = True,
    save_psd: bool = True,
    skip_nb_control: bool = False,
) -> dict:
    nb_fc_hz = float(cfg.nb_center_override_hz) if cfg.nb_center_override_hz is not None else nb_center_freq_hz(cfg.nb_channel)
    uwb_fc_hz = uwb_center_freq_hz(cfg.uwb_channel)
    wifi_ch = int(cfg.wifi_channel if wifi_channel_override is None else wifi_channel_override)
    wifi_fc_hz = wifi6_center_freq_hz(wifi_ch)
    wifi_bw_hz = cfg.wifi_bw_mhz * 1e6
    nb_bw_hz = 2.0e6
    uwb_bw_hz = 499.2e6
    nb_wifi_overlap = _spectral_overlap_ratio(nb_fc_hz, nb_bw_hz, wifi_fc_hz, wifi_bw_hz)
    uwb_wifi_overlap = _spectral_overlap_ratio(uwb_fc_hz, uwb_bw_hz, wifi_fc_hz, wifi_bw_hz)

    tx_nb = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    rx_nb = OQPSK_SF32_Rx(chip_rate_hz=2e6, osr=8)
    WiFiOFDMTx = _load_wifi_tx_class()
    wifi_tx = WiFiOFDMTx(rng_seed=cfg.seed, center_freq_hz=wifi_fc_hz)
    uwb_fs_hz = 499.2e6

    control_ok = bool(skip_nb_control)
    nb_poll_attempts = 0
    if not skip_nb_control:
        # NB initiation/control frames
        poll = AdvPoll(
            init_slot_dur=10,
            cap_dur=8,
            supported_mod_modes=0x03,
            smid_tlvs=[SmidTlv(tag=CFID_ADV_RESP, values=b"\x00"), SmidTlv(tag=CFID_ADV_CONF, values=b"\x00")],
        )
        poll_b = encode_adv_poll(poll)

        resp = AdvResp(
            rpa_hash=b"\xAA\xBB\xCC",
            message_id=0x42,
            nb_full_channel_map=b"\x01\x02\x03\x04\x05\x06",
            mgmt_phy_cfg=0x21,
            mgmt_mac_cfg=b"\x10\x11\x12\x13\x14\x15\x16",
            ranging_phy_cfg=b"\x20\x21\x22\x23",
            mms_num_frags=2,
        )
        resp_b = encode_adv_resp(resp)

        conf = AdvConf(
            rpa_hash=b"\x11\x22\x33",
            message_id=0x42,
            responder_addr=b"\xAA\xBB\xCC",
            sor_time_offset=b"\x01\x02\x03",
        )
        conf_b = encode_adv_conf(conf)

        # Try with retries on poll->resp leg.
        for r in range(1, 6):
            nb_poll_attempts = r
            rx_poll_at_responder = _nb_phy_send_and_receive(
                tx_bytes=poll_b,
                tx_nb=tx_nb,
                rx_nb=rx_nb,
                nb_fc_hz=nb_fc_hz,
                cfg=cfg,
                wifi_on=wifi_on,
                wifi_tx=wifi_tx,
                wifi_fc_hz=wifi_fc_hz,
                nb_wifi_overlap=nb_wifi_overlap,
                step_seed=cfg.seed + 1000 * r,
            )
            if rx_poll_at_responder is None:
                continue
            try:
                _ = decode_adv_poll(rx_poll_at_responder)
            except Exception:
                continue

            rx_resp_at_initiator = _nb_phy_send_and_receive(
                tx_bytes=resp_b,
                tx_nb=tx_nb,
                rx_nb=rx_nb,
                nb_fc_hz=nb_fc_hz,
                cfg=cfg,
                wifi_on=wifi_on,
                wifi_tx=wifi_tx,
                wifi_fc_hz=wifi_fc_hz,
                nb_wifi_overlap=nb_wifi_overlap,
                step_seed=cfg.seed + 2000 * r,
            )
            if rx_resp_at_initiator is None:
                continue
            try:
                _ = decode_adv_resp(rx_resp_at_initiator)
            except Exception:
                continue

            rx_conf_at_responder = _nb_phy_send_and_receive(
                tx_bytes=conf_b,
                tx_nb=tx_nb,
                rx_nb=rx_nb,
                nb_fc_hz=nb_fc_hz,
                cfg=cfg,
                wifi_on=wifi_on,
                wifi_tx=wifi_tx,
                wifi_fc_hz=wifi_fc_hz,
                nb_wifi_overlap=nb_wifi_overlap,
                step_seed=cfg.seed + 3000 * r,
            )
            if rx_conf_at_responder is None:
                continue
            try:
                _ = decode_adv_conf(rx_conf_at_responder)
                control_ok = True
                break
            except Exception:
                continue

    interference_wf = None
    interference_fs_hz = None
    interference_fc_hz = None
    interference_block_dbw = None
    wifi_oob_profile = None
    if wifi_on:
        wifi_oob_atten_db = cfg.wifi_oob_atten_db
        if wifi_oob_atten_db is None:
            # Simple ACLR-like OOB profile vs RF offset.
            def wifi_oob_atten_db(abs_f_off_hz: float) -> float:
                if abs_f_off_hz <= (cfg.wifi_bw_mhz * 1e6) / 2.0:
                    return 0.0
                if abs_f_off_hz <= 300e6:
                    return 12.0
                return 20.0
        wifi_oob_profile = wifi_oob_atten_db
        interference_wf, interference_fs_hz, interference_fc_hz, _ = _build_wifi_interference_for_uwb(
            wifi_tx=wifi_tx,
            cfg=cfg,
            wifi_fc_hz=wifi_fc_hz,
            length_samples_at_uwb_fs=160_000,
            uwb_fs_hz=uwb_fs_hz,
            seed=cfg.seed,
        )

    # UWB MMS ranging phase performance (initiation/control assumed complete)
    perf = simulate_mms_performance(
        distances_m=(cfg.distance_m,),
        n_trials=n_trials,
        rif_payload_bits=256,
        fc_hz=uwb_fc_hz,
        uwb_fs_hz=uwb_fs_hz,
        uwb_fc_hz=uwb_fc_hz,
        tx_eirp_dbw=cfg.nb_eirp_dbw,
        nf_db=cfg.nf_db,
        temperature_k=cfg.temperature_k,
        seed=cfg.seed + 9000,
        detector_mode="first_path",
        quality_min_db=8.0,
        toa_refine_method=cfg.toa_refine_method,
        corr_upsample=cfg.corr_upsample,
        corr_win=cfg.corr_win,
        first_path=cfg.first_path,
        first_path_thr_db=cfg.first_path_thr_db,
        first_path_peak_frac=cfg.first_path_peak_frac,
        fp_use_adaptive_thr=cfg.fp_use_adaptive_thr,
        fp_snr_switch_db=cfg.fp_snr_switch_db,
        fp_thr_noise_cap_mult=cfg.fp_thr_noise_cap_mult,
        fp_thr_min_floor_mult=cfg.fp_thr_min_floor_mult,
        first_path_search_back=cfg.first_path_search_back,
        first_path_persist=cfg.first_path_persist,
        first_path_local_win=cfg.first_path_local_win,
        interference_wf=interference_wf,
        interference_fs_hz=interference_fs_hz,
        interference_fc_hz=interference_fc_hz,
        interference_bw_hz=wifi_bw_hz if wifi_on else None,
        interference_block_dbw=interference_block_dbw,
        external_interference_penalty_db=0.0,
        wifi_interference_on=False,
        interference_aclr_db=cfg.wifi_aclr_db,
        use_leakage_equivalent_for_alias=False,
        wifi_params={"wifi_oob_atten_db": wifi_oob_profile},
        rx_selectivity={
            "pass_bw_hz": 120e6,
            "transition_hz": 60e6,
            "stop_atten_db": cfg.rx_stop_db,
            "taps": 257,
        } if rx_selectivity_override is None else rx_selectivity_override,
        enable_agc=cfg.enable_agc,
        agc_stage=cfg.agc_stage,
        agc_target_dbfs=cfg.agc_target_dbfs,
        agc_min_gain_db=cfg.agc_min_gain_db,
        agc_max_gain_db=cfg.agc_max_gain_db,
        lna_p1db_dbm=cfg.lna_p1db_dbm,
        lna_max_gain_db=cfg.lna_max_gain_db,
        adc_clip_db=cfg.adc_clip_dbfs,
        quant_bits=cfg.quant_bits,
        channel_delays_s=cfg.channel_delays_s,
        channel_powers_db=cfg.channel_powers_db,
        channel_k_factor_db=cfg.channel_k_factor_db,
        channel_delay_jitter_std_s=max(0.0, float(cfg.channel_delay_jitter_std_ns)) * 1e-9,
        channel_power_jitter_std_db=max(0.0, float(cfg.channel_power_jitter_std_db)),
        psd_unit="dBm/MHz",
        psd_sanity_check=True,
        psd_prefix_base=f"simulation/mms/psd_{case_tag}",
        baseline_sanity_mode=baseline_sanity_mode,
        auto_calibrate=(cfg.toa_calibration_override is None),
        toa_calibration_samples_override=cfg.toa_calibration_override,
        range_bias_correction_m=cfg.range_bias_correction_m,
        toa_calibration_distance_m=cfg.distance_m,
        toa_calibration_use_runtime_channel=cfg.toa_calibration_use_runtime_channel,
        distance_adaptive_multipath=cfg.distance_adaptive_multipath,
        enable_crc=True,
        debug_first_trial=debug_first_trial,
        save_psd=save_psd,
    )[0]

    return {
        "wifi_on": wifi_on,
        "nb_fc_hz": nb_fc_hz,
        "uwb_fc_hz": uwb_fc_hz,
        "wifi_fc_hz": wifi_fc_hz,
        "wifi_channel": wifi_ch,
        "df_nb_wifi_mhz": (wifi_fc_hz - nb_fc_hz) / 1e6,
        "df_uwb_wifi_mhz": (wifi_fc_hz - uwb_fc_hz) / 1e6,
        "df_uwb_nb_mhz": (uwb_fc_hz - nb_fc_hz) / 1e6,
        "nb_wifi_overlap": nb_wifi_overlap,
        "uwb_wifi_overlap": uwb_wifi_overlap,
        "uwb_waveform_interference_on": bool(interference_wf is not None),
        "control_ok": control_ok,
        "nb_poll_attempts": int(nb_poll_attempts),
        "nb_control_retries": int(max(0, nb_poll_attempts - 1)),
        "ranging_rmse_m": perf["ranging_rmse_m"],
        "ranging_rmse_all_m": perf["ranging_rmse_all_m"],
        "ranging_bias_m": perf.get("ranging_bias_m", float("nan")),
        "ranging_std_m": perf.get("ranging_std_m", float("nan")),
        "ranging_bias_all_m": perf.get("ranging_bias_all_m", float("nan")),
        "ranging_std_all_m": perf.get("ranging_std_all_m", float("nan")),
        "first_path_index_mean": perf.get("first_path_index_mean", float("nan")),
        "peak_index_mean": perf.get("peak_index_mean", float("nan")),
        "detection_threshold_abs_mean": perf.get("detection_threshold_abs_mean", float("nan")),
        "estimated_tof_ns_mean": perf.get("estimated_tof_ns_mean", float("nan")),
        "sample_period_ns": perf.get("sample_period_ns", float("nan")),
        "applied_calibration_offset_ns": perf.get("applied_calibration_offset_ns", float("nan")),
        "first_path_thr_db_mean": perf.get("first_path_thr_db_mean", float("nan")),
        "first_path_peak_frac_mean": perf.get("first_path_peak_frac_mean", float("nan")),
        "first_path_noise_floor_mean": perf.get("first_path_noise_floor_mean", float("nan")),
        "first_path_thr_noise_abs_mean": perf.get("first_path_thr_noise_abs_mean", float("nan")),
        "first_path_thr_peak_abs_mean": perf.get("first_path_thr_peak_abs_mean", float("nan")),
        "first_path_snr_corr_db_mean": perf.get("first_path_snr_corr_db_mean", float("nan")),
        "first_path_peak_ratio_db_mean": perf.get("first_path_peak_ratio_db_mean", float("nan")),
        "noise_win_start_mean": perf.get("noise_win_start_mean", float("nan")),
        "noise_win_end_mean": perf.get("noise_win_end_mean", float("nan")),
        "first_path_fallback_rate": perf.get("first_path_fallback_rate", float("nan")),
        "a2b1_rstu_mean": perf.get("a2b1_rstu_mean", float("nan")),
        "b2a1_rstu_mean": perf.get("b2a1_rstu_mean", float("nan")),
        "a2b2_rstu_mean": perf.get("a2b2_rstu_mean", float("nan")),
        "b2a2_rstu_mean": perf.get("b2a2_rstu_mean", float("nan")),
        "ds_ra_rstu_mean": perf.get("ds_ra_rstu_mean", float("nan")),
        "ds_rb_rstu_mean": perf.get("ds_rb_rstu_mean", float("nan")),
        "ds_da_rstu_mean": perf.get("ds_da_rstu_mean", float("nan")),
        "ds_db_rstu_mean": perf.get("ds_db_rstu_mean", float("nan")),
        "ds_tof_rstu_mean": perf.get("ds_tof_rstu_mean", float("nan")),
        "ber": perf["ber"],
        "fer": perf["fer"],
        "snr_db_avg": perf["snr_db_avg"],
        "ranging_fail_rate": perf["ranging_fail_rate"],
        "range_bias_correction_m_used": cfg.range_bias_correction_m,
    }


def _parse_csv_floats(text: str) -> list[float]:
    vals: list[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def run_full_stack_demo(
    cfg_override: FullStackConfig | None = None,
    sweep_peak_fracs: list[float] | None = None,
) -> None:
    cfg = cfg_override if cfg_override is not None else FullStackConfig(
        distance_m=20.0,
        nb_channel=1,
        uwb_channel=5,
        wifi_channel=108,  # 6 GHz (co-channel example near 6.49 GHz)
        wifi_bw_mhz=160,
        wifi_standard="wifi7",
        nb_center_override_hz=6.4890e9,
        nb_eirp_dbw=-16.0,
        wifi_tx_power_dbw=-20.0,
        nf_db=6.0,
        seed=20260214,
    )

    print("=== Full Stack MMS Demo ===")
    print("Flow: NB initiation/control (ADV-POLL/RESP/CONF) -> UWB MMS ranging")
    print(
        f"Config: d={cfg.distance_m:.1f} m, NB_ch={cfg.nb_channel}, UWB_ch={cfg.uwb_channel}, "
        f"Wi-Fi={cfg.wifi_standard} 6GHz ch{cfg.wifi_channel}, BW={cfg.wifi_bw_mhz} MHz"
    )
    case_trials = int(os.getenv("MMS_DEMO_N_TRIALS", "200"))

    def _calibrated_cfg(c: FullStackConfig, seed_add: int = 7777) -> FullStackConfig:
        cal_res = simulate_mms_performance(
            distances_m=(c.distance_m,),
            n_trials=1,
            rif_payload_bits=64,
            fc_hz=uwb_center_freq_hz(c.uwb_channel),
            uwb_fs_hz=499.2e6,
            tx_eirp_dbw=c.nb_eirp_dbw,
            nf_db=c.nf_db,
            temperature_k=c.temperature_k,
            seed=c.seed + seed_add,
            detector_mode="first_path",
            toa_refine_method=c.toa_refine_method,
            corr_upsample=c.corr_upsample,
            corr_win=c.corr_win,
            first_path=c.first_path,
            first_path_thr_db=c.first_path_thr_db,
            first_path_peak_frac=c.first_path_peak_frac,
            fp_use_adaptive_thr=c.fp_use_adaptive_thr,
            fp_snr_switch_db=c.fp_snr_switch_db,
            fp_thr_noise_cap_mult=c.fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=c.fp_thr_min_floor_mult,
            first_path_search_back=c.first_path_search_back,
            first_path_persist=c.first_path_persist,
            first_path_local_win=c.first_path_local_win,
            baseline_sanity_mode=True,
            auto_calibrate=True,
            enable_crc=True,
            debug_first_trial=False,
            save_psd=False,
        )[0]
        cal_override_l = float(cal_res["toa_calibration_samples"])
        print(f"[MMS calib] using fixed ToA calibration={cal_override_l:.6f} samples for all cases")
        return replace(c, toa_calibration_override=cal_override_l)

    cfg = _calibrated_cfg(cfg)
    if cfg.auto_bias_calibrate or cfg.auto_range_bias_calibrate:
        bias_res = simulate_mms_performance(
            distances_m=(cfg.distance_m,),
            n_trials=120,
            rif_payload_bits=256,
            fc_hz=uwb_center_freq_hz(cfg.uwb_channel),
            uwb_fs_hz=499.2e6,
            uwb_fc_hz=uwb_center_freq_hz(cfg.uwb_channel),
            tx_eirp_dbw=cfg.nb_eirp_dbw,
            nf_db=cfg.nf_db,
            temperature_k=cfg.temperature_k,
            seed=cfg.seed + 7878,
            detector_mode="first_path",
            toa_refine_method=cfg.toa_refine_method,
            corr_upsample=cfg.corr_upsample,
            corr_win=cfg.corr_win,
            first_path=cfg.first_path,
            first_path_thr_db=cfg.first_path_thr_db,
            first_path_peak_frac=cfg.first_path_peak_frac,
            fp_use_adaptive_thr=cfg.fp_use_adaptive_thr,
            fp_snr_switch_db=cfg.fp_snr_switch_db,
            fp_thr_noise_cap_mult=cfg.fp_thr_noise_cap_mult,
            fp_thr_min_floor_mult=cfg.fp_thr_min_floor_mult,
            first_path_search_back=cfg.first_path_search_back,
            first_path_persist=cfg.first_path_persist,
            first_path_local_win=cfg.first_path_local_win,
            baseline_sanity_mode=False,
            auto_calibrate=False,
            toa_calibration_samples_override=cal_override,
            enable_crc=True,
            debug_first_trial=False,
            save_psd=False,
        )[0]
        auto_bias = float(bias_res.get("ranging_bias_m", 0.0))
        cfg = replace(cfg, range_bias_correction_m=cfg.range_bias_correction_m + auto_bias)
        print(
            "[MMS calib] auto range bias correction "
            f"auto_bias={auto_bias:.6f} m, applied_range_bias_correction={cfg.range_bias_correction_m:.6f} m"
        )

    if sweep_peak_fracs is not None and len(sweep_peak_fracs) > 0:
        rows = []
        print("\n=== Peak Fraction Sweep (baseline + oob60) ===")
        for pf in sweep_peak_fracs:
            cfg_pf = _calibrated_cfg(replace(cfg, first_path_peak_frac=float(pf)), seed_add=7777 + int(1000 * pf))
            base_cfg = replace(cfg_pf)
            oob60_cfg = replace(
                cfg_pf,
                wifi_tx_power_dbw=-6.0,
                wifi_aclr_db=30.0,
                rx_stop_db=60.0,
                enable_agc=False,
                adc_clip_dbfs=None,
            )
            base = run_full_stack_case(
                base_cfg,
                wifi_on=False,
                baseline_sanity_mode=False,
                case_tag=f"sweep_pf{pf:.3f}_base",
                n_trials=case_trials,
            )
            oob60 = run_full_stack_case(
                oob60_cfg,
                wifi_on=True,
                wifi_channel_override=200,
                baseline_sanity_mode=False,
                case_tag=f"sweep_pf{pf:.3f}_oob60",
                n_trials=case_trials,
            )
            rows.append((pf, base, oob60))

        print("peak_frac | baseline(Bias/RMSE/Std) | oob60(Bias/RMSE/Std)")
        best = None
        for pf, base, oob60 in rows:
            print(
                f"{pf:7.3f} | "
                f"{base['ranging_bias_m']:+.3f}/{base['ranging_rmse_m']:.3f}/{base['ranging_std_m']:.3f} | "
                f"{oob60['ranging_bias_m']:+.3f}/{oob60['ranging_rmse_m']:.3f}/{oob60['ranging_std_m']:.3f}"
            )
            pass_bias = (abs(base["ranging_bias_m"]) <= 0.03) and (abs(oob60["ranging_bias_m"]) <= 0.04)
            score = base["ranging_rmse_m"] + oob60["ranging_rmse_m"]
            if pass_bias:
                if (best is None) or (score < best[0]):
                    best = (score, pf, base, oob60)
        if best is None:
            print("[SWEEP] No peak_frac met bias constraints; choose by lowest RMSE sum anyway.")
            best_any = min(rows, key=lambda r: float(r[1]["ranging_rmse_m"] + r[2]["ranging_rmse_m"]))
            pf, base, oob60 = best_any
            print(
                f"[SWEEP BEST] peak_frac={pf:.3f} | "
                f"baseline bias/rmse/std={base['ranging_bias_m']:+.3f}/{base['ranging_rmse_m']:.3f}/{base['ranging_std_m']:.3f} | "
                f"oob60 bias/rmse/std={oob60['ranging_bias_m']:+.3f}/{oob60['ranging_rmse_m']:.3f}/{oob60['ranging_std_m']:.3f}"
            )
        else:
            _, pf, base, oob60 = best
            print(
                f"[SWEEP BEST] peak_frac={pf:.3f} | "
                f"baseline bias/rmse/std={base['ranging_bias_m']:+.3f}/{base['ranging_rmse_m']:.3f}/{base['ranging_std_m']:.3f} | "
                f"oob60 bias/rmse/std={oob60['ranging_bias_m']:+.3f}/{oob60['ranging_rmse_m']:.3f}/{oob60['ranging_std_m']:.3f}"
            )
        return

    cases = [
        ("wifi_off", "Wi-Fi OFF baseline", cfg, {"wifi_on": False, "baseline_sanity_mode": False}),
        (
            "wifi_inband",
            "Wi-Fi ON in-band (Δf~0.4MHz)",
            replace(cfg, wifi_tx_power_dbw=-20.0, rx_stop_db=25.0),
            {"wifi_on": True, "wifi_channel_override": 108, "baseline_sanity_mode": False},
        ),
        (
            "wifi_oob_25",
            "Wi-Fi ON out-of-band, stop=25dB + nonlinearity",
            replace(
                cfg,
                wifi_tx_power_dbw=-6.0,
                wifi_aclr_db=30.0,
                rx_stop_db=25.0,
                enable_agc=True,
                agc_stage="pre_selectivity",
                agc_target_dbfs=-3.0,
                agc_max_gain_db=35.0,
                lna_p1db_dbm=-65.0,
                lna_max_gain_db=12.0,
                adc_clip_dbfs=-1.0,
                quant_bits=None,
            ),
            {"wifi_on": True, "wifi_channel_override": 200, "baseline_sanity_mode": False},
        ),
        (
            "wifi_oob_60",
            "Wi-Fi ON out-of-band, stop=60dB",
            replace(cfg, wifi_tx_power_dbw=-6.0, wifi_aclr_db=30.0, rx_stop_db=60.0, enable_agc=False, adc_clip_dbfs=None),
            {"wifi_on": True, "wifi_channel_override": 200, "baseline_sanity_mode": False},
        ),
        (
            "wifi_off_sanity",
            "Wi-Fi OFF baseline sanity (no multipath/ppm)",
            replace(cfg),
            {"wifi_on": False, "baseline_sanity_mode": True, "n_trials": 80},
        ),
    ]

    results_by_tag: dict[str, dict] = {}
    for tag, name, cfg_case, kwargs in cases:
        if "n_trials" not in kwargs:
            kwargs = {**kwargs, "n_trials": case_trials}
        res = run_full_stack_case(cfg_case, case_tag=tag, **kwargs)
        results_by_tag[tag] = res
        print(f"\n--- Case: {name} ---")
        print(
            f"fc NB={res['nb_fc_hz']/1e9:.6f} GHz, UWB={res['uwb_fc_hz']/1e9:.6f} GHz, "
            f"Wi-Fi(ch{res['wifi_channel']})={res['wifi_fc_hz']/1e9:.6f} GHz"
        )
        print(
            f"df NB-WiFi={res['df_nb_wifi_mhz']:.3f} MHz, "
            f"df UWB-WiFi={res['df_uwb_wifi_mhz']:.3f} MHz, "
            f"df UWB-NB={res['df_uwb_nb_mhz']:.3f} MHz"
        )
        print(
            f"NB/Wi-Fi overlap={res['nb_wifi_overlap']:.3f}, "
            f"UWB/Wi-Fi overlap={res['uwb_wifi_overlap']:.3f}, "
            f"UWB waveform interference={'ON' if res['uwb_waveform_interference_on'] else 'OFF'}, "
            f"range_bias_corr={res['range_bias_correction_m_used']:.3f} m"
        )
        rmse_txt = "N/A" if not np.isfinite(res["ranging_rmse_m"]) else f"{res['ranging_rmse_m']:.3f} m"
        rmse_all_txt = "N/A" if not np.isfinite(res["ranging_rmse_all_m"]) else f"{res['ranging_rmse_all_m']:.3f} m"
        bias_txt = "N/A" if not np.isfinite(res["ranging_bias_m"]) else f"{res['ranging_bias_m']:.3f} m"
        std_txt = "N/A" if not np.isfinite(res["ranging_std_m"]) else f"{res['ranging_std_m']:.3f} m"
        bias_all_txt = "N/A" if not np.isfinite(res["ranging_bias_all_m"]) else f"{res['ranging_bias_all_m']:.3f} m"
        std_all_txt = "N/A" if not np.isfinite(res["ranging_std_all_m"]) else f"{res['ranging_std_all_m']:.3f} m"
        print(
            f"NB control_ok={res['control_ok']} | "
            f"BER={res['ber']:.6e} | FER={res['fer']:.3f} | "
            f"RMSE(valid)={rmse_txt} | Bias(valid)={bias_txt} | Std(valid)={std_txt} | "
            f"RMSE(all-phy)={rmse_all_txt} | Bias(all-phy)={bias_all_txt} | Std(all-phy)={std_all_txt} | "
            f"RangingFail={res['ranging_fail_rate']:.3f} | "
            f"SNR={res['snr_db_avg']:.2f} dB"
        )

    # Lightweight regression check (non-fatal).
    base = results_by_tag.get("wifi_off")
    oob60 = results_by_tag.get("wifi_oob_60")
    if base is not None and np.isfinite(base["ranging_bias_m"]) and np.isfinite(base["ranging_rmse_m"]):
        ok_base = (base["ranging_bias_m"] <= 0.03) and (base["ranging_rmse_m"] <= 0.11)
        print(
            "[MMS regression] baseline target "
            f"(bias<=0.03m, rmse<=0.11m): {'PASS' if ok_base else 'WARN'} | "
            f"bias={base['ranging_bias_m']:.3f} m, rmse={base['ranging_rmse_m']:.3f} m"
        )
    if oob60 is not None and np.isfinite(oob60["ranging_bias_m"]) and np.isfinite(oob60["ranging_rmse_m"]):
        ok_oob60 = (oob60["ranging_bias_m"] <= 0.04) and (oob60["ranging_rmse_m"] <= 0.12)
        print(
            "[MMS regression] oob60 target "
            f"(bias<=0.04m, rmse<=0.12m): {'PASS' if ok_oob60 else 'WARN'} | "
            f"bias={oob60['ranging_bias_m']:.3f} m, rmse={oob60['ranging_rmse_m']:.3f} m"
        )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MMS full-stack demo with optional ToA/threshold overrides.")
    p.add_argument("--range-bias-corr", type=float, default=None, help="Manual range bias correction [m].")
    p.add_argument(
        "--auto-range-bias-calibrate",
        action="store_true",
        help="Estimate range bias from Wi-Fi OFF baseline and apply to all cases.",
    )
    p.add_argument("--fp-peak-frac", type=float, default=None, help="First-path peak fraction threshold.")
    p.add_argument("--fp-thr-db", type=float, default=None, help="First-path noise threshold [dB].")
    p.add_argument("--fp-snr-switch-db", type=float, default=None, help="Adaptive threshold SNR switch [dB].")
    p.add_argument("--fp-use-adaptive-thr", dest="fp_use_adaptive_thr", action="store_true")
    p.add_argument("--no-fp-use-adaptive-thr", dest="fp_use_adaptive_thr", action="store_false")
    p.set_defaults(fp_use_adaptive_thr=None)
    p.add_argument("--n-trials", type=int, default=None, help="Override number of trials per case (except sanity).")
    p.add_argument(
        "--sweep-peak-frac",
        type=str,
        default=None,
        help='Comma-separated peak_frac sweep list. Example: "0.16,0.18,0.20,0.22"',
    )
    return p


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()
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
    env_bias = os.getenv("MMS_RANGE_BIAS_CORR_M")
    if env_bias is not None and args.range_bias_corr is None:
        try:
            args.range_bias_corr = float(env_bias)
        except ValueError:
            pass
    if args.range_bias_corr is not None:
        cfg = replace(cfg, range_bias_correction_m=float(args.range_bias_corr))
    if args.auto_range_bias_calibrate:
        cfg = replace(cfg, auto_range_bias_calibrate=True)
    if args.fp_peak_frac is not None:
        cfg = replace(cfg, first_path_peak_frac=float(args.fp_peak_frac))
    if args.fp_thr_db is not None:
        cfg = replace(cfg, first_path_thr_db=float(args.fp_thr_db))
    if args.fp_snr_switch_db is not None:
        cfg = replace(cfg, fp_snr_switch_db=float(args.fp_snr_switch_db))
    if args.fp_use_adaptive_thr is not None:
        cfg = replace(cfg, fp_use_adaptive_thr=bool(args.fp_use_adaptive_thr))

    # Optional global trials override via env/CLI for quick experiments.
    if args.n_trials is not None:
        os.environ["MMS_DEMO_N_TRIALS"] = str(int(args.n_trials))

    sweep_vals = _parse_csv_floats(args.sweep_peak_frac) if args.sweep_peak_frac else None
    run_full_stack_demo(cfg_override=cfg, sweep_peak_fracs=sweep_vals)
