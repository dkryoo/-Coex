from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
        9: 7987.2e6,
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
    nf_db: float = 6.0
    temperature_k: float = 290.0
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


def run_full_stack_case(cfg: FullStackConfig, wifi_on: bool) -> dict:
    nb_fc_hz = float(cfg.nb_center_override_hz) if cfg.nb_center_override_hz is not None else nb_center_freq_hz(cfg.nb_channel)
    uwb_fc_hz = uwb_center_freq_hz(cfg.uwb_channel)
    wifi_fc_hz = wifi6_center_freq_hz(cfg.wifi_channel)
    wifi_bw_hz = cfg.wifi_bw_mhz * 1e6
    nb_bw_hz = 2.0e6
    uwb_bw_hz = 499.2e6
    nb_wifi_overlap = _spectral_overlap_ratio(nb_fc_hz, nb_bw_hz, wifi_fc_hz, wifi_bw_hz)
    uwb_wifi_overlap = _spectral_overlap_ratio(uwb_fc_hz, uwb_bw_hz, wifi_fc_hz, wifi_bw_hz)
    uwb_wifi_penalty_db = 18.0 * uwb_wifi_overlap if wifi_on else 0.0

    tx_nb = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    rx_nb = OQPSK_SF32_Rx(chip_rate_hz=2e6, osr=8)
    WiFiOFDMTx = _load_wifi_tx_class()
    wifi_tx = WiFiOFDMTx(rng_seed=cfg.seed, center_freq_hz=wifi_fc_hz)

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
    control_ok = False
    for r in range(1, 6):
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

    # UWB MMS ranging phase performance (initiation/control assumed complete)
    perf = simulate_mms_performance(
        distances_m=(cfg.distance_m,),
        n_trials=200,
        rif_payload_bits=256,
        fc_hz=uwb_fc_hz,
        tx_eirp_dbw=cfg.nb_eirp_dbw,
        nf_db=cfg.nf_db,
        temperature_k=cfg.temperature_k,
        seed=cfg.seed + 9000,
        detector_mode="legacy",
        external_interference_penalty_db=uwb_wifi_penalty_db,
    )[0]

    return {
        "wifi_on": wifi_on,
        "nb_fc_hz": nb_fc_hz,
        "uwb_fc_hz": uwb_fc_hz,
        "wifi_fc_hz": wifi_fc_hz,
        "df_nb_wifi_mhz": (wifi_fc_hz - nb_fc_hz) / 1e6,
        "df_uwb_wifi_mhz": (wifi_fc_hz - uwb_fc_hz) / 1e6,
        "df_uwb_nb_mhz": (uwb_fc_hz - nb_fc_hz) / 1e6,
        "nb_wifi_overlap": nb_wifi_overlap,
        "uwb_wifi_overlap": uwb_wifi_overlap,
        "uwb_wifi_penalty_db": uwb_wifi_penalty_db,
        "control_ok": control_ok,
        "ranging_rmse_m": perf["ranging_rmse_m"],
        "ber": perf["ber"],
        "fer": perf["fer"],
        "snr_db_avg": perf["snr_db_avg"],
    }


def run_full_stack_demo() -> None:
    cfg = FullStackConfig(
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

    for wifi_on in (False, True):
        res = run_full_stack_case(cfg, wifi_on=wifi_on)
        print("\n--- Case:", "Wi-Fi OFF" if not wifi_on else "Wi-Fi ON", "---")
        print(
            f"fc NB={res['nb_fc_hz']/1e9:.6f} GHz, UWB={res['uwb_fc_hz']/1e9:.6f} GHz, "
            f"Wi-Fi={res['wifi_fc_hz']/1e9:.6f} GHz"
        )
        print(
            f"df NB-WiFi={res['df_nb_wifi_mhz']:.3f} MHz, "
            f"df UWB-WiFi={res['df_uwb_wifi_mhz']:.3f} MHz, "
            f"df UWB-NB={res['df_uwb_nb_mhz']:.3f} MHz"
        )
        print(
            f"NB/Wi-Fi overlap={res['nb_wifi_overlap']:.3f}, "
            f"UWB/Wi-Fi overlap={res['uwb_wifi_overlap']:.3f}, "
            f"UWB interference penalty={res['uwb_wifi_penalty_db']:.2f} dB"
        )
        print(
            f"NB control_ok={res['control_ok']} | "
            f"BER={res['ber']:.6e} | FER={res['fer']:.3f} | "
            f"RMSE={res['ranging_rmse_m']:.3f} m | SNR={res['snr_db_avg']:.2f} dB"
        )


if __name__ == "__main__":
    run_full_stack_demo()
