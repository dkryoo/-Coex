from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional


# -----------------------------------------------------------------------------
# Compact Frame IDs (4ab Table 10; constants for serialization in this model)
# -----------------------------------------------------------------------------
CFID_ADV_POLL = 0x01
CFID_ADV_RESP = 0x02
CFID_ADV_CONF = 0x03


class FrameCrcError(ValueError):
    pass


class FrameParseError(ValueError):
    pass


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SmidTlv:
    """
    SMID TLV per 4ab Figure 64:
      Tag (1), Length (1), Values (Length)
    """

    tag: int
    values: bytes

    def __post_init__(self) -> None:
        if not (0 <= self.tag <= 0xFF):
            raise ValueError("SMID tag must fit in 1 octet")
        if len(self.values) > 0xFF:
            raise ValueError("SMID values length must fit in 1 octet")


@dataclass(frozen=True)
class AdvPoll:
    """
    4ab Clause 10.39.11.3.x ADV-POLL, MC=1 content:
      Initialization Slot Duration (1)
      CAP Duration (1)
      Supported O-QPSK Modulation Modes (1)
      SMID TLVs (0..variable)
    """

    init_slot_dur: int
    cap_dur: int
    supported_mod_modes: int
    smid_tlvs: List[SmidTlv]

    def __post_init__(self) -> None:
        for n, v in (
            ("init_slot_dur", self.init_slot_dur),
            ("cap_dur", self.cap_dur),
            ("supported_mod_modes", self.supported_mod_modes),
        ):
            if not (0 <= v <= 0xFF):
                raise ValueError(f"{n} must fit in 1 octet")


@dataclass(frozen=True)
class AdvResp:
    """
    4ab Clause 10.39.11.3.x ADV-RESP, MC=0 content (Figure 82 style):
      Responder RPA Hash (3)
      Message ID (1)
      NB Full Channel Map (6)
      Management PHY Configuration (1)
      Management MAC Configuration (7)
      Ranging PHY Configuration (4)
      MMS Number of Fragments (1)
    """

    rpa_hash: bytes
    message_id: int
    nb_full_channel_map: bytes
    mgmt_phy_cfg: int
    mgmt_mac_cfg: bytes
    ranging_phy_cfg: bytes
    mms_num_frags: int

    def __post_init__(self) -> None:
        if len(self.rpa_hash) != 3:
            raise ValueError("rpa_hash must be exactly 3 octets")
        if not (0 <= self.message_id <= 0xFF):
            raise ValueError("message_id must fit in 1 octet")
        if len(self.nb_full_channel_map) != 6:
            raise ValueError("nb_full_channel_map must be exactly 6 octets")
        if not (0 <= self.mgmt_phy_cfg <= 0xFF):
            raise ValueError("mgmt_phy_cfg must fit in 1 octet")
        if len(self.mgmt_mac_cfg) != 7:
            raise ValueError("mgmt_mac_cfg must be exactly 7 octets")
        if len(self.ranging_phy_cfg) != 4:
            raise ValueError("ranging_phy_cfg must be exactly 4 octets")
        if not (0 <= self.mms_num_frags <= 0xFF):
            raise ValueError("mms_num_frags must fit in 1 octet")


@dataclass(frozen=True)
class AdvConf:
    """
    4ab Clause 10.39.11.3.x ADV-CONF, MC=0 content:
      Initiator RPA Hash (3)
      Message ID (1)
      Responder Address (3)
      SOR Time Offset (3)
    """

    rpa_hash: bytes
    message_id: int
    responder_addr: bytes
    sor_time_offset: bytes

    def __post_init__(self) -> None:
        if len(self.rpa_hash) != 3:
            raise ValueError("rpa_hash must be exactly 3 octets")
        if not (0 <= self.message_id <= 0xFF):
            raise ValueError("message_id must fit in 1 octet")
        if len(self.responder_addr) != 3:
            raise ValueError("responder_addr must be exactly 3 octets")
        if len(self.sor_time_offset) != 3:
            raise ValueError("sor_time_offset must be exactly 3 octets")


# -----------------------------------------------------------------------------
# CRC-16 IEEE 802.15.4 FCS
# Polynomial x^16 + x^12 + x^5 + 1, bit-reflected form 0x8408.
# -----------------------------------------------------------------------------
def crc16_802154(data: bytes) -> int:
    crc = 0x0000
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


def verify_crc(frame_bytes: bytes) -> bool:
    if len(frame_bytes) < 4:
        return False
    body = frame_bytes[:-2]
    rx_fcs = int.from_bytes(frame_bytes[-2:], "little")
    return crc16_802154(body) == rx_fcs


def _append_fcs(body: bytes) -> bytes:
    fcs = crc16_802154(body)
    return body + fcs.to_bytes(2, "little")


def _check_crc_or_raise(frame: bytes) -> bytes:
    if not verify_crc(frame):
        raise FrameCrcError("FCS/CRC check failed")
    return frame[:-2]


# -----------------------------------------------------------------------------
# Encoders/Decoders
# Simplified compact frame wire format for this simulation:
#   [FrameID(1)] [MessageControl(1)] [frame-specific content] [FCS(2)]
# -----------------------------------------------------------------------------
def encode_adv_poll(pkt: AdvPoll) -> bytes:
    # 4ab ADV-POLL MC=1
    body = bytearray([CFID_ADV_POLL, 0x01, pkt.init_slot_dur, pkt.cap_dur, pkt.supported_mod_modes])
    for tlv in pkt.smid_tlvs:
        body.extend([tlv.tag & 0xFF, len(tlv.values) & 0xFF])
        body.extend(tlv.values)
    return _append_fcs(bytes(body))


def decode_adv_poll(frame: bytes) -> AdvPoll:
    body = _check_crc_or_raise(frame)
    if len(body) < 5:
        raise FrameParseError("ADV-POLL too short")
    if body[0] != CFID_ADV_POLL:
        raise FrameParseError("Not an ADV-POLL frame")
    if body[1] != 0x01:
        raise FrameParseError("ADV-POLL supports MC=1 only")

    init_slot_dur = body[2]
    cap_dur = body[3]
    supported_mod_modes = body[4]

    tlvs: List[SmidTlv] = []
    i = 5
    while i < len(body):
        if i + 2 > len(body):
            raise FrameParseError("Truncated SMID TLV header")
        tag = body[i]
        ln = body[i + 1]
        i += 2
        if i + ln > len(body):
            raise FrameParseError("Truncated SMID TLV values")
        values = bytes(body[i : i + ln])
        i += ln
        tlvs.append(SmidTlv(tag=tag, values=values))

    return AdvPoll(
        init_slot_dur=init_slot_dur,
        cap_dur=cap_dur,
        supported_mod_modes=supported_mod_modes,
        smid_tlvs=tlvs,
    )


def encode_adv_resp(pkt: AdvResp) -> bytes:
    # 4ab ADV-RESP MC=0
    body = bytearray([CFID_ADV_RESP, 0x00])
    body.extend(pkt.rpa_hash)
    body.append(pkt.message_id & 0xFF)
    body.extend(pkt.nb_full_channel_map)
    body.append(pkt.mgmt_phy_cfg & 0xFF)
    body.extend(pkt.mgmt_mac_cfg)
    body.extend(pkt.ranging_phy_cfg)
    body.append(pkt.mms_num_frags & 0xFF)
    return _append_fcs(bytes(body))


def decode_adv_resp(frame: bytes) -> AdvResp:
    body = _check_crc_or_raise(frame)
    # fixed body length without FCS: 2 + 3 + 1 + 6 + 1 + 7 + 4 + 1 = 25
    if len(body) != 25:
        raise FrameParseError(f"ADV-RESP length mismatch: {len(body)}")
    if body[0] != CFID_ADV_RESP:
        raise FrameParseError("Not an ADV-RESP frame")
    if body[1] != 0x00:
        raise FrameParseError("ADV-RESP supports MC=0 only")

    i = 2
    rpa_hash = bytes(body[i : i + 3]); i += 3
    message_id = body[i]; i += 1
    nb_full_channel_map = bytes(body[i : i + 6]); i += 6
    mgmt_phy_cfg = body[i]; i += 1
    mgmt_mac_cfg = bytes(body[i : i + 7]); i += 7
    ranging_phy_cfg = bytes(body[i : i + 4]); i += 4
    mms_num_frags = body[i]

    return AdvResp(
        rpa_hash=rpa_hash,
        message_id=message_id,
        nb_full_channel_map=nb_full_channel_map,
        mgmt_phy_cfg=mgmt_phy_cfg,
        mgmt_mac_cfg=mgmt_mac_cfg,
        ranging_phy_cfg=ranging_phy_cfg,
        mms_num_frags=mms_num_frags,
    )


def encode_adv_conf(pkt: AdvConf) -> bytes:
    # 4ab ADV-CONF MC=0
    body = bytearray([CFID_ADV_CONF, 0x00])
    body.extend(pkt.rpa_hash)
    body.append(pkt.message_id & 0xFF)
    body.extend(pkt.responder_addr)
    body.extend(pkt.sor_time_offset)
    return _append_fcs(bytes(body))


def decode_adv_conf(frame: bytes) -> AdvConf:
    body = _check_crc_or_raise(frame)
    # fixed body length without FCS: 2 + 3 + 1 + 3 + 3 = 12
    if len(body) != 12:
        raise FrameParseError(f"ADV-CONF length mismatch: {len(body)}")
    if body[0] != CFID_ADV_CONF:
        raise FrameParseError("Not an ADV-CONF frame")
    if body[1] != 0x00:
        raise FrameParseError("ADV-CONF supports MC=0 only")

    i = 2
    rpa_hash = bytes(body[i : i + 3]); i += 3
    message_id = body[i]; i += 1
    responder_addr = bytes(body[i : i + 3]); i += 3
    sor_time_offset = bytes(body[i : i + 3])

    return AdvConf(
        rpa_hash=rpa_hash,
        message_id=message_id,
        responder_addr=responder_addr,
        sor_time_offset=sor_time_offset,
    )


# -----------------------------------------------------------------------------
# Handshake state machines (in-memory transport model)
# -----------------------------------------------------------------------------
class Initiator:
    def __init__(self) -> None:
        self.poll_tx_count = 0

    def initiator_tx(self) -> bytes:
        self.poll_tx_count += 1
        poll = AdvPoll(
            init_slot_dur=10,
            cap_dur=20,
            supported_mod_modes=0x03,
            smid_tlvs=[
                SmidTlv(tag=CFID_ADV_RESP, values=bytes([0x01, 0x02])),
                SmidTlv(tag=CFID_ADV_CONF, values=bytes([0x10])),
            ],
        )
        return encode_adv_poll(poll)

    def initiator_rx(self, frame: bytes) -> Optional[bytes]:
        # On valid ADV-RESP, return optional ADV-CONF bytes.
        resp = decode_adv_resp(frame)
        conf = AdvConf(
            rpa_hash=bytes([0x11, 0x22, 0x33]),
            message_id=resp.message_id,
            responder_addr=resp.rpa_hash,
            sor_time_offset=bytes([0x01, 0x02, 0x03]),
        )
        return encode_adv_conf(conf)


class Responder:
    def responder_rx(self, frame: bytes) -> Optional[bytes]:
        try:
            _ = decode_adv_poll(frame)
        except (FrameCrcError, FrameParseError):
            return None

        resp = AdvResp(
            rpa_hash=bytes([0xAA, 0xBB, 0xCC]),
            message_id=0x42,
            nb_full_channel_map=bytes([1, 2, 3, 4, 5, 6]),
            mgmt_phy_cfg=0x15,
            mgmt_mac_cfg=bytes([10, 11, 12, 13, 14, 15, 16]),
            ranging_phy_cfg=bytes([21, 22, 23, 24]),
            mms_num_frags=2,
        )
        return encode_adv_resp(resp)


def _bytes_to_bits_lsb(data: bytes) -> List[int]:
    out: List[int] = []
    for b in data:
        for k in range(8):
            out.append((b >> k) & 1)
    return out


def _bits_lsb_to_bytes(bits: List[int], n_bytes: int) -> bytes:
    need = n_bytes * 8
    if len(bits) < need:
        raise ValueError("Not enough bits to reconstruct bytes")
    bits = bits[:need]
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        v = 0
        for k in range(8):
            v |= (int(bits[i * 8 + k]) & 1) << k
        out[i] = v
    return bytes(out)


def _resample_complex_linear(x, fs_in_hz: float, fs_out_hz: float):
    import numpy as np

    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("sample rates must be > 0")
    if len(x) <= 1:
        return x.astype(np.complex128)

    t_in = np.arange(len(x), dtype=float) / fs_in_hz
    n_out = max(1, int(np.floor((len(x) - 1) * fs_out_hz / fs_in_hz)) + 1)
    t_out = np.arange(n_out, dtype=float) / fs_out_hz
    re = np.interp(t_out, t_in, np.real(x))
    im = np.interp(t_out, t_in, np.imag(x))
    return (re + 1j * im).astype(np.complex128)


def _load_wifi_tx_class():
    wifi_path = Path(__file__).resolve().parents[1] / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("wifi_tx_module_for_handshake", wifi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Wi-Fi TX module from {wifi_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WiFiOFDMTx


def _transmit_compact_frame_over_nb_phy(
    frame_bytes: bytes,
    tx_nb,
    rx_nb,
    fs_hz: float,
    fc_hz: float,
    distance_m: float,
    nb_tx_eirp_dbw: float,
    nf_db: float,
    seed_base: int,
    use_wifi_interference: bool = False,
    wifi_tx=None,
    wifi_fc_hz: float = 6.52e9,
    wifi_bw_mhz: int = 160,
    wifi_standard: str = "wifi7",
    wifi_target_tp_mbps: float = 600.0,
    wifi_tx_power_dbw: float = -20.0,
    center_freq_gap_mhz: float = 20.0,
):
    import numpy as np
    from Channel.Rician import apply_distance_rician_channel
    from Channel.Thermal_noise import add_thermal_noise_white

    # bytes -> bitstream (LSB-first per octet) -> NB PHY waveform
    frame_bits = np.asarray(_bytes_to_bits_lsb(frame_bytes), dtype=int)
    tx_wf, _, _ = tx_nb.build_tx_waveform(
        psdu_bits=frame_bits,
        tx_eirp_db=nb_tx_eirp_dbw,
        regulatory_profile="unlicensed_6g_lpi_ap",
    )

    # NB channel (distance ToA + multipath/pathloss)
    rx_nb_wf, _, _, _, _ = apply_distance_rician_channel(
        wf=tx_wf,
        fs_hz=fs_hz,
        fc_hz=fc_hz,
        distance_m=distance_m,
        pathloss_exp=2.0,
        delays_s=(0.0, 50e-9, 120e-9),
        powers_db=(0.0, -6.0, -10.0),
        k_factor_db=8.0,
        include_toa=True,
        seed=seed_base + 1000,
    )
    rx_wf = np.concatenate([np.zeros(80, dtype=np.complex128), rx_nb_wf])

    # Optional Wi-Fi interference over its own channel then shifted in NB baseband.
    if use_wifi_interference:
        if wifi_tx is None:
            raise RuntimeError("wifi_tx must be provided when use_wifi_interference=True")

        wifi_wf, _ = wifi_tx.generate_for_target_rx_throughput(
            target_rx_throughput_mbps=wifi_target_tp_mbps,
            duration_s=max(0.001, len(rx_wf) / fs_hz),
            channel_bw_mhz=wifi_bw_mhz,
            standard=wifi_standard,
            tx_power_dbw=wifi_tx_power_dbw,
            center_freq_hz=wifi_fc_hz,
        )
        wifi_rs = _resample_complex_linear(wifi_wf, fs_in_hz=wifi_bw_mhz * 1e6, fs_out_hz=fs_hz)
        if len(wifi_rs) < len(rx_wf):
            wifi_rs = np.pad(wifi_rs, (0, len(rx_wf) - len(wifi_rs)))
        wifi_rs = wifi_rs[: len(rx_wf)]
        t = np.arange(len(wifi_rs), dtype=float) / fs_hz
        f_off = center_freq_gap_mhz * 1e6
        wifi_shift_tx = wifi_rs * np.exp(1j * 2.0 * np.pi * f_off * t)

        rx_wifi_wf, _, _, _, _ = apply_distance_rician_channel(
            wf=wifi_shift_tx,
            fs_hz=fs_hz,
            fc_hz=wifi_fc_hz,
            distance_m=distance_m,
            pathloss_exp=2.0,
            delays_s=(0.0, 30e-9, 80e-9),
            powers_db=(0.0, -6.0, -10.0),
            k_factor_db=6.0,
            include_toa=True,
            seed=seed_base + 2000,
        )
        rx_wifi_wf = np.concatenate([np.zeros(80, dtype=np.complex128), rx_wifi_wf])
        if len(rx_wifi_wf) < len(rx_wf):
            rx_wifi_wf = np.pad(rx_wifi_wf, (0, len(rx_wf) - len(rx_wifi_wf)))
        rx_wf = rx_wf + rx_wifi_wf[: len(rx_wf)]

    # One receiver thermal-noise injection.
    rx_wf = add_thermal_noise_white(
        wf=rx_wf,
        fs_hz=fs_hz,
        nf_db=nf_db,
        temperature_k=290.0,
        seed=seed_base + 3000,
    )

    # NB PHY decode -> bits -> original bytes
    try:
        rx_bits, _, _ = rx_nb.decode(rx_wf, tx_fir=None, verbose=False)
    except Exception:
        return None
    try:
        rx_bytes = _bits_lsb_to_bytes(list(map(int, rx_bits.tolist())), n_bytes=len(frame_bytes))
    except ValueError:
        return None
    return rx_bytes


def handshake_demo() -> None:
    """
    In-memory transport model:
      initiator_tx() -> responder_rx(bytes) -> initiator_rx(bytes) -> optional ADV-CONF
    """
    initiator = Initiator()
    responder = Responder()

    # Initiator repeatedly transmits ADV-POLL until ADV-RESP arrives.
    for attempt in range(1, 6):
        poll_bytes = initiator.initiator_tx()
        print(f"[Initiator] TX ADV-POLL attempt={attempt}, len={len(poll_bytes)}, crc_ok={verify_crc(poll_bytes)}")

        resp_bytes = responder.responder_rx(poll_bytes)
        if resp_bytes is None:
            print("[Responder] Discarded frame (CRC/parse fail)")
            continue

        print(f"[Responder] TX ADV-RESP len={len(resp_bytes)}, crc_ok={verify_crc(resp_bytes)}")
        parsed_resp = decode_adv_resp(resp_bytes)
        print(
            "[Initiator] RX ADV-RESP "
            f"msg_id={parsed_resp.message_id}, rpa_hash={parsed_resp.rpa_hash.hex()}, "
            f"map_len={len(parsed_resp.nb_full_channel_map)}"
        )

        conf_bytes = initiator.initiator_rx(resp_bytes)
        if conf_bytes is not None:
            print(f"[Initiator] TX ADV-CONF len={len(conf_bytes)}, crc_ok={verify_crc(conf_bytes)}")
            parsed_conf = decode_adv_conf(conf_bytes)
            print(
                "[Responder] RX ADV-CONF "
                f"msg_id={parsed_conf.message_id}, responder_addr={parsed_conf.responder_addr.hex()}, "
                f"sor={parsed_conf.sor_time_offset.hex()}"
            )
        break


def handshake_demo_over_phy(
    distance_m: float = 10.0,
    max_poll_retries: int = 5,
    use_wifi_interference: bool = True,
    seed_base: int = 10000,
) -> dict:
    """
    PHY-backed demo:
      Compact frame bytes -> NB PHY TX -> channel (Rician + optional Wi-Fi) -> NB PHY RX -> bytes
    """
    try:
        from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
        from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from NarrowBand.TX_NarrowBand import OQPSK_SF32_Tx
        from NarrowBand.RX_NarrowBand import OQPSK_SF32_Rx

    initiator = Initiator()
    responder = Responder()

    tx_i = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    rx_i = OQPSK_SF32_Rx(chip_rate_hz=2e6, osr=8)
    tx_r = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)
    rx_r = OQPSK_SF32_Rx(chip_rate_hz=2e6, osr=8)
    fs = tx_i.fs
    fc = 6.5e9

    wifi_tx = None
    if use_wifi_interference:
        WiFiOFDMTx = _load_wifi_tx_class()
        wifi_tx = WiFiOFDMTx(rng_seed=9001, center_freq_hz=6.52e9)

    print("=== PHY-backed Compact Handshake Demo ===")
    print(
        f"distance={distance_m:.1f} m, retries={max_poll_retries}, "
        f"wifi_interference={'ON' if use_wifi_interference else 'OFF'}"
    )

    for attempt in range(1, max_poll_retries + 1):
        poll = initiator.initiator_tx()
        print(f"[I] TX ADV-POLL attempt={attempt}, len={len(poll)}, crc_ok={verify_crc(poll)}")
        poll_at_r = _transmit_compact_frame_over_nb_phy(
            frame_bytes=poll,
            tx_nb=tx_i,
            rx_nb=rx_r,
            fs_hz=fs,
            fc_hz=fc,
            distance_m=distance_m,
            nb_tx_eirp_dbw=-30.0,
            nf_db=6.0,
            seed_base=seed_base + 10 * attempt,
            use_wifi_interference=use_wifi_interference,
            wifi_tx=wifi_tx,
            center_freq_gap_mhz=20.0,
        )
        if poll_at_r is None:
            print("[R] RX poll: PHY decode failed")
            continue

        resp = responder.responder_rx(poll_at_r)
        if resp is None:
            print("[R] RX poll: CRC/parse fail, no response")
            continue
        print(f"[R] TX ADV-RESP len={len(resp)}, crc_ok={verify_crc(resp)}")

        resp_at_i = _transmit_compact_frame_over_nb_phy(
            frame_bytes=resp,
            tx_nb=tx_r,
            rx_nb=rx_i,
            fs_hz=fs,
            fc_hz=fc,
            distance_m=distance_m,
            nb_tx_eirp_dbw=-30.0,
            nf_db=6.0,
            seed_base=seed_base + 10000 + 10 * attempt,
            use_wifi_interference=use_wifi_interference,
            wifi_tx=wifi_tx,
            center_freq_gap_mhz=20.0,
        )
        if resp_at_i is None:
            print("[I] RX resp: PHY decode failed")
            continue

        try:
            conf = initiator.initiator_rx(resp_at_i)
        except (FrameCrcError, FrameParseError) as exc:
            print(f"[I] RX resp: CRC/parse fail: {exc}")
            continue

        if conf is None:
            print("[I] No CONF generated")
            return
        print(f"[I] TX ADV-CONF len={len(conf)}, crc_ok={verify_crc(conf)}")

        conf_at_r = _transmit_compact_frame_over_nb_phy(
            frame_bytes=conf,
            tx_nb=tx_i,
            rx_nb=rx_r,
            fs_hz=fs,
            fc_hz=fc,
            distance_m=distance_m,
            nb_tx_eirp_dbw=-30.0,
            nf_db=6.0,
            seed_base=seed_base + 20000 + 10 * attempt,
            use_wifi_interference=use_wifi_interference,
            wifi_tx=wifi_tx,
            center_freq_gap_mhz=20.0,
        )
        if conf_at_r is None:
            print("[R] RX conf: PHY decode failed")
            continue

        try:
            parsed_conf = decode_adv_conf(conf_at_r)
        except (FrameCrcError, FrameParseError) as exc:
            print(f"[R] RX conf: CRC/parse fail: {exc}")
            continue

        print(
            "[R] Handshake complete: "
            f"msg_id={parsed_conf.message_id}, responder_addr={parsed_conf.responder_addr.hex()}"
        )
        return {"success": True, "attempts_used": attempt}

    print("[I/R] Handshake failed after retry budget")
    return {"success": False, "attempts_used": max_poll_retries}


if __name__ == "__main__":
    handshake_demo()

    print("\n--- PHY Demo (Wi-Fi OFF, one run) ---")
    off_result = handshake_demo_over_phy(
        distance_m=10.0,
        max_poll_retries=5,
        use_wifi_interference=False,
        seed_base=10000,
    )
    print(f"[SUMMARY OFF] success={off_result['success']}, attempts={off_result['attempts_used']}")

    print("\n--- PHY Demo (Wi-Fi ON, keep trying) ---")
    # Try with a larger retry budget and report the first successful attempt number.
    on_result = handshake_demo_over_phy(
        distance_m=10.0,
        max_poll_retries=200,
        use_wifi_interference=True,
        seed_base=10000,
    )
    if on_result["success"]:
        print(f"[SUMMARY ON] first success at attempt={on_result['attempts_used']}")
    else:
        print(f"[SUMMARY ON] no success within attempts={on_result['attempts_used']}")
