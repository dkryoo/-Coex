from __future__ import annotations

from dataclasses import dataclass
from typing import List


CFID_ADV_POLL = 0x01
CFID_ADV_RESP = 0x02
CFID_ADV_CONF = 0x03


class FrameCrcError(ValueError):
    pass


class FrameParseError(ValueError):
    pass


@dataclass(frozen=True)
class SmidTlv:
    tag: int
    values: bytes

    def __post_init__(self) -> None:
        if not (0 <= self.tag <= 0xFF):
            raise ValueError("SMID tag must fit in 1 octet")
        if len(self.values) > 0xFF:
            raise ValueError("SMID values length must fit in 1 octet")


@dataclass(frozen=True)
class AdvPoll:
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


def encode_adv_poll(pkt: AdvPoll) -> bytes:
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
    body = bytearray([CFID_ADV_CONF, 0x00])
    body.extend(pkt.rpa_hash)
    body.append(pkt.message_id & 0xFF)
    body.extend(pkt.responder_addr)
    body.extend(pkt.sor_time_offset)
    return _append_fcs(bytes(body))


def decode_adv_conf(frame: bytes) -> AdvConf:
    body = _check_crc_or_raise(frame)
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
