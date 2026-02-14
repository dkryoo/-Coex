from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# IEEE P802.15.4ab/D03 Sep 2025 Table 10 (subset used here + public variants)
CFID_ADV_POLL = 0x00
CFID_ADV_RESP = 0x01
CFID_ADV_CONF = 0x02
CFID_START_OF_RANGING = 0x03
CFID_ONE_TO_ONE_POLL = 0x04
CFID_ONE_TO_ONE_RESP = 0x05
CFID_ONE_TO_ONE_INIT_REPORT = 0x06
CFID_ONE_TO_ONE_RESP_REPORT = 0x07
CFID_PUBLIC_ADV_POLL = 0x08
CFID_PUBLIC_ADV_RESP = 0x09
CFID_PUBLIC_ADV_CONF = 0x0A
CFID_PUBLIC_START_OF_RANGING = 0x0B


class FrameParseError(ValueError):
    """Raised when compact frame parsing fails."""


class FrameCrcError(ValueError):
    """Raised when FCS is present but invalid."""


@dataclass(frozen=True)
class TLV:
    """SMID TLV (Figure 64): tag(1), length(1), value(length)."""

    tag: int
    value: bytes

    def __post_init__(self) -> None:
        if not (0 <= self.tag <= 0xFF):
            raise ValueError("TLV tag must be 1 octet")
        if len(self.value) > 0xFF:
            raise ValueError("TLV value length must fit in 1 octet")

    def pack(self) -> bytes:
        return bytes([self.tag, len(self.value)]) + self.value


def pack_tlvs(tlvs: Iterable[TLV]) -> bytes:
    out = bytearray()
    for tlv in tlvs:
        out.extend(tlv.pack())
    return bytes(out)


def parse_tlvs(data: bytes) -> list[TLV]:
    tlvs: list[TLV] = []
    idx = 0
    while idx < len(data):
        if idx + 2 > len(data):
            raise FrameParseError("Truncated TLV header")
        tag = data[idx]
        ln = data[idx + 1]
        idx += 2
        if idx + ln > len(data):
            raise FrameParseError("Truncated TLV value")
        tlvs.append(TLV(tag=tag, value=bytes(data[idx : idx + ln])))
        idx += ln
    return tlvs


def pack_message_id(message_control: int, message_version: int = 0) -> int:
    if not (0 <= message_control <= 0x0F):
        raise ValueError("message_control must be 0..15")
    if not (0 <= message_version <= 0x0F):
        raise ValueError("message_version must be 0..15")
    return ((message_control & 0x0F) << 4) | (message_version & 0x0F)


def unpack_message_id(byte: int) -> tuple[int, int]:
    if not (0 <= byte <= 0xFF):
        raise ValueError("message ID byte must be 0..255")
    message_version = byte & 0x0F
    message_control = (byte >> 4) & 0x0F
    return message_control, message_version


def crc16_802154(data: bytes) -> int:
    """
    IEEE 802.15.4 FCS reflected implementation (poly 0x1021 reflected => 0x8408).
    """
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


def append_fcs(frame_wo_fcs: bytes) -> bytes:
    return frame_wo_fcs + crc16_802154(frame_wo_fcs).to_bytes(2, "little")


def verify_crc(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    body = frame[:-2]
    rx = int.from_bytes(frame[-2:], "little")
    return crc16_802154(body) == rx


def _split_body_and_fcs(frame: bytes, fcs_mode: str = "auto") -> tuple[bytes, bytes | None]:
    if fcs_mode not in {"auto", "required", "forbidden"}:
        raise ValueError("fcs_mode must be one of auto/required/forbidden")
    if fcs_mode == "forbidden":
        return frame, None
    if len(frame) >= 3 and verify_crc(frame):
        return frame[:-2], frame[-2:]
    if fcs_mode == "required":
        raise FrameCrcError("FCS/CRC check failed")
    return frame, None


_RSF_FRAG_COUNT_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
_RIF_FRAG_COUNT_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8}


def pack_mms_num_frags(rsfi: int, rifi: int) -> int:
    if rsfi not in range(0, 6):
        raise ValueError("RSF index must be 0..5")
    if rifi not in range(0, 5):
        raise ValueError("RIF index must be 0..4")
    return (rsfi & 0x07) | ((rifi & 0x07) << 3)


def unpack_mms_num_frags(byte: int) -> dict[str, int]:
    rsf_index = byte & 0x07
    rif_index = (byte >> 3) & 0x07
    if rsf_index not in _RSF_FRAG_COUNT_MAP:
        raise FrameParseError(f"Reserved RSF fragment index: {rsf_index}")
    if rif_index not in _RIF_FRAG_COUNT_MAP:
        raise FrameParseError(f"Reserved RIF fragment index: {rif_index}")
    return {
        "rsf_index": rsf_index,
        "rif_index": rif_index,
        "rsf_nfrags": _RSF_FRAG_COUNT_MAP[rsf_index],
        "rif_nfrags": _RIF_FRAG_COUNT_MAP[rif_index],
    }


def _nb_channel_map_len_from_pb(pb1: int) -> int:
    enc = (pb1 >> 2) & 0x03
    return {0: 0, 1: 2, 2: 5, 3: 6}[enc]


def _pack_presence_bitmap(pb1: int, pb2: int | None) -> bytes:
    if not (0 <= pb1 <= 0xFF):
        raise ValueError("pb1 must fit in 1 octet")
    if pb1 & 0x01:
        if pb2 is None:
            raise ValueError("pb2 is required when extended bitmap bit is set")
        if not (0 <= pb2 <= 0xFF):
            raise ValueError("pb2 must fit in 1 octet")
        return bytes([pb1, pb2])
    return bytes([pb1])


def _unpack_presence_bitmap(data: bytes, idx: int) -> tuple[int, int | None, int]:
    if idx >= len(data):
        raise FrameParseError("Missing presence bitmap")
    pb1 = data[idx]
    idx += 1
    pb2 = None
    if pb1 & 0x01:
        if idx >= len(data):
            raise FrameParseError("Missing extended presence bitmap")
        pb2 = data[idx]
        idx += 1
    return pb1, pb2, idx


def _require_len(name: str, value: bytes, n: int) -> None:
    if len(value) != n:
        raise ValueError(f"{name} must be exactly {n} octets")


@dataclass(frozen=True)
class AdvertisingPoll:
    # 10.39.11.3.x (RPA addressing)
    initiator_rpa_hash: bytes
    responder_rpa_hash: bytes
    rpa_prand: bytes
    message_id: int
    # MC=1 fields (None for MC=0)
    init_slot_dur: int | None = None
    cap_dur: int | None = None
    supported_mod_modes: int | None = None
    presence_bitmap: int | None = None
    smid_tlvs: list[TLV] = field(default_factory=list)
    fcs: bytes | None = None

    def __post_init__(self) -> None:
        _require_len("initiator_rpa_hash", self.initiator_rpa_hash, 3)
        _require_len("responder_rpa_hash", self.responder_rpa_hash, 3)
        _require_len("rpa_prand", self.rpa_prand, 3)
        if not (0 <= self.message_id <= 0xFF):
            raise ValueError("message_id must be 1 octet")
        mc, _ = unpack_message_id(self.message_id)
        if mc == 0:
            return
        if mc != 1:
            raise ValueError("AdvertisingPoll supports MC=0 or MC=1 only")
        for name, value in (
            ("init_slot_dur", self.init_slot_dur),
            ("cap_dur", self.cap_dur),
            ("supported_mod_modes", self.supported_mod_modes),
            ("presence_bitmap", self.presence_bitmap),
        ):
            if value is None:
                raise ValueError(f"{name} is required for MC=1")
            if not (0 <= value <= 0xFF):
                raise ValueError(f"{name} must be 1 octet")

    def pack(self, include_fcs: bool = True) -> bytes:
        body = bytearray([CFID_ADV_POLL])
        body.extend(self.initiator_rpa_hash)
        body.extend(self.responder_rpa_hash)
        body.extend(self.rpa_prand)
        body.append(self.message_id)
        mc, _ = unpack_message_id(self.message_id)
        if mc == 1:
            body.extend(
                [
                    int(self.init_slot_dur),
                    int(self.cap_dur),
                    int(self.supported_mod_modes),
                    int(self.presence_bitmap),
                ]
            )
            body.extend(pack_tlvs(self.smid_tlvs))
        return append_fcs(bytes(body)) if include_fcs else bytes(body)

    @classmethod
    def unpack(cls, frame: bytes, fcs_mode: str = "auto") -> "AdvertisingPoll":
        body, fcs = _split_body_and_fcs(frame, fcs_mode=fcs_mode)
        if len(body) < 11:
            raise FrameParseError("AdvertisingPoll too short")
        if body[0] != CFID_ADV_POLL:
            raise FrameParseError("Not AdvertisingPoll CFID")
        idx = 1
        initiator_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        responder_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        rpa_prand = bytes(body[idx : idx + 3])
        idx += 3
        message_id = body[idx]
        idx += 1
        mc, _ = unpack_message_id(message_id)
        if mc == 0:
            if idx != len(body):
                raise FrameParseError("MC=0 AdvertisingPoll must have empty content")
            return cls(
                initiator_rpa_hash=initiator_rpa_hash,
                responder_rpa_hash=responder_rpa_hash,
                rpa_prand=rpa_prand,
                message_id=message_id,
                fcs=fcs,
            )
        if mc != 1:
            raise FrameParseError("AdvertisingPoll supports MC=0/1 only")
        if idx + 4 > len(body):
            raise FrameParseError("MC=1 AdvertisingPoll content too short")
        init_slot_dur = body[idx]
        cap_dur = body[idx + 1]
        supported_mod_modes = body[idx + 2]
        presence_bitmap = body[idx + 3]
        idx += 4
        tlvs = parse_tlvs(body[idx:])
        return cls(
            initiator_rpa_hash=initiator_rpa_hash,
            responder_rpa_hash=responder_rpa_hash,
            rpa_prand=rpa_prand,
            message_id=message_id,
            init_slot_dur=init_slot_dur,
            cap_dur=cap_dur,
            supported_mod_modes=supported_mod_modes,
            presence_bitmap=presence_bitmap,
            smid_tlvs=tlvs,
            fcs=fcs,
        )


@dataclass(frozen=True)
class AdvertisingResponse:
    initiator_rpa_hash: bytes
    responder_rpa_hash: bytes
    rpa_prand: bytes
    message_id: int
    # MC=0 fixed layout fields
    nb_full_channel_map: bytes | None = None
    management_phy_configuration: int | None = None
    management_mac_configuration: bytes | None = None
    ranging_phy_configuration: bytes | None = None
    mms_number_of_fragments: int | None = None
    # MC=1 presence-based fields
    presence_bitmap1: int | None = None
    presence_bitmap2: int | None = None
    nb_channel_map: bytes | None = None
    mms_ranging_mode_configuration: int | None = None
    supported_oqpsk_modulation_modes: int | None = None
    smid_tlvs: list[TLV] = field(default_factory=list)
    fcs: bytes | None = None

    def __post_init__(self) -> None:
        _require_len("initiator_rpa_hash", self.initiator_rpa_hash, 3)
        _require_len("responder_rpa_hash", self.responder_rpa_hash, 3)
        _require_len("rpa_prand", self.rpa_prand, 3)
        if not (0 <= self.message_id <= 0xFF):
            raise ValueError("message_id must be 1 octet")
        mc, _ = unpack_message_id(self.message_id)
        if mc == 0:
            if self.nb_full_channel_map is None or len(self.nb_full_channel_map) != 6:
                raise ValueError("MC=0 requires nb_full_channel_map(6)")
            if self.management_phy_configuration is None:
                raise ValueError("MC=0 requires management_phy_configuration(1)")
            if not (0 <= self.management_phy_configuration <= 0xFF):
                raise ValueError("management_phy_configuration must be 1 octet")
            if self.management_mac_configuration is None or len(self.management_mac_configuration) != 7:
                raise ValueError("MC=0 requires management_mac_configuration(7)")
            if self.ranging_phy_configuration is None or len(self.ranging_phy_configuration) != 4:
                raise ValueError("MC=0 requires ranging_phy_configuration(4)")
            if self.mms_number_of_fragments is None or not (0 <= self.mms_number_of_fragments <= 0xFF):
                raise ValueError("MC=0 requires mms_number_of_fragments(1)")
            return
        if mc != 1:
            raise ValueError("AdvertisingResponse supports MC=0 or MC=1 only")
        if self.presence_bitmap1 is None or not (0 <= self.presence_bitmap1 <= 0xFF):
            raise ValueError("MC=1 requires presence_bitmap1")
        if (self.presence_bitmap1 & 0x01) and (
            self.presence_bitmap2 is None or not (0 <= self.presence_bitmap2 <= 0xFF)
        ):
            raise ValueError("MC=1 extended bitmap requires presence_bitmap2")

    def pack(self, include_fcs: bool = True) -> bytes:
        body = bytearray([CFID_ADV_RESP])
        body.extend(self.initiator_rpa_hash)
        body.extend(self.responder_rpa_hash)
        body.extend(self.rpa_prand)
        body.append(self.message_id)
        mc, _ = unpack_message_id(self.message_id)
        if mc == 0:
            body.extend(self.nb_full_channel_map or b"")
            body.append(int(self.management_phy_configuration))
            body.extend(self.management_mac_configuration or b"")
            body.extend(self.ranging_phy_configuration or b"")
            body.append(int(self.mms_number_of_fragments))
        else:
            body.extend(_pack_presence_bitmap(int(self.presence_bitmap1), self.presence_bitmap2))
            pb1 = int(self.presence_bitmap1)
            nb_len = _nb_channel_map_len_from_pb(pb1)
            if nb_len > 0:
                if self.nb_channel_map is None or len(self.nb_channel_map) != nb_len:
                    raise ValueError(f"MC=1 requires nb_channel_map({nb_len}) per bitmap")
                body.extend(self.nb_channel_map)
            if pb1 & (1 << 4):
                body.append(int(self.management_phy_configuration))
            if pb1 & (1 << 5):
                mmc = self.management_mac_configuration or b""
                if len(mmc) != 7:
                    raise ValueError("MC=1 management_mac_configuration must be 7 octets")
                body.extend(mmc)
            if pb1 & (1 << 6):
                rpc = self.ranging_phy_configuration or b""
                if len(rpc) != 4:
                    raise ValueError("MC=1 ranging_phy_configuration must be 4 octets")
                body.extend(rpc)
            if pb1 & (1 << 7):
                body.append(int(self.mms_number_of_fragments))
            pb2 = self.presence_bitmap2 if (pb1 & 0x01) else None
            if pb2 is not None and (pb2 & (1 << 0)):
                body.append(int(self.mms_ranging_mode_configuration))
            if pb2 is not None and (pb2 & (1 << 1)):
                body.append(int(self.supported_oqpsk_modulation_modes))
            body.extend(pack_tlvs(self.smid_tlvs))
        return append_fcs(bytes(body)) if include_fcs else bytes(body)

    @classmethod
    def unpack(cls, frame: bytes, fcs_mode: str = "auto") -> "AdvertisingResponse":
        body, fcs = _split_body_and_fcs(frame, fcs_mode=fcs_mode)
        if len(body) < 11:
            raise FrameParseError("AdvertisingResponse too short")
        if body[0] != CFID_ADV_RESP:
            raise FrameParseError("Not AdvertisingResponse CFID")
        idx = 1
        initiator_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        responder_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        rpa_prand = bytes(body[idx : idx + 3])
        idx += 3
        message_id = body[idx]
        idx += 1
        mc, _ = unpack_message_id(message_id)
        if mc == 0:
            if idx + 19 != len(body):
                raise FrameParseError("MC=0 AdvertisingResponse must have 19-octet content")
            nb_full_channel_map = bytes(body[idx : idx + 6])
            idx += 6
            management_phy_configuration = body[idx]
            idx += 1
            management_mac_configuration = bytes(body[idx : idx + 7])
            idx += 7
            ranging_phy_configuration = bytes(body[idx : idx + 4])
            idx += 4
            mms_number_of_fragments = body[idx]
            return cls(
                initiator_rpa_hash=initiator_rpa_hash,
                responder_rpa_hash=responder_rpa_hash,
                rpa_prand=rpa_prand,
                message_id=message_id,
                nb_full_channel_map=nb_full_channel_map,
                management_phy_configuration=management_phy_configuration,
                management_mac_configuration=management_mac_configuration,
                ranging_phy_configuration=ranging_phy_configuration,
                mms_number_of_fragments=mms_number_of_fragments,
                fcs=fcs,
            )
        if mc != 1:
            raise FrameParseError("AdvertisingResponse supports MC=0/1 only")
        pb1, pb2, idx = _unpack_presence_bitmap(body, idx)
        nb_len = _nb_channel_map_len_from_pb(pb1)
        nb_channel_map = None
        if nb_len:
            if idx + nb_len > len(body):
                raise FrameParseError("Truncated NB channel map")
            nb_channel_map = bytes(body[idx : idx + nb_len])
            idx += nb_len
        management_phy_configuration = None
        if pb1 & (1 << 4):
            if idx >= len(body):
                raise FrameParseError("Missing management PHY configuration")
            management_phy_configuration = body[idx]
            idx += 1
        management_mac_configuration = None
        if pb1 & (1 << 5):
            if idx + 7 > len(body):
                raise FrameParseError("Missing management MAC configuration")
            management_mac_configuration = bytes(body[idx : idx + 7])
            idx += 7
        ranging_phy_configuration = None
        if pb1 & (1 << 6):
            if idx + 4 > len(body):
                raise FrameParseError("Missing ranging PHY configuration")
            ranging_phy_configuration = bytes(body[idx : idx + 4])
            idx += 4
        mms_number_of_fragments = None
        if pb1 & (1 << 7):
            if idx >= len(body):
                raise FrameParseError("Missing MMS number of fragments")
            mms_number_of_fragments = body[idx]
            idx += 1
        mms_ranging_mode_configuration = None
        supported_oqpsk_modulation_modes = None
        if pb2 is not None and (pb2 & (1 << 0)):
            if idx >= len(body):
                raise FrameParseError("Missing MMS ranging mode configuration")
            mms_ranging_mode_configuration = body[idx]
            idx += 1
        if pb2 is not None and (pb2 & (1 << 1)):
            if idx >= len(body):
                raise FrameParseError("Missing supported O-QPSK modulation modes")
            supported_oqpsk_modulation_modes = body[idx]
            idx += 1
        smid_tlvs = parse_tlvs(body[idx:])
        return cls(
            initiator_rpa_hash=initiator_rpa_hash,
            responder_rpa_hash=responder_rpa_hash,
            rpa_prand=rpa_prand,
            message_id=message_id,
            presence_bitmap1=pb1,
            presence_bitmap2=pb2,
            nb_channel_map=nb_channel_map,
            management_phy_configuration=management_phy_configuration,
            management_mac_configuration=management_mac_configuration,
            ranging_phy_configuration=ranging_phy_configuration,
            mms_number_of_fragments=mms_number_of_fragments,
            mms_ranging_mode_configuration=mms_ranging_mode_configuration,
            supported_oqpsk_modulation_modes=supported_oqpsk_modulation_modes,
            smid_tlvs=smid_tlvs,
            fcs=fcs,
        )


@dataclass(frozen=True)
class AdvertisingConfirmation:
    initiator_rpa_hash: bytes
    responder_rpa_hash: bytes
    rpa_prand: bytes
    message_id: int
    responder_address: bytes | None = None
    sor_time_offset: bytes | None = None
    number_of_responders: int | None = None
    responder_sor_entries: list[tuple[bytes, bytes]] = field(default_factory=list)
    fcs: bytes | None = None

    def __post_init__(self) -> None:
        _require_len("initiator_rpa_hash", self.initiator_rpa_hash, 3)
        _require_len("responder_rpa_hash", self.responder_rpa_hash, 3)
        _require_len("rpa_prand", self.rpa_prand, 3)
        if not (0 <= self.message_id <= 0xFF):
            raise ValueError("message_id must be 1 octet")
        mc, _ = unpack_message_id(self.message_id)
        if mc == 0:
            if self.responder_address is None or len(self.responder_address) != 3:
                raise ValueError("MC=0 requires responder_address(3)")
            if self.sor_time_offset is None or len(self.sor_time_offset) != 3:
                raise ValueError("MC=0 requires sor_time_offset(3)")
            return
        if mc != 1:
            raise ValueError("AdvertisingConfirmation supports MC=0 or MC=1 only")
        if self.number_of_responders is None or not (0 <= self.number_of_responders <= 0xFF):
            raise ValueError("MC=1 requires number_of_responders")
        if self.number_of_responders != len(self.responder_sor_entries):
            raise ValueError("number_of_responders must match responder_sor_entries length")
        for addr, sor in self.responder_sor_entries:
            _require_len("responder_address entry", addr, 3)
            _require_len("sor_time_offset entry", sor, 3)

    def pack(self, include_fcs: bool = True) -> bytes:
        body = bytearray([CFID_ADV_CONF])
        body.extend(self.initiator_rpa_hash)
        body.extend(self.responder_rpa_hash)
        body.extend(self.rpa_prand)
        body.append(self.message_id)
        mc, _ = unpack_message_id(self.message_id)
        if mc == 0:
            body.extend(self.responder_address or b"")
            body.extend(self.sor_time_offset or b"")
        else:
            body.append(int(self.number_of_responders))
            for addr, sor in self.responder_sor_entries:
                body.extend(addr)
                body.extend(sor)
        return append_fcs(bytes(body)) if include_fcs else bytes(body)

    @classmethod
    def unpack(cls, frame: bytes, fcs_mode: str = "auto") -> "AdvertisingConfirmation":
        body, fcs = _split_body_and_fcs(frame, fcs_mode=fcs_mode)
        if len(body) < 11:
            raise FrameParseError("AdvertisingConfirmation too short")
        if body[0] != CFID_ADV_CONF:
            raise FrameParseError("Not AdvertisingConfirmation CFID")
        idx = 1
        initiator_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        responder_rpa_hash = bytes(body[idx : idx + 3])
        idx += 3
        rpa_prand = bytes(body[idx : idx + 3])
        idx += 3
        message_id = body[idx]
        idx += 1
        mc, _ = unpack_message_id(message_id)
        if mc == 0:
            if idx + 6 != len(body):
                raise FrameParseError("MC=0 AdvertisingConfirmation content length must be 6")
            responder_address = bytes(body[idx : idx + 3])
            idx += 3
            sor_time_offset = bytes(body[idx : idx + 3])
            return cls(
                initiator_rpa_hash=initiator_rpa_hash,
                responder_rpa_hash=responder_rpa_hash,
                rpa_prand=rpa_prand,
                message_id=message_id,
                responder_address=responder_address,
                sor_time_offset=sor_time_offset,
                fcs=fcs,
            )
        if mc != 1:
            raise FrameParseError("AdvertisingConfirmation supports MC=0/1 only")
        if idx >= len(body):
            raise FrameParseError("Missing number_of_responders")
        number_of_responders = body[idx]
        idx += 1
        expected = idx + number_of_responders * 6
        if expected != len(body):
            raise FrameParseError("Responder SOR list length mismatch")
        entries: list[tuple[bytes, bytes]] = []
        for _ in range(number_of_responders):
            addr = bytes(body[idx : idx + 3])
            idx += 3
            sor = bytes(body[idx : idx + 3])
            idx += 3
            entries.append((addr, sor))
        return cls(
            initiator_rpa_hash=initiator_rpa_hash,
            responder_rpa_hash=responder_rpa_hash,
            rpa_prand=rpa_prand,
            message_id=message_id,
            number_of_responders=number_of_responders,
            responder_sor_entries=entries,
            fcs=fcs,
        )


def encode_adv_poll(pkt: AdvertisingPoll, include_fcs: bool = True) -> bytes:
    return pkt.pack(include_fcs=include_fcs)


def decode_adv_poll(frame: bytes, fcs_mode: str = "auto") -> AdvertisingPoll:
    return AdvertisingPoll.unpack(frame, fcs_mode=fcs_mode)


def encode_adv_resp(pkt: AdvertisingResponse, include_fcs: bool = True) -> bytes:
    return pkt.pack(include_fcs=include_fcs)


def decode_adv_resp(frame: bytes, fcs_mode: str = "auto") -> AdvertisingResponse:
    return AdvertisingResponse.unpack(frame, fcs_mode=fcs_mode)


def encode_adv_conf(pkt: AdvertisingConfirmation, include_fcs: bool = True) -> bytes:
    return pkt.pack(include_fcs=include_fcs)


def decode_adv_conf(frame: bytes, fcs_mode: str = "auto") -> AdvertisingConfirmation:
    return AdvertisingConfirmation.unpack(frame, fcs_mode=fcs_mode)

