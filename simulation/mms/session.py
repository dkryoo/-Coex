from __future__ import annotations

from dataclasses import dataclass

from packet.uwb_compact_frames import (
    AdvertisingConfirmation,
    AdvertisingPoll,
    AdvertisingResponse,
    CFID_ADV_CONF,
    CFID_ADV_POLL,
    CFID_ADV_RESP,
    TLV,
    decode_adv_conf,
    decode_adv_poll,
    decode_adv_resp,
    encode_adv_conf,
    encode_adv_poll,
    encode_adv_resp,
    pack_message_id,
    pack_mms_num_frags,
    unpack_mms_num_frags,
)
from UWB.mms_uwb_packet_mode import C_MPS, RSTU_S, MmsUwbConfig, build_mms_uwb_fragments, propagation_delay_rstu


@dataclass
class Initiator:
    initiator_rpa_hash: bytes = b"\x10\x20\x30"
    responder_rpa_hash: bytes = b"\xAA\xBB\xCC"
    rpa_prand: bytes = b"\x01\x02\x03"

    def build_adv_poll(self) -> bytes:
        pkt = AdvertisingPoll(
            initiator_rpa_hash=self.initiator_rpa_hash,
            responder_rpa_hash=self.responder_rpa_hash,
            rpa_prand=self.rpa_prand,
            message_id=pack_message_id(message_control=1, message_version=0),
            init_slot_dur=10,
            cap_dur=8,
            supported_mod_modes=0x03,
            presence_bitmap=0x01,
            smid_tlvs=[
                TLV(tag=CFID_ADV_RESP, value=b"\x00\x01"),
                TLV(tag=CFID_ADV_CONF, value=b"\x00\x01"),
            ],
        )
        return encode_adv_poll(pkt, include_fcs=True)

    def build_adv_conf(self, responder: AdvertisingResponse) -> bytes:
        pkt = AdvertisingConfirmation(
            initiator_rpa_hash=self.initiator_rpa_hash,
            responder_rpa_hash=self.responder_rpa_hash,
            rpa_prand=self.rpa_prand,
            message_id=pack_message_id(message_control=1, message_version=0),
            number_of_responders=1,
            responder_sor_entries=[(responder.responder_rpa_hash, b"\x05\x00\x00")],
        )
        return encode_adv_conf(pkt, include_fcs=True)


@dataclass
class Responder:
    initiator_rpa_hash: bytes = b"\x10\x20\x30"
    responder_rpa_hash: bytes = b"\xAA\xBB\xCC"
    rpa_prand: bytes = b"\x01\x02\x03"

    def on_adv_poll(self, frame: bytes) -> bytes:
        poll = decode_adv_poll(frame, fcs_mode="required")
        print(
            "[Responder] RX ADV-POLL:"
            f" mc={(poll.message_id >> 4) & 0x0F}, init_slot={poll.init_slot_dur},"
            f" cap={poll.cap_dur}, tlvs={len(poll.smid_tlvs)}"
        )
        pb1 = 0x80 | 0x10 | 0x01
        pb2 = 0x03
        pkt = AdvertisingResponse(
            initiator_rpa_hash=self.initiator_rpa_hash,
            responder_rpa_hash=self.responder_rpa_hash,
            rpa_prand=self.rpa_prand,
            message_id=pack_message_id(message_control=1, message_version=0),
            presence_bitmap1=pb1,
            presence_bitmap2=pb2,
            management_phy_configuration=0x21,
            mms_number_of_fragments=pack_mms_num_frags(rsfi=2, rifi=1),
            mms_ranging_mode_configuration=0x01,
            supported_oqpsk_modulation_modes=0x03,
            smid_tlvs=[TLV(tag=CFID_ADV_POLL, value=b"\x00\x01")],
        )
        return encode_adv_resp(pkt, include_fcs=True)

    def on_adv_conf(self, frame: bytes) -> AdvertisingConfirmation:
        conf = decode_adv_conf(frame, fcs_mode="required")
        print(
            "[Responder] RX ADV-CONF:"
            f" mc={(conf.message_id >> 4) & 0x0F}, n_resp={conf.number_of_responders},"
            f" entries={len(conf.responder_sor_entries)}"
        )
        return conf


def _hex(b: bytes) -> str:
    return b.hex()


def _first_n_rmarkers(fragments: list, kind: str, n: int) -> list[int]:
    out = [f.rmarker_rstu for f in fragments if f.kind == kind]
    if len(out) < n:
        raise RuntimeError(f"Need at least {n} {kind} fragments")
    return out[:n]


def _estimate_ss_twr(i_tx1: int, i_rx1: int, r_rx1: int, r_tx1: int) -> tuple[float, float]:
    round_trip = i_rx1 - i_tx1
    reply = r_tx1 - r_rx1
    tof_rstu = max(0.0, (round_trip - reply) / 2.0)
    tof_s = tof_rstu * RSTU_S
    return tof_s, tof_s * C_MPS


def _estimate_ds_twr(
    i_tx1: int,
    i_rx1: int,
    r_rx1: int,
    r_tx1: int,
    i_tx2: int,
    i_rx2: int,
    r_rx2: int,
    r_tx2: int,
) -> tuple[float, float]:
    ra = i_rx1 - i_tx1
    rb = i_rx2 - i_tx2
    da = r_tx1 - r_rx1
    db = r_tx2 - r_rx2
    denom = ra + rb + da + db
    if denom <= 0:
        return 0.0, 0.0
    tof_rstu = max(0.0, (ra * rb - da * db) / denom)
    tof_s = tof_rstu * RSTU_S
    return tof_s, tof_s * C_MPS


def run_demo(distance_m: float = 300.0) -> None:
    print("=== MMS Compact Handshake + Symbolic UWB Ranging Demo ===")
    initiator = Initiator()
    responder = Responder()

    adv_poll_b = initiator.build_adv_poll()
    print(f"[Initiator] TX ADV-POLL bytes={_hex(adv_poll_b)}")

    adv_resp_b = responder.on_adv_poll(adv_poll_b)
    print(f"[Responder] TX ADV-RESP bytes={_hex(adv_resp_b)}")
    adv_resp = decode_adv_resp(adv_resp_b, fcs_mode="required")
    print(
        "[Initiator] RX ADV-RESP:"
        f" pb1=0x{adv_resp.presence_bitmap1:02x}, pb2=0x{adv_resp.presence_bitmap2:02x},"
        f" mms_num=0x{adv_resp.mms_number_of_fragments:02x}"
    )

    adv_conf_b = initiator.build_adv_conf(adv_resp)
    print(f"[Initiator] TX ADV-CONF bytes={_hex(adv_conf_b)}")
    responder.on_adv_conf(adv_conf_b)

    frag_cfg = unpack_mms_num_frags(int(adv_resp.mms_number_of_fragments))
    cfg = MmsUwbConfig(
        phy_uwb_mms_rsf_number_frags=frag_cfg["rsf_nfrags"],
        phy_uwb_mms_rif_number_frags=frag_cfg["rif_nfrags"],
    )
    print(
        "[Ranging Config]"
        f" RSF fragments={cfg.phy_uwb_mms_rsf_number_frags},"
        f" RIF fragments={cfg.phy_uwb_mms_rif_number_frags},"
        f" RSTU={RSTU_S*1e9:.3f} ns"
    )

    phase_start = 100_000
    i_frags = build_mms_uwb_fragments("initiator", phase_start, cfg)
    r_frags = build_mms_uwb_fragments("responder", phase_start, cfg)
    delay_rstu = propagation_delay_rstu(distance_m)

    print(
        f"[Propagation] distance={distance_m:.3f} m, delay={delay_rstu} RSTU (~{delay_rstu*RSTU_S*1e9:.2f} ns)"
    )
    print(f"[Initiator] fragments={[(f.kind, f.start_rstu, f.rmarker_rstu) for f in i_frags]}")
    print(f"[Responder] fragments={[(f.kind, f.start_rstu, f.rmarker_rstu) for f in r_frags]}")

    i_rsf = _first_n_rmarkers(i_frags, "RSF", 2)
    r_rsf = _first_n_rmarkers(r_frags, "RSF", 2)
    i_tx1, i_tx2 = i_rsf[0], i_rsf[1]
    r_tx1, r_tx2 = r_rsf[0], r_rsf[1]
    r_rx1, r_rx2 = i_tx1 + delay_rstu, i_tx2 + delay_rstu
    i_rx1, i_rx2 = r_tx1 + delay_rstu, r_tx2 + delay_rstu

    ss_tof_s, ss_dist = _estimate_ss_twr(i_tx1=i_tx1, i_rx1=i_rx1, r_rx1=r_rx1, r_tx1=r_tx1)
    ds_tof_s, ds_dist = _estimate_ds_twr(
        i_tx1=i_tx1,
        i_rx1=i_rx1,
        r_rx1=r_rx1,
        r_tx1=r_tx1,
        i_tx2=i_tx2,
        i_rx2=i_rx2,
        r_rx2=r_rx2,
        r_tx2=r_tx2,
    )

    print(
        "[RMARKER]"
        f" i_tx1={i_tx1}, r_rx1={r_rx1}, r_tx1={r_tx1}, i_rx1={i_rx1},"
        f" i_tx2={i_tx2}, r_rx2={r_rx2}, r_tx2={r_tx2}, i_rx2={i_rx2}"
    )
    print(f"[SS-TWR] ToF={ss_tof_s*1e9:.2f} ns, distance_est={ss_dist:.3f} m")
    print(f"[DS-TWR] ToF={ds_tof_s*1e9:.2f} ns, distance_est={ds_dist:.3f} m")
    print(f"[Truth]  distance={distance_m:.3f} m")
