import unittest
from dataclasses import replace

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
    parse_tlvs,
    unpack_message_id,
)


class TestMessageId(unittest.TestCase):
    def test_message_id_pack_unpack(self):
        b = pack_message_id(message_control=1, message_version=0)
        mc, mv = unpack_message_id(b)
        self.assertEqual(mc, 1)
        self.assertEqual(mv, 0)


class TestSmidTlv(unittest.TestCase):
    def test_smid_tlv_roundtrip(self):
        tlvs = [TLV(tag=CFID_ADV_RESP, value=b"\x01\x02"), TLV(tag=CFID_ADV_CONF, value=b"\x10")]
        packed = b"".join(t.pack() for t in tlvs)
        parsed = parse_tlvs(packed)
        self.assertEqual(tlvs, parsed)


class TestAdvPoll(unittest.TestCase):
    def test_adv_poll_pack_unpack_mc0(self):
        p = AdvertisingPoll(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\x04\x05\x06",
            rpa_prand=b"\x07\x08\x09",
            message_id=pack_message_id(0, 0),
        )
        b = encode_adv_poll(p, include_fcs=True)
        d = decode_adv_poll(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)

    def test_adv_poll_pack_unpack_mc1(self):
        p = AdvertisingPoll(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\x04\x05\x06",
            rpa_prand=b"\x07\x08\x09",
            message_id=pack_message_id(1, 0),
            init_slot_dur=10,
            cap_dur=11,
            supported_mod_modes=0x03,
            presence_bitmap=0x01,
            smid_tlvs=[TLV(tag=CFID_ADV_RESP, value=b"\x00\x01"), TLV(tag=CFID_ADV_CONF, value=b"\x00")],
        )
        b = encode_adv_poll(p, include_fcs=True)
        d = decode_adv_poll(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)


class TestAdvResp(unittest.TestCase):
    def test_adv_resp_pack_unpack_mc0(self):
        p = AdvertisingResponse(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\xAA\xBB\xCC",
            rpa_prand=b"\x10\x11\x12",
            message_id=pack_message_id(0, 0),
            nb_full_channel_map=b"\x01\x02\x03\x04\x05\x06",
            management_phy_configuration=0x21,
            management_mac_configuration=b"\x31\x32\x33\x34\x35\x36\x37",
            ranging_phy_configuration=b"\x41\x42\x43\x44",
            mms_number_of_fragments=pack_mms_num_frags(2, 1),
        )
        b = encode_adv_resp(p, include_fcs=True)
        d = decode_adv_resp(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)

    def test_adv_resp_pack_unpack_mc1_presence(self):
        p = AdvertisingResponse(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\xAA\xBB\xCC",
            rpa_prand=b"\x10\x11\x12",
            message_id=pack_message_id(1, 0),
            presence_bitmap1=0xF1,  # ext + mgmt phy/mac/ranging + mms num frags
            presence_bitmap2=0x03,  # mms mode + supported modulation
            management_phy_configuration=0x20,
            management_mac_configuration=b"\x01\x02\x03\x04\x05\x06\x07",
            ranging_phy_configuration=b"\x11\x12\x13\x14",
            mms_number_of_fragments=pack_mms_num_frags(1, 1),
            mms_ranging_mode_configuration=0x01,
            supported_oqpsk_modulation_modes=0x03,
            smid_tlvs=[TLV(tag=CFID_ADV_POLL, value=b"\x01"), TLV(tag=CFID_ADV_RESP, value=b"\x00\x01")],
        )
        b = encode_adv_resp(p, include_fcs=True)
        d = decode_adv_resp(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)


class TestAdvConf(unittest.TestCase):
    def test_adv_conf_pack_unpack_mc0(self):
        p = AdvertisingConfirmation(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\xAA\xBB\xCC",
            rpa_prand=b"\x10\x11\x12",
            message_id=pack_message_id(0, 0),
            responder_address=b"\x21\x22\x23",
            sor_time_offset=b"\x01\x02\x03",
        )
        b = encode_adv_conf(p, include_fcs=True)
        d = decode_adv_conf(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)

    def test_adv_conf_pack_unpack_mc1_multi(self):
        p = AdvertisingConfirmation(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\xAA\xBB\xCC",
            rpa_prand=b"\x10\x11\x12",
            message_id=pack_message_id(1, 0),
            number_of_responders=2,
            responder_sor_entries=[
                (b"\x21\x22\x23", b"\x01\x02\x03"),
                (b"\x31\x32\x33", b"\x04\x05\x06"),
            ],
        )
        b = encode_adv_conf(p, include_fcs=True)
        d = decode_adv_conf(b, fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)


if __name__ == "__main__":
    unittest.main()
