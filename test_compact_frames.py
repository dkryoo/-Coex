import unittest

from packet.narrowband_compact_frames import (
    AdvConf,
    AdvPoll,
    AdvResp,
    CFID_ADV_CONF,
    CFID_ADV_RESP,
    FrameCrcError,
    SmidTlv,
    crc16_802154,
    decode_adv_conf,
    decode_adv_poll,
    decode_adv_resp,
    encode_adv_conf,
    encode_adv_poll,
    encode_adv_resp,
    verify_crc,
)


class TestCrc(unittest.TestCase):
    def test_crc_known_vector(self):
        # CRC-16/KERMIT style result used by IEEE 802.15.4 reflected implementation.
        self.assertEqual(crc16_802154(b"123456789"), 0x2189)

    def test_verify_crc(self):
        p = AdvPoll(1, 2, 3, [])
        b = encode_adv_poll(p)
        self.assertTrue(verify_crc(b))

        bad = bytearray(b)
        bad[2] ^= 0x01
        self.assertFalse(verify_crc(bytes(bad)))


class TestRoundTrip(unittest.TestCase):
    def test_adv_poll_roundtrip(self):
        p = AdvPoll(
            init_slot_dur=10,
            cap_dur=20,
            supported_mod_modes=0x03,
            smid_tlvs=[
                SmidTlv(tag=CFID_ADV_RESP, values=bytes([0x01, 0x02, 0x03])),
                SmidTlv(tag=CFID_ADV_CONF, values=bytes([0x10])),
            ],
        )
        b = encode_adv_poll(p)
        d = decode_adv_poll(b)
        self.assertEqual(d, p)

    def test_adv_resp_roundtrip(self):
        p = AdvResp(
            rpa_hash=bytes([1, 2, 3]),
            message_id=7,
            nb_full_channel_map=bytes([11, 12, 13, 14, 15, 16]),
            mgmt_phy_cfg=0x21,
            mgmt_mac_cfg=bytes([31, 32, 33, 34, 35, 36, 37]),
            ranging_phy_cfg=bytes([41, 42, 43, 44]),
            mms_num_frags=2,
        )
        b = encode_adv_resp(p)
        d = decode_adv_resp(b)
        self.assertEqual(d, p)

    def test_adv_conf_roundtrip(self):
        p = AdvConf(
            rpa_hash=bytes([0xAA, 0xBB, 0xCC]),
            message_id=0x42,
            responder_addr=bytes([0x11, 0x22, 0x33]),
            sor_time_offset=bytes([0x01, 0x02, 0x03]),
        )
        b = encode_adv_conf(p)
        d = decode_adv_conf(b)
        self.assertEqual(d, p)


class TestSmidTlvParsing(unittest.TestCase):
    def test_multiple_variable_tlvs(self):
        p = AdvPoll(
            init_slot_dur=1,
            cap_dur=2,
            supported_mod_modes=3,
            smid_tlvs=[
                SmidTlv(tag=0x10, values=b"\x01\x02\x03\x04"),
                SmidTlv(tag=0x20, values=b"\xAA"),
                SmidTlv(tag=0x30, values=b""),
            ],
        )
        b = encode_adv_poll(p)
        d = decode_adv_poll(b)
        self.assertEqual(len(d.smid_tlvs), 3)
        self.assertEqual(d.smid_tlvs[0].values, b"\x01\x02\x03\x04")
        self.assertEqual(d.smid_tlvs[1].values, b"\xAA")
        self.assertEqual(d.smid_tlvs[2].values, b"")

    def test_crc_fail_raises(self):
        p = AdvPoll(1, 2, 3, [SmidTlv(tag=1, values=b"\x01")])
        b = bytearray(encode_adv_poll(p))
        b[-1] ^= 0xFF
        with self.assertRaises(FrameCrcError):
            decode_adv_poll(bytes(b))


if __name__ == "__main__":
    unittest.main()
