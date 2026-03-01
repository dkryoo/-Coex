import unittest
from dataclasses import replace
import numpy as np

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
from packet.uwb_compact_frames import (
    AdvertisingConfirmation,
    AdvertisingPoll,
    AdvertisingResponse,
    CFID_ADV_POLL,
    CFID_ADV_RESP as UWB_CFID_ADV_RESP,
    TLV,
    decode_adv_conf as decode_uwb_adv_conf,
    decode_adv_poll as decode_uwb_adv_poll,
    decode_adv_resp as decode_uwb_adv_resp,
    encode_adv_conf as encode_uwb_adv_conf,
    encode_adv_poll as encode_uwb_adv_poll,
    encode_adv_resp as encode_uwb_adv_resp,
    pack_message_id,
    pack_mms_num_frags,
    parse_tlvs,
    unpack_message_id,
)
from UWB.mms_uwb_packet_mode import MmsUwbConfig, build_mms_uwb_fragments, propagation_delay_rstu
from UWB.uwb_modem import run_uwb_modem_ber_test
from Channel.Interference import alias_frequency
from simulation.mms.performance import _fft_zero_pad_interp_complex, simulate_mms_performance


class TestNbCompactFrames(unittest.TestCase):
    def test_nb_crc_known_vector(self):
        self.assertEqual(crc16_802154(b"123456789"), 0x2189)

    def test_nb_verify_crc(self):
        p = AdvPoll(1, 2, 3, [])
        b = encode_adv_poll(p)
        self.assertTrue(verify_crc(b))
        bad = bytearray(b)
        bad[2] ^= 0x01
        self.assertFalse(verify_crc(bytes(bad)))

    def test_nb_adv_poll_roundtrip(self):
        p = AdvPoll(
            init_slot_dur=10,
            cap_dur=20,
            supported_mod_modes=0x03,
            smid_tlvs=[
                SmidTlv(tag=CFID_ADV_RESP, values=bytes([0x01, 0x02, 0x03])),
                SmidTlv(tag=CFID_ADV_CONF, values=bytes([0x10])),
            ],
        )
        self.assertEqual(decode_adv_poll(encode_adv_poll(p)), p)

    def test_nb_adv_resp_roundtrip(self):
        p = AdvResp(
            rpa_hash=bytes([1, 2, 3]),
            message_id=7,
            nb_full_channel_map=bytes([11, 12, 13, 14, 15, 16]),
            mgmt_phy_cfg=0x21,
            mgmt_mac_cfg=bytes([31, 32, 33, 34, 35, 36, 37]),
            ranging_phy_cfg=bytes([41, 42, 43, 44]),
            mms_num_frags=2,
        )
        self.assertEqual(decode_adv_resp(encode_adv_resp(p)), p)

    def test_nb_adv_conf_roundtrip(self):
        p = AdvConf(
            rpa_hash=bytes([0xAA, 0xBB, 0xCC]),
            message_id=0x42,
            responder_addr=bytes([0x11, 0x22, 0x33]),
            sor_time_offset=bytes([0x01, 0x02, 0x03]),
        )
        self.assertEqual(decode_adv_conf(encode_adv_conf(p)), p)

    def test_nb_crc_fail_raises(self):
        p = AdvPoll(1, 2, 3, [SmidTlv(tag=1, values=b"\x01")])
        b = bytearray(encode_adv_poll(p))
        b[-1] ^= 0xFF
        with self.assertRaises(FrameCrcError):
            decode_adv_poll(bytes(b))


class TestUwbCompactFrames(unittest.TestCase):
    def test_uwb_message_id_pack_unpack(self):
        b = pack_message_id(message_control=1, message_version=0)
        mc, mv = unpack_message_id(b)
        self.assertEqual(mc, 1)
        self.assertEqual(mv, 0)

    def test_uwb_smid_tlv_roundtrip(self):
        tlvs = [TLV(tag=UWB_CFID_ADV_RESP, value=b"\x01\x02"), TLV(tag=0x02, value=b"\x10")]
        self.assertEqual(parse_tlvs(b"".join(t.pack() for t in tlvs)), tlvs)

    def test_uwb_adv_poll_pack_unpack(self):
        p = AdvertisingPoll(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\x04\x05\x06",
            rpa_prand=b"\x07\x08\x09",
            message_id=pack_message_id(1, 0),
            init_slot_dur=10,
            cap_dur=11,
            supported_mod_modes=0x03,
            presence_bitmap=0x01,
            smid_tlvs=[TLV(tag=UWB_CFID_ADV_RESP, value=b"\x00\x01")],
        )
        d = decode_uwb_adv_poll(encode_uwb_adv_poll(p, include_fcs=True), fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)

    def test_uwb_adv_resp_pack_unpack(self):
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
        d = decode_uwb_adv_resp(encode_uwb_adv_resp(p, include_fcs=True), fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)

    def test_uwb_adv_conf_pack_unpack(self):
        p = AdvertisingConfirmation(
            initiator_rpa_hash=b"\x01\x02\x03",
            responder_rpa_hash=b"\xAA\xBB\xCC",
            rpa_prand=b"\x10\x11\x12",
            message_id=pack_message_id(0, 0),
            responder_address=b"\x21\x22\x23",
            sor_time_offset=b"\x01\x02\x03",
        )
        d = decode_uwb_adv_conf(encode_uwb_adv_conf(p, include_fcs=True), fcs_mode="required")
        self.assertEqual(replace(d, fcs=None), p)


class TestUwbPacketMode(unittest.TestCase):
    def test_fragment_scheduling_interleaved(self):
        cfg = MmsUwbConfig(phy_uwb_mms_rsf_number_frags=2, phy_uwb_mms_rif_number_frags=1)
        i_frags = build_mms_uwb_fragments("initiator", 1000, cfg)
        r_frags = build_mms_uwb_fragments("responder", 1000, cfg)
        self.assertEqual(i_frags[0].start_rstu, 1000)
        self.assertEqual(r_frags[0].start_rstu, 1600)
        self.assertEqual(i_frags[1].start_rstu - i_frags[0].start_rstu, 1200)
        self.assertEqual(r_frags[1].start_rstu - r_frags[0].start_rstu, 1200)

    def test_propagation_delay_nonnegative(self):
        self.assertEqual(propagation_delay_rstu(0.0), 0)
        self.assertGreaterEqual(propagation_delay_rstu(10.0), 0)


class TestUwbModemSmoke(unittest.TestCase):
    def test_modem_no_channel_smoke(self):
        r = run_uwb_modem_ber_test(
            n_frames=10,
            bits_per_frame=64,
            distance_m=10.0,
            tx_eirp_dbw=-20.0,
            use_channel=False,
            seed=12345,
        )
        self.assertIn("ber", r)
        self.assertIn("fer", r)


class TestInterferenceUtils(unittest.TestCase):
    def test_alias_frequency_wrap(self):
        fs = 100.0
        self.assertAlmostEqual(alias_frequency(10.0, fs), 10.0, places=9)
        self.assertAlmostEqual(alias_frequency(60.0, fs), -40.0, places=9)
        self.assertAlmostEqual(alias_frequency(-60.0, fs), 40.0, places=9)

    def test_db_attenuation_power_relation(self):
        rng = np.random.default_rng(123)
        n = 4096
        t = np.arange(n, dtype=float)
        x = np.exp(1j * 2.0 * np.pi * 0.07 * t) + 0.05 * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        a_db = 25.0
        y = x * (10.0 ** (-a_db / 20.0))
        p_x = float(np.mean(np.abs(x) ** 2))
        p_y = float(np.mean(np.abs(y) ** 2))
        att_db = 10.0 * np.log10((p_x + 1e-30) / (p_y + 1e-30))
        self.assertAlmostEqual(att_db, a_db, places=1)


class TestToaRefinement(unittest.TestCase):
    def test_fft_interp_grid_consistency(self):
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(31) + 1j * rng.standard_normal(31)).astype(np.complex128)
        up = 8
        x_up = _fft_zero_pad_interp_complex(x, up=up, validate=True)
        err = float(np.max(np.abs(x_up[::up] - x)))
        ref = float(np.max(np.abs(x)))
        self.assertLess(err, 1e-6 * max(ref, 1e-12))

    def test_sanity_rmse_small(self):
        r = simulate_mms_performance(
            distances_m=(20.0,),
            n_trials=40,
            seed=20260226,
            wifi_interference_on=False,
            baseline_sanity_mode=True,
            save_psd=False,
            debug_first_trial=False,
            toa_refine_method="fft_upsample",
            corr_upsample=8,
            corr_win=64,
            first_path=False,
            toa_calibration_distance_m=20.0,
            toa_calibration_trials=32,
        )[0]
        self.assertLess(float(r["ranging_rmse_m"]), 0.12)


if __name__ == "__main__":
    unittest.main()
