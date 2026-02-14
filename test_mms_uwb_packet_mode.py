import unittest

from UWB.mms_uwb_packet_mode import (
    MmsUwbConfig,
    build_mms_uwb_fragments,
    propagation_delay_rstu,
)


class TestMmsUwbPacketMode(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
