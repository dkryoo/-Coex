from __future__ import annotations

import unittest

from simulation.mms.nb_channel_switching import (
    NbChannelSwitchConfig,
    build_nb_channel_allow_list,
    selected_nb_channel_for_block,
)


class TestNbChannelSwitching(unittest.TestCase):
    def test_seed0_allow_1_to_5_first_8(self) -> None:
        allow = [1, 2, 3, 4, 5]
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=tuple(allow),
            mms_prng_seed=0,
            channel_switching_field=1,
        )
        seq = [selected_nb_channel_for_block(cfg, i) for i in range(8)]
        self.assertEqual(seq, [4, 5, 1, 5, 1, 5, 5, 5])

    def test_switching_field_zero_uses_lowest(self) -> None:
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=(9, 3, 12),
            mms_prng_seed=0,
            channel_switching_field=0,
        )
        seq = [selected_nb_channel_for_block(cfg, i) for i in range(8)]
        self.assertEqual(seq, [3] * 8)

    def test_allow_list_from_step_and_bitmask_intersection(self) -> None:
        # Bitmask allows channels {0,1,2,3,4,5}; step-set with start=1, step=2 -> {1,3,5,...}
        # Intersection should be {1,3,5}.
        bitmask = sum(1 << k for k in [0, 1, 2, 3, 4, 5])
        allow = build_nb_channel_allow_list(
            explicit_allow_list=None,
            nb_channel_start=1,
            nb_channel_step=2,
            nb_channel_bitmask_hex=hex(bitmask),
        )
        self.assertEqual(allow, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
