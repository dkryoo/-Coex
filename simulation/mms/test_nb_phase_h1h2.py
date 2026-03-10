from __future__ import annotations

import json
import unittest

from simulation.mms.check_nb_phase_hopping import _run_case
from simulation.mms.nb_channel_switching import (
    NbChannelSwitchConfig,
    selected_nb_channel_for_block,
    selected_nb_channel_for_phase,
    spec_select_nb_channel_reference,
)


class TestNbPhaseH1H2(unittest.TestCase):
    def test_phase_rules_off_init_fixed_ctrl_lowest(self) -> None:
        row = _run_case(enable_switching=False, allow_list=(0, 1, 2, 3), sim_seed=20260307, mac_seed=0)
        self.assertEqual(int(row["nb_ch_unique_count_init"]), 1)
        self.assertEqual(json.loads(str(row["nb_ch_init_seq_first8_json"]))[:4], [2, 2, 2, 2])
        self.assertEqual(int(row["nb_ch_unique_count_ctrl"]), 1)
        self.assertEqual(json.loads(str(row["nb_ch_ctrl_seq_first8_json"]))[:4], [0, 0, 0, 0])
        # Report phase is not explicitly modeled in current latency runner.
        self.assertEqual(int(row["nb_ch_unique_count_report"]), 0)

    def test_phase_rules_on_init_fixed_ctrl_hops(self) -> None:
        row = _run_case(enable_switching=True, allow_list=(0, 1, 2, 3), sim_seed=20260307, mac_seed=0)
        self.assertEqual(int(row["nb_ch_unique_count_init"]), 1)
        self.assertEqual(json.loads(str(row["nb_ch_init_seq_first8_json"]))[:4], [2, 2, 2, 2])
        self.assertGreater(int(row["nb_ch_unique_count_ctrl"]), 1)
        self.assertEqual(json.loads(str(row["nb_ch_ctrl_seq_first8_json"]))[:4], [2, 2, 0, 0])
        self.assertEqual(int(row["nb_ch_unique_count_report"]), 0)

    def test_spec_reference_matches_runtime_selector(self) -> None:
        allow = [0, 1, 2, 3]
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=tuple(allow),
            mms_prng_seed=0,
            channel_switching_field=1,
        )
        seq_rt = [int(selected_nb_channel_for_block(cfg, i)) for i in range(8)]
        seq_ref = [int(spec_select_nb_channel_reference(allow, 0, i)) for i in range(8)]
        self.assertEqual(seq_rt, seq_ref)
        self.assertEqual(seq_rt[:4], [2, 2, 0, 0])

    def test_seed_changes_sequence(self) -> None:
        allow = [0, 1, 2, 3]
        s0 = [int(spec_select_nb_channel_reference(allow, 0, i)) for i in range(8)]
        s1 = [int(spec_select_nb_channel_reference(allow, 1, i)) for i in range(8)]
        s2 = [int(spec_select_nb_channel_reference(allow, 2, i)) for i in range(8)]
        s255 = [int(spec_select_nb_channel_reference(allow, 255, i)) for i in range(8)]
        self.assertNotEqual(s0, s1)
        self.assertNotEqual(s0, s2)
        self.assertNotEqual(s0, s255)

    def test_selected_nb_channel_for_phase_init_always_fixed(self) -> None:
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=(0, 1, 2, 3),
            mms_prng_seed=0,
            channel_switching_field=1,
            mms_nb_init_channel=2,
        )
        seq = [int(selected_nb_channel_for_phase(cfg, "init", i)) for i in range(4)]
        self.assertEqual(seq, [2, 2, 2, 2])

    def test_control_and_report_use_same_rule_same_block(self) -> None:
        cfg_off = NbChannelSwitchConfig(
            enable_switching=False,
            allow_list=(0, 1, 2, 3),
            mms_prng_seed=0,
            channel_switching_field=0,
            mms_nb_init_channel=2,
        )
        cfg_on = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=(0, 1, 2, 3),
            mms_prng_seed=0,
            channel_switching_field=1,
            mms_nb_init_channel=2,
        )
        for k in range(8):
            self.assertEqual(
                int(selected_nb_channel_for_phase(cfg_off, "ctrl", k)),
                int(selected_nb_channel_for_phase(cfg_off, "report", k)),
            )
            self.assertEqual(
                int(selected_nb_channel_for_phase(cfg_on, "ctrl", k)),
                int(selected_nb_channel_for_phase(cfg_on, "report", k)),
            )

    def test_report_phase_sequence_non_empty_when_enabled(self) -> None:
        row = _run_case(
            enable_switching=True,
            allow_list=(0, 1, 2, 3),
            sim_seed=20260307,
            mac_seed=0,
            include_report_phase=True,
        )
        seq = json.loads(str(row["nb_ch_report_seq_first8_json"]))
        self.assertGreaterEqual(len(seq), 1)

    def test_allow_0_3_coverage_with_256_blocks(self) -> None:
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=(0, 1, 2, 3),
            mms_prng_seed=0,
            channel_switching_field=1,
            mms_nb_init_channel=2,
        )
        seq = [int(selected_nb_channel_for_phase(cfg, "ctrl", k)) for k in range(256)]
        self.assertEqual(set(seq), {0, 1, 2, 3})

    def test_all_250_coverage_and_near_uniform_with_4096_blocks(self) -> None:
        allow = tuple(range(250))
        cfg = NbChannelSwitchConfig(
            enable_switching=True,
            allow_list=allow,
            mms_prng_seed=0,
            channel_switching_field=1,
            mms_nb_init_channel=2,
        )
        seq = [int(selected_nb_channel_for_phase(cfg, "ctrl", k)) for k in range(4096)]
        # Coverage: with 4096 draws and 250 bins, all bins should almost always appear.
        uniq = len(set(seq))
        self.assertGreaterEqual(uniq, 248)
        # Rough uniformity sanity (not strict statistical test): max/min count ratio bounded.
        cnt = {ch: 0 for ch in allow}
        for v in seq:
            cnt[int(v)] += 1
        mx = max(cnt.values())
        mn = min(cnt.values())
        self.assertLessEqual(mx - mn, 30)


if __name__ == "__main__":
    unittest.main()
