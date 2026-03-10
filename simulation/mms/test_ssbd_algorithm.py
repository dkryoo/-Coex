from __future__ import annotations

import unittest

import numpy as np

from simulation.mms.nb_ssbd_access import SsbdConfig, run_ssbd_access


class TestSsbdAlgorithm(unittest.TestCase):
    def test_idle_immediate_success(self) -> None:
        cfg = SsbdConfig(
            mac_ssbd_tx_on_end=False,
            mac_ssbd_max_backoffs=5,
            mac_ssbd_min_bf=1,
            mac_ssbd_max_bf=5,
            mac_ssbd_unit_backoff_ms=0.001,
            mac_ssbd_persistence=False,
        )
        rng = np.random.default_rng(11)
        res = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -120.0,  # always idle wrt threshold
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=False,
        )
        self.assertTrue(res.access_granted)
        self.assertEqual(res.reason, "idle")
        self.assertEqual(res.nb_count, 0)
        self.assertEqual(res.cca_count, 1)

    def test_busy_forever_tx_on_end_false_fails(self) -> None:
        cfg = SsbdConfig(
            mac_ssbd_tx_on_end=False,
            mac_ssbd_max_backoffs=5,
            mac_ssbd_min_bf=1,
            mac_ssbd_max_bf=5,
            mac_ssbd_unit_backoff_ms=0.001,
            mac_ssbd_persistence=False,
        )
        rng = np.random.default_rng(1)
        res = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -50.0,  # always busy wrt -72 dBm
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=False,
        )
        self.assertFalse(res.access_granted)
        self.assertEqual(res.reason, "maxbackoffs_fail")

    def test_busy_forever_tx_on_end_true_succeeds(self) -> None:
        cfg = SsbdConfig(mac_ssbd_tx_on_end=True, mac_ssbd_max_backoffs=5)
        rng = np.random.default_rng(2)
        res = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -50.0,
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=False,
        )
        self.assertTrue(res.access_granted)
        self.assertEqual(res.reason, "maxbackoffs_txonend")

    def test_persistence_retransmission_bf_init(self) -> None:
        cfg = SsbdConfig(
            mac_ssbd_persistence=True,
            mac_ssbd_min_bf=1,
            mac_ssbd_max_bf=5,
            mac_ssbd_tx_on_end=True,
        )
        rng = np.random.default_rng(3)
        # previous terminating BF=4 => next init should be min(4+1,5)=5
        res = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -100.0,  # immediate idle
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=True,
            prev_terminating_bf=4,
        )
        self.assertEqual(res.bf_init, 5)
        self.assertTrue(res.access_granted)
        self.assertEqual(res.reason, "idle")

    def test_persistence_across_attempts_bf_evolution(self) -> None:
        cfg = SsbdConfig(
            mac_ssbd_persistence=True,
            mac_ssbd_min_bf=1,
            mac_ssbd_max_bf=5,
            mac_ssbd_max_backoffs=5,
            mac_ssbd_tx_on_end=True,
        )
        rng = np.random.default_rng(123)
        r1 = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -50.0,  # busy => terminate by tx-on-end path eventually
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=False,
        )
        self.assertEqual(r1.bf_init, 1)
        r2 = run_ssbd_access(
            t_start_ms=0.0,
            sense_inband_dbm_fn=lambda _t: -100.0,  # idle quickly
            cfg=cfg,
            rng=rng,
            is_retransmission_attempt=True,
            prev_terminating_bf=r1.bf_final,
        )
        self.assertEqual(r2.bf_init, min(r1.bf_final + 1, cfg.mac_ssbd_max_bf))


if __name__ == "__main__":
    unittest.main()
