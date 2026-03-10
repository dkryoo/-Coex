from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_wifi_tx_class():
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("tx_wifi_module_for_tests", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.WiFiOFDMTx


class TestWiFiOFDMASubcarriers(unittest.TestCase):
    def setUp(self) -> None:
        WiFiOFDMTx = _load_wifi_tx_class()
        self.tx = WiFiOFDMTx(rng_seed=7, center_freq_hz=6.5e9)

    def test_dc_edge_guard_and_nominal_count_for_all_bw(self) -> None:
        tol = 1e-12
        for bw in (20, 40, 80, 160):
            dbg = self.tx.build_debug_ofdm_grid(channel_bw_mhz=bw, standard="wifi6e", force_mcs=0)
            X = np.asarray(dbg["grid"], dtype=np.complex128)
            nfft = int(dbg["fft_n"])
            self.assertEqual(X.size, nfft)

            dc = int(dbg["dc_index"])
            self.assertAlmostEqual(float(np.abs(X[dc])), 0.0, places=12, msg=f"DC tone is not null (bw={bw})")

            occ_bins = np.flatnonzero(np.abs(X) > tol).astype(int)
            n_nom = int(dbg["n_data_subcarriers_nominal"])
            self.assertEqual(int(occ_bins.size), n_nom, msg=f"occupied-tone count mismatch (bw={bw})")

            Xs = np.fft.fftshift(X)
            occ = np.flatnonzero(np.abs(Xs) > tol).astype(int)
            self.assertGreater(len(occ), 0, f"No occupied tones found (bw={bw})")
            min_occ, max_occ = int(np.min(occ)), int(np.max(occ))

            left = Xs[:min_occ]
            right = Xs[max_occ + 1 :]
            self.assertGreater(left.size, 0, f"No left edge bins for guard check (bw={bw})")
            self.assertGreater(right.size, 0, f"No right edge bins for guard check (bw={bw})")

            left_zero_ratio = float(np.mean(np.abs(left) <= tol))
            right_zero_ratio = float(np.mean(np.abs(right) <= tol))
            self.assertGreaterEqual(left_zero_ratio, 0.999, msg=f"left guard not null enough (bw={bw})")
            self.assertGreaterEqual(right_zero_ratio, 0.999, msg=f"right guard not null enough (bw={bw})")

            # Metadata consistency checks.
            occ_signed = set(int(v) for v in np.asarray(dbg["occupied_signed"], dtype=int).tolist())
            null_signed = set(int(v) for v in np.asarray(dbg["null_signed"], dtype=int).tolist())
            guard_signed = set(int(v) for v in np.asarray(dbg["guard_signed"], dtype=int).tolist())
            self.assertNotIn(0, occ_signed, msg=f"DC appears in occupied_signed (bw={bw})")
            self.assertIn(0, null_signed, msg=f"DC missing in null_signed (bw={bw})")
            self.assertTrue(occ_signed.isdisjoint(null_signed), msg=f"occupied/null overlap (bw={bw})")
            self.assertTrue(guard_signed.issubset(null_signed), msg=f"guard should be subset of null (bw={bw})")

    def test_partial_allocation_keeps_unassigned_null(self) -> None:
        # Two separated RU-like blocks with unique symbols.
        grp_a = list(range(-50, -24))
        grp_b = list(range(25, 51))
        sym_map = {k: (1.0 + 1.0j) for k in grp_a}
        sym_map.update({k: (2.0 + 2.0j) for k in grp_b})

        dbg = self.tx.build_debug_ofdm_grid(
            channel_bw_mhz=20,
            standard="wifi6e",
            force_mcs=0,
            symbol_map_by_signed=sym_map,
        )
        X = np.asarray(dbg["grid"], dtype=np.complex128)
        nfft = int(dbg["fft_n"])
        tol = 1e-12

        for k in grp_a:
            self.assertAlmostEqual(float(np.abs(X[k % nfft] - (1.0 + 1.0j))), 0.0, places=10)
        for k in grp_b:
            self.assertAlmostEqual(float(np.abs(X[k % nfft] - (2.0 + 2.0j))), 0.0, places=10)

        expected_bins = {int(k % nfft) for k in (grp_a + grp_b)}
        nz_bins = {int(i) for i in np.flatnonzero(np.abs(X) > tol).tolist()}
        self.assertSetEqual(nz_bins, expected_bins, "Unexpected non-zero bins outside assigned RU-like blocks")

        self.assertAlmostEqual(float(np.abs(X[0])), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
