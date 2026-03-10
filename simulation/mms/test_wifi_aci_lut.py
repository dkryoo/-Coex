from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simulation.mms.wifi_spatial_model import WiFiACILUT


class TestWiFiACILUT(unittest.TestCase):
    def test_interp_and_symmetry(self) -> None:
        rows = [
            {"wifi_bw_mhz": 160, "nb_bw_mhz": 2.0, "offset_mhz": -40.0, "coupling_linear": 1e-2},
            {"wifi_bw_mhz": 160, "nb_bw_mhz": 2.0, "offset_mhz": -20.0, "coupling_linear": 2e-2},
            {"wifi_bw_mhz": 160, "nb_bw_mhz": 2.0, "offset_mhz": 0.0, "coupling_linear": 3e-2},
            {"wifi_bw_mhz": 160, "nb_bw_mhz": 2.0, "offset_mhz": 20.0, "coupling_linear": 2e-2},
            {"wifi_bw_mhz": 160, "nb_bw_mhz": 2.0, "offset_mhz": 40.0, "coupling_linear": 1e-2},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lut.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            lut = WiFiACILUT.from_csv(p)
            self.assertAlmostEqual(lut.coupling_linear(160, 0), 3e-2, places=10)
            self.assertAlmostEqual(lut.coupling_linear(160, 10), 2.5e-2, places=10)
            self.assertAlmostEqual(lut.coupling_linear(160, -10), 2.5e-2, places=10)
            self.assertGreater(lut.coupling_linear(160, 200), 0.0)


if __name__ == "__main__":
    unittest.main()

