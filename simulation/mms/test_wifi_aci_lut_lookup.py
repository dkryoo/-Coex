from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from simulation.mms.wifi_spatial_model import WiFiACILUT


def _truth_coupling(offset_mhz: np.ndarray, bw_mhz: float) -> np.ndarray:
    # Smooth synthetic leakage profile for lookup/interp validation.
    sigma = 18.0 + 0.08 * float(bw_mhz)
    return np.clip(np.exp(-(np.abs(offset_mhz) / sigma) ** 1.35), 1e-15, 1.0)


class TestWiFiACILUTLookup(unittest.TestCase):
    def _build_lut_csv(self, path: Path) -> None:
        rows: list[dict] = []
        for bw in [20, 40, 80, 160]:
            offs = np.arange(-240.0, 242.0, 2.0, dtype=float)
            coup = _truth_coupling(offs, bw)
            for o, c in zip(offs.tolist(), coup.tolist()):
                rows.append(
                    {
                        "wifi_bw_mhz": int(bw),
                        "nb_bw_mhz": 2.0,
                        "offset_mhz": float(o),
                        "coupling_linear": float(c),
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_lookup_coverage_for_bw_mix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lut.csv"
            self._build_lut_csv(p)
            lut = WiFiACILUT.from_csv(p)
            for bw in [20, 40, 80, 160]:
                for off in range(-200, 201, 5):
                    v = float(lut.coupling_linear(float(bw), float(off)))
                    self.assertTrue(np.isfinite(v), msg=f"non-finite lookup bw={bw} off={off}")
                    self.assertGreater(v, 0.0, msg=f"non-positive lookup bw={bw} off={off}")

    def test_model_vs_truth_delta_small(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lut.csv"
            self._build_lut_csv(p)
            lut = WiFiACILUT.from_csv(p)
            deltas_db: list[float] = []
            for bw in [20, 40, 80, 160]:
                offs = np.linspace(-200.0, 200.0, 401, dtype=float)
                truth = _truth_coupling(offs, bw)
                for o, t in zip(offs.tolist(), truth.tolist()):
                    m = float(lut.coupling_linear(float(bw), float(o)))
                    d = 10.0 * np.log10((m + 1e-30) / (float(t) + 1e-30))
                    deltas_db.append(float(abs(d)))
            mae = float(np.mean(np.asarray(deltas_db, dtype=float)))
            self.assertLessEqual(mae, 0.1, msg=f"lookup mean abs delta too large: {mae:.3f} dB")


if __name__ == "__main__":
    unittest.main()

