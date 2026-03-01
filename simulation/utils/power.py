from __future__ import annotations

import numpy as np


def mean_power_w(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=np.complex128)) ** 2))


def dbw_to_w(dbw: float) -> float:
    return 10.0 ** (float(dbw) / 10.0)


def w_to_dbw(watt: float) -> float:
    return 10.0 * np.log10(max(float(watt), 1e-30))


def dbw_to_dbm(dbw: float) -> float:
    return float(dbw) + 30.0


def dbm_to_dbw(dbm: float) -> float:
    return float(dbm) - 30.0


def power_dbw(x: np.ndarray) -> float:
    return w_to_dbw(mean_power_w(x) + 1e-30)


def power_dbm(x: np.ndarray) -> float:
    return dbw_to_dbm(power_dbw(x))


def fmt_dbw_dbm(p_dbw: float) -> str:
    p_dbw = float(p_dbw)
    return f"{p_dbw:.2f} dBW ({dbw_to_dbm(p_dbw):.2f} dBm)"
