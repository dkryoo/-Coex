from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SsbdConfig:
    # IEEE P802.15.4ab/D03 Sep 2025:
    # - Figure 207 (SSBD algorithm flow)
    # - Table 67 (SSBD PIB defaults)
    phy_cca_duration_ms: float = 0.05
    phy_cca_ed_threshold_dbm: float = -72.0
    cca_mode: int = 1
    mac_ssbd_unit_backoff_ms: float = 0.001  # 1 us
    mac_ssbd_min_bf: int = 1
    mac_ssbd_max_bf: int = 5
    mac_ssbd_max_backoffs: int = 5
    mac_ssbd_tx_on_end: bool = True
    mac_ssbd_persistence: bool = False


@dataclass(frozen=True)
class SsbdResult:
    access_granted: bool
    total_deferral_ms: float
    total_cca_ms: float
    cca_count: int
    nb_count: int
    bf_init: int
    bf_final: int
    bf_history: tuple[int, ...]
    reason: str
    last_sensed_dbm: float
    trace: tuple[dict, ...]


def run_ssbd_access(
    *,
    t_start_ms: float,
    sense_inband_dbm_fn: Callable[[float], float],
    cfg: SsbdConfig,
    rng: np.random.Generator,
    is_retransmission_attempt: bool = False,
    prev_terminating_bf: int | None = None,
    debug: bool = False,
) -> SsbdResult:
    """
    SSBD access loop aligned with Figure 207 semantics (10.45.x):
    delay(random BF) -> CCA -> idle? success : busy -> NB++, BF++ -> NB limit handling.
    """
    # BF initialization policy
    if bool(cfg.mac_ssbd_persistence) and bool(is_retransmission_attempt):
        prev_bf = int(prev_terminating_bf if prev_terminating_bf is not None else cfg.mac_ssbd_min_bf)
        bf = min(int(cfg.mac_ssbd_max_bf), max(int(cfg.mac_ssbd_min_bf), prev_bf + 1))
    else:
        bf = int(cfg.mac_ssbd_min_bf)
    bf_init = int(bf)

    t_ms = float(t_start_ms)
    nb = 0
    cca_count = 0
    bf_hist: list[int] = [int(bf)]
    trace: list[dict] = []
    total_def = 0.0
    total_cca = 0.0
    last_sensed = float("-inf")

    while True:
        # 1) Randomized deferral before CCA
        slots = int(rng.integers(0, int(bf) + 1))
        d_ms = float(slots * float(cfg.mac_ssbd_unit_backoff_ms))
        t_def_start = t_ms
        t_ms += d_ms
        total_def += d_ms

        # 2) Perform CCA
        t_cca_start = t_ms
        t_ms += float(cfg.phy_cca_duration_ms)
        total_cca += float(cfg.phy_cca_duration_ms)
        cca_count += 1
        last_sensed = float(sense_inband_dbm_fn(t_ms))
        busy = bool(last_sensed >= float(cfg.phy_cca_ed_threshold_dbm))

        step = {
            "t_def_start_ms": float(t_def_start),
            "t_def_end_ms": float(t_cca_start),
            "deferral_ms": float(d_ms),
            "t_cca_start_ms": float(t_cca_start),
            "t_cca_end_ms": float(t_ms),
            "sensed_inband_dbm": float(last_sensed),
            "threshold_dbm": float(cfg.phy_cca_ed_threshold_dbm),
            "cca_mode": int(cfg.cca_mode),
            "decision": "busy" if busy else "idle",
            "nb": int(nb),
            "bf": int(bf),
            "slots": int(slots),
        }
        trace.append(step)

        if debug:
            print(
                "[nb_ssbd] "
                f"sensed={last_sensed:.2f} dBm thr={cfg.phy_cca_ed_threshold_dbm:.2f} dBm "
                f"cca_mode={cfg.cca_mode} cca_us={cfg.phy_cca_duration_ms*1000.0:.1f} "
                f"decision={step['decision']} def_ms={d_ms:.3f} NB={nb} BF={bf}"
            )

        if not busy:
            return SsbdResult(
                access_granted=True,
                total_deferral_ms=float(total_def),
                total_cca_ms=float(total_cca),
                cca_count=int(cca_count),
                nb_count=int(nb),
                bf_init=int(bf_init),
                bf_final=int(bf),
                bf_history=tuple(bf_hist),
                reason="idle",
                last_sensed_dbm=float(last_sensed),
                trace=tuple(trace),
            )

        # 3) busy branch
        nb += 1
        bf = min(int(cfg.mac_ssbd_max_bf), int(bf) + 1)
        bf_hist.append(int(bf))
        if nb > int(cfg.mac_ssbd_max_backoffs):
            # IEEE P802.15.4ab/D03 10.45 + Figure 207:
            # - macSsbdTxOnEnd=TRUE  -> terminate SSBD with success (TX proceeds)
            # - macSsbdTxOnEnd=FALSE -> terminate SSBD with failure
            granted = bool(cfg.mac_ssbd_tx_on_end)
            return SsbdResult(
                access_granted=granted,
                total_deferral_ms=float(total_def),
                total_cca_ms=float(total_cca),
                cca_count=int(cca_count),
                nb_count=int(nb),
                bf_init=int(bf_init),
                bf_final=int(bf),
                bf_history=tuple(bf_hist),
                reason="maxbackoffs_txonend" if granted else "maxbackoffs_fail",
                last_sensed_dbm=float(last_sensed),
                trace=tuple(trace),
            )
