from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


CHIP_RATE = 499.2e6
RSTU_S = 416.0 / CHIP_RATE
C_MPS = 299_792_458.0


@dataclass(frozen=True)
class MmsUwbConfig:
    """
    Symbolic MMS UWB packet mode configuration.
    """

    phy_uwb_mms_rsf_number_frags: int = 2
    phy_uwb_mms_rif_number_frags: int = 0
    phy_uwb_mms_rsf_length_units512: int = 2
    phy_uwb_mms_rif_length_units512: int = 1
    phy_uwb_mms_rsf_code_index: int = 15
    phy_uwb_mms_rsf_msr: int = 32
    phy_uwb_mms_rsf_gap: int = 0
    phy_uwb_mms_rsf_spreading_factor: int = 4
    phy_uwb_mms_rsf_reps: int = 2
    phy_uwb_mms_rsf_symbol_reps_hprf: int = 1
    hprf: bool = True

    def rsf_duration_rstu(self) -> int:
        chips = (
            self.phy_uwb_mms_rsf_length_units512
            * 512
            * self.phy_uwb_mms_rsf_reps
            * (self.phy_uwb_mms_rsf_symbol_reps_hprf if self.hprf else 1)
        )
        return max(1, math.ceil(chips / 416.0))

    def rif_duration_rstu(self) -> int:
        chips = self.phy_uwb_mms_rif_length_units512 * 512
        return max(1, math.ceil(chips / 416.0))


@dataclass(frozen=True)
class Fragment:
    start_rstu: int
    kind: str  # "RSF" | "RIF"
    duration_rstu: int
    rmarker_rstu: int
    payload_bits: bytes | None = None


def get_rsf_base_sequence(code_index: int) -> np.ndarray:
    """
    Placeholder sequence generator.
    Later this can be replaced by exact tables (16-8/16-9/75).
    """
    if not (0 <= code_index <= 255):
        raise ValueError("code_index must be 0..255")
    if 9 <= code_index <= 24:
        ln = 127
    elif 25 <= code_index <= 32:
        ln = 91
    elif 33 <= code_index <= 48:
        ln = 127
    else:
        ln = 63
    rng = np.random.default_rng(code_index)
    seq = rng.choice(np.array([-1, 1], dtype=np.int8), size=ln)
    return seq.astype(np.int8)


def build_mms_uwb_fragments(role: str, phase_start_rstu: int, cfg: MmsUwbConfig) -> list[Fragment]:
    """
    Build interleaved MMS UWB fragment schedule.

    - Initiator starts at phase start.
    - Responder starts 600 RSTU after phase start.
    - Each next fragment start is +1200 RSTU.
    """
    role_l = role.lower().strip()
    if role_l not in {"initiator", "responder"}:
        raise ValueError("role must be 'initiator' or 'responder'")
    base = phase_start_rstu + (0 if role_l == "initiator" else 600)
    frags: list[Fragment] = []
    stride = 1200
    rsf_dur = cfg.rsf_duration_rstu()
    rif_dur = cfg.rif_duration_rstu()

    cursor = base
    for _ in range(cfg.phy_uwb_mms_rsf_number_frags):
        frags.append(
            Fragment(
                start_rstu=cursor,
                kind="RSF",
                duration_rstu=rsf_dur,
                rmarker_rstu=cursor,
                payload_bits=None,
            )
        )
        cursor += stride

    for rif_idx in range(cfg.phy_uwb_mms_rif_number_frags):
        payload = bytes([rif_idx & 0xFF])
        frags.append(
            Fragment(
                start_rstu=cursor,
                kind="RIF",
                duration_rstu=rif_dur,
                rmarker_rstu=cursor,
                payload_bits=payload,
            )
        )
        cursor += stride

    return frags


def propagation_delay_rstu(distance_m: float) -> int:
    if distance_m < 0:
        raise ValueError("distance_m must be >= 0")
    delay_s = distance_m / C_MPS
    return int(round(delay_s / RSTU_S))

