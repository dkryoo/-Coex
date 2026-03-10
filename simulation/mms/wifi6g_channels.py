from __future__ import annotations

"""Wi-Fi 6 GHz channel helpers and full-band partition generator.

Regulatory default: FCC 5925-7125 MHz.
Channel-number convention in this repo: fc_MHz = 5950 + 5*ch.
"""

from dataclasses import dataclass
import numpy as np

FCC_BAND_LO_MHZ = 5925.0
FCC_BAND_HI_MHZ = 7125.0

# Canonical 6 GHz 20 MHz channels on 5 MHz numbering grid.
CANON_20MHZ_CH = list(range(1, 234, 4))  # [1,5,9,...,233], 59 channels
CANON_40MHZ_CH = list(range(3, 228, 8))
CANON_80MHZ_CH = list(range(7, 216, 16))
CANON_160MHZ_CH = [15, 47, 79, 111, 143, 175, 207]


@dataclass(frozen=True)
class ChannelSegment:
    bw_mhz: int
    center_ch: int
    f_center_hz: float
    f_lo_hz: float
    f_hi_hz: float
    covered_20ch_list: list[int]


def ch_to_fc_hz(ch: int) -> float:
    return float((5950.0 + 5.0 * float(ch)) * 1e6)


def make_channel_segment(start20_idx: int, n20: int) -> ChannelSegment:
    if n20 not in {1, 2, 4, 8}:
        raise ValueError("n20 must be one of {1,2,4,8}")
    if start20_idx < 0 or start20_idx + n20 > len(CANON_20MHZ_CH):
        raise ValueError("start20_idx out of range")

    start_ch = CANON_20MHZ_CH[start20_idx]
    covered = [start_ch + 4 * k for k in range(n20)]
    center_ch = int(start_ch + 2 * (n20 - 1))
    bw_mhz = int(20 * n20)
    f_center_hz = ch_to_fc_hz(center_ch)
    half = 0.5 * bw_mhz * 1e6
    return ChannelSegment(
        bw_mhz=bw_mhz,
        center_ch=center_ch,
        f_center_hz=float(f_center_hz),
        f_lo_hz=float(f_center_hz - half),
        f_hi_hz=float(f_center_hz + half),
        covered_20ch_list=covered,
    )


def _parse_mix_weights(bw_mix_weights: dict[int, float] | None) -> dict[int, float]:
    default = {20: 0.3, 40: 0.3, 80: 0.2, 160: 0.2}
    w = dict(default if bw_mix_weights is None else bw_mix_weights)
    out = {20: float(w.get(20, 0.0)), 40: float(w.get(40, 0.0)), 80: float(w.get(80, 0.0)), 160: float(w.get(160, 0.0))}
    if sum(max(0.0, v) for v in out.values()) <= 0:
        raise ValueError("bw_mix_weights must have positive mass")
    return out


def generate_fullband_partition(
    *,
    rng: np.random.Generator,
    bw_mix_weights: dict[int, float] | None = None,
    regulatory_domain: str = "FCC",
) -> list[ChannelSegment]:
    if regulatory_domain.upper() != "FCC":
        raise ValueError("only FCC is supported currently")

    weights = _parse_mix_weights(bw_mix_weights)
    n_slots = len(CANON_20MHZ_CH)
    i = 0
    segs: list[ChannelSegment] = []

    while i < n_slots:
        feasible_n20: list[int] = []
        feasible_w: list[float] = []
        for n20, bw in [(1, 20), (2, 40), (4, 80), (8, 160)]:
            if i % n20 == 0 and (i + n20) <= n_slots:
                feasible_n20.append(n20)
                feasible_w.append(max(0.0, weights[bw]))
        if not feasible_n20:
            feasible_n20 = [1]
            feasible_w = [1.0]

        w = np.asarray(feasible_w, dtype=float)
        if np.sum(w) <= 0:
            w = np.ones_like(w)
        w = w / np.sum(w)
        n20_sel = int(rng.choice(np.asarray(feasible_n20), p=w))
        segs.append(make_channel_segment(i, n20_sel))
        i += n20_sel

    # Coverage sanity: every canonical 20 MHz channel appears exactly once.
    cov = [c for s in segs for c in s.covered_20ch_list]
    if sorted(cov) != CANON_20MHZ_CH:
        raise RuntimeError("full-band partition coverage failed")
    return segs


def summarize_partition(segs: list[ChannelSegment]) -> dict:
    hist = {20: 0, 40: 0, 80: 0, 160: 0}
    covered = []
    for s in segs:
        hist[int(s.bw_mhz)] = int(hist.get(int(s.bw_mhz), 0) + 1)
        covered.extend(s.covered_20ch_list)
    cov_ratio = float(len(set(covered)) / max(len(CANON_20MHZ_CH), 1))
    return {
        "n_segments": int(len(segs)),
        "bw_hist": hist,
        "coverage_ratio": cov_ratio,
    }


def validate_partition(
    segs: list[ChannelSegment],
    *,
    edge_tol_hz: float = 1.0,
) -> dict:
    """
    Strictly validate a full-band partition.

    Checks:
    - Covered 20 MHz channels exactly match canonical FCC 20 MHz channel list.
    - Coverage ratio is exactly 1.0.
    - Frequency segments are contiguous without overlap/gap when sorted by f_lo_hz.
    """
    if not segs:
        raise AssertionError("partition is empty")

    covered = [c for s in segs for c in s.covered_20ch_list]
    covered_sorted = sorted(covered)
    canon = list(CANON_20MHZ_CH)
    missing = [c for c in canon if c not in covered_sorted]
    extra = [c for c in covered_sorted if c not in set(canon)]
    dup = sorted({c for c in covered_sorted if covered_sorted.count(c) > 1})
    if missing or extra or dup or covered_sorted != canon:
        raise AssertionError(
            "partition coverage mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, dup={dup[:8]}"
        )

    by_lo = sorted(segs, key=lambda s: float(s.f_lo_hz))
    for i in range(1, len(by_lo)):
        prev_hi = float(by_lo[i - 1].f_hi_hz)
        cur_lo = float(by_lo[i].f_lo_hz)
        delta = cur_lo - prev_hi
        if abs(delta) > float(edge_tol_hz):
            raise AssertionError(
                f"partition edge mismatch at idx={i}: prev_hi={prev_hi}, cur_lo={cur_lo}, delta_hz={delta}"
            )

    bw_hist = {20: 0, 40: 0, 80: 0, 160: 0}
    for s in by_lo:
        bw_hist[int(s.bw_mhz)] = int(bw_hist.get(int(s.bw_mhz), 0) + 1)

    return {
        "n_segments": int(len(by_lo)),
        "bw_hist": bw_hist,
        "f_lo_hz": float(by_lo[0].f_lo_hz),
        "f_hi_hz": float(by_lo[-1].f_hi_hz),
        "coverage_ratio": 1.0,
        "coverage_ok": True,
    }
