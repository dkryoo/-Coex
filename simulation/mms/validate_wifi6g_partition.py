from __future__ import annotations

"""Strict validation helpers for random 6 GHz full-band Wi-Fi partitioning."""

from simulation.mms.wifi6g_channels import ChannelSegment, validate_partition as _validate_partition


def validate_partition(segs: list[ChannelSegment], *, edge_tol_hz: float = 1.0) -> dict:
    return _validate_partition(segs, edge_tol_hz=edge_tol_hz)

