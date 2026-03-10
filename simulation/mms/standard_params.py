from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardParams:
    source: str
    # NB control defaults
    nb_ifs_ms: float
    nb_lbt_cca_slots: int
    nb_lbt_slot_ms: float
    nb_retry_limit: int
    # NB SSBD defaults (802.15.4ab Clause 10.45 style, simulator profile)
    nb_phy_cca_duration_ms: float
    nb_phy_cca_ed_threshold_dbm: float
    nb_ssbd_unit_backoff_ms: float
    nb_ssbd_min_bf: int
    nb_ssbd_max_bf: int
    nb_ssbd_max_backoffs: int
    nb_ssbd_tx_on_end: bool
    nb_ssbd_persistence: bool
    # UWB ranging defaults
    uwb_shots_per_session: int
    uwb_require_k_successes: int
    uwb_reply_delay_ms: float
    # Notes for traceability
    note: str
    section_hint: str


def get_default_params(source: str = "repo_defaults") -> StandardParams:
    src = source.strip().lower()
    if src == "802154ab":
        # Placeholder mapped to current simulator defaults.
        # TODO: replace with exact values once the local 802.15.4ab PDF section table
        # is available in the repository with explicit parameter mapping.
        return StandardParams(
            source="802154ab",
            nb_ifs_ms=0.15,
            nb_lbt_cca_slots=4,
            nb_lbt_slot_ms=0.05,
            nb_retry_limit=5,
            nb_phy_cca_duration_ms=0.05,
            nb_phy_cca_ed_threshold_dbm=-72.0,
            nb_ssbd_unit_backoff_ms=0.001,
            nb_ssbd_min_bf=1,
            nb_ssbd_max_bf=5,
            nb_ssbd_max_backoffs=5,
            nb_ssbd_tx_on_end=True,
            nb_ssbd_persistence=False,
            uwb_shots_per_session=2,
            uwb_require_k_successes=1,
            uwb_reply_delay_ms=0.5,
            note="Using simulator-aligned defaults pending explicit PDF table ingestion.",
            section_hint="IEEE P802.15.4ab/D03 Clause 10.39.x, Clause 16 (timing placeholders)",
        )
    return StandardParams(
        source="repo_defaults",
        nb_ifs_ms=0.15,
        nb_lbt_cca_slots=4,
        nb_lbt_slot_ms=0.05,
        nb_retry_limit=5,
        nb_phy_cca_duration_ms=0.05,
        nb_phy_cca_ed_threshold_dbm=-72.0,
        nb_ssbd_unit_backoff_ms=0.001,
        nb_ssbd_min_bf=1,
        nb_ssbd_max_bf=5,
        nb_ssbd_max_backoffs=5,
        nb_ssbd_tx_on_end=True,
        nb_ssbd_persistence=False,
        uwb_shots_per_session=1,
        uwb_require_k_successes=1,
        uwb_reply_delay_ms=0.5,
        note="Derived from existing simulation constants.",
        section_hint="simulation/mms/mms_latency_sweep.py + UWB/mms_uwb_packet_mode.py",
    )
