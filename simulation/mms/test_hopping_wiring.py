from __future__ import annotations

from simulation.mms.nb_channel_switching import NbChannelSwitchConfig, selected_nb_channel_for_block


def _simulate_init_only(
    *,
    enable_switching: bool,
    allow_list: tuple[int, ...],
    seed: int,
    max_attempts: int,
    always_busy_channels: set[int],
) -> tuple[bool, list[int]]:
    cfg = NbChannelSwitchConfig(
        enable_switching=bool(enable_switching),
        allow_list=allow_list,
        mms_prng_seed=int(seed) & 0xFF,
        channel_switching_field=1 if enable_switching else 0,
        nb_channel_spacing_mhz=2.0,
    )
    seq: list[int] = []
    for att in range(max_attempts):
        ch = int(selected_nb_channel_for_block(cfg, att))
        seq.append(ch)
        if ch in always_busy_channels:
            continue
        return True, seq
    return False, seq


def test_hopping_wiring_deterministic_busy_idle_channels() -> None:
    """
    Deterministic wiring test:
    - channel 1 always busy for SSBD CCA
    - channel 2 always idle
    - hopping OFF (fixed channel 1) must fail
    - hopping ON with same seed must eventually hit channel 2 and succeed
    """
    busy = {1}
    max_attempts = 8

    ok_off, seq_off = _simulate_init_only(
        enable_switching=False,
        allow_list=(1,),
        seed=0,
        max_attempts=max_attempts,
        always_busy_channels=busy,
    )
    assert not ok_off
    assert seq_off == [1] * max_attempts

    ok_on, seq_on = _simulate_init_only(
        enable_switching=True,
        allow_list=(1, 2),
        seed=0,
        max_attempts=max_attempts,
        always_busy_channels=busy,
    )
    assert ok_on
    assert 2 in seq_on
    assert seq_on[:7] == [1, 1, 1, 1, 1, 1, 2]
