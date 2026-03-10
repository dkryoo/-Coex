from __future__ import annotations

try:
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig, selected_nb_channel_for_block
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.nb_channel_switching import NbChannelSwitchConfig, selected_nb_channel_for_block


def generate_seq(cfg: NbChannelSwitchConfig, n: int = 16) -> list[int]:
    return [int(selected_nb_channel_for_block(cfg, i)) for i in range(n)]


def test_nb_hopping_off_constant_lowest() -> None:
    allow = (80, 120, 160, 200)
    cfg = NbChannelSwitchConfig(
        enable_switching=False,
        allow_list=allow,
        mms_prng_seed=0,
        channel_switching_field=0,
        nb_channel_spacing_mhz=2.0,
    )
    seq = generate_seq(cfg, 16)
    assert seq == [min(allow)] * 16


def test_nb_hopping_on_reproducible_nonconstant() -> None:
    allow = (80, 120, 160, 200)
    cfg = NbChannelSwitchConfig(
        enable_switching=True,
        allow_list=allow,
        mms_prng_seed=0,
        channel_switching_field=1,
        nb_channel_spacing_mhz=2.0,
    )
    s1 = generate_seq(cfg, 16)
    s2 = generate_seq(cfg, 16)
    assert s1 == s2
    assert len(set(s1)) > 1


if __name__ == "__main__":
    allow = (80, 120, 160, 200)
    cfg_off = NbChannelSwitchConfig(False, allow, 0, channel_switching_field=0, nb_channel_spacing_mhz=2.0)
    cfg_on = NbChannelSwitchConfig(True, allow, 0, channel_switching_field=1, nb_channel_spacing_mhz=2.0)
    print("OFF first16:", generate_seq(cfg_off, 16))
    print("ON  first16:", generate_seq(cfg_on, 16))
