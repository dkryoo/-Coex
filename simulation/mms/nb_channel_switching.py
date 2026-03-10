from __future__ import annotations

from dataclasses import dataclass


def _aes128_ecb_encrypt_block(key16: bytes, block16: bytes) -> bytes:
    """
    AES-128 ECB encryption for a single 16-byte block.
    Priority: Crypto.Cipher.AES (pycryptodome), fallback: cryptography.
    """
    if len(key16) != 16 or len(block16) != 16:
        raise ValueError("AES-128 ECB requires 16-byte key and block")
    try:
        from Crypto.Cipher import AES  # type: ignore

        return AES.new(key16, AES.MODE_ECB).encrypt(block16)
    except Exception:
        pass
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore

        enc = Cipher(algorithms.AES(key16), modes.ECB()).encryptor()
        return enc.update(block16) + enc.finalize()
    except Exception:
        pass
    # Pure-python fallback (AES-128, single block) to keep simulator self-contained.
    return _aes128_encrypt_block_pure_python(key16, block16)


_SBOX = [
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
]
_RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]


def _xtime(a: int) -> int:
    return (((a << 1) & 0xFF) ^ 0x1B) if (a & 0x80) else ((a << 1) & 0xFF)


def _mul(a: int, b: int) -> int:
    p = 0
    aa = a & 0xFF
    bb = b & 0xFF
    for _ in range(8):
        if bb & 1:
            p ^= aa
        hi = aa & 0x80
        aa = (aa << 1) & 0xFF
        if hi:
            aa ^= 0x1B
        bb >>= 1
    return p & 0xFF


def _sub_word(w: list[int]) -> list[int]:
    return [_SBOX[b] for b in w]


def _rot_word(w: list[int]) -> list[int]:
    return w[1:] + w[:1]


def _key_expansion_128(key16: bytes) -> list[list[int]]:
    if len(key16) != 16:
        raise ValueError("AES-128 key must be 16 bytes")
    w: list[list[int]] = [list(key16[4 * i : 4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        temp = list(w[i - 1])
        if i % 4 == 0:
            temp = _sub_word(_rot_word(temp))
            temp[0] ^= _RCON[(i // 4) - 1]
        w.append([(w[i - 4][j] ^ temp[j]) & 0xFF for j in range(4)])
    round_keys = []
    for r in range(11):
        rk = []
        for c in range(4):
            rk.extend(w[4 * r + c])
        round_keys.append(rk)
    return round_keys


def _add_round_key(state: list[int], rk: list[int]) -> None:
    for i in range(16):
        state[i] ^= rk[i]


def _sub_bytes(state: list[int]) -> None:
    for i in range(16):
        state[i] = _SBOX[state[i]]


def _shift_rows(state: list[int]) -> None:
    state[1], state[5], state[9], state[13] = state[5], state[9], state[13], state[1]
    state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
    state[3], state[7], state[11], state[15] = state[15], state[3], state[7], state[11]


def _mix_columns(state: list[int]) -> None:
    for c in range(4):
        i = 4 * c
        s0, s1, s2, s3 = state[i], state[i + 1], state[i + 2], state[i + 3]
        state[i] = _mul(s0, 2) ^ _mul(s1, 3) ^ s2 ^ s3
        state[i + 1] = s0 ^ _mul(s1, 2) ^ _mul(s2, 3) ^ s3
        state[i + 2] = s0 ^ s1 ^ _mul(s2, 2) ^ _mul(s3, 3)
        state[i + 3] = _mul(s0, 3) ^ s1 ^ s2 ^ _mul(s3, 2)


def _aes128_encrypt_block_pure_python(key16: bytes, block16: bytes) -> bytes:
    rks = _key_expansion_128(key16)
    state = list(block16)
    _add_round_key(state, rks[0])
    for rnd in range(1, 10):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, rks[rnd])
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, rks[10])
    return bytes(state)


def nba_prng_value_u32(mac_mms_prng_seed: int, ranging_block_index: int) -> int:
    """
    IEEE P802.15.4ab/D03 10.39.8.4.3 style NbaPrng interface:
    - key: macMmsPrngSeed interpreted as unsigned integer, MSB zero-padded to 128-bit
    - data: RangingBlockIndex interpreted as unsigned integer, MSB zero-padded to 128-bit
    - output: bits 0..31 from AES output.

    Simulator bit-order interpretation:
    - convert ciphertext block to unsigned 128-bit big-endian integer,
    - take the least-significant 32 bits as bits 0..31 (LSB-indexed).
    This is the convention used by this simulator and regression tests.
    """
    seed = int(mac_mms_prng_seed) & 0xFF
    rbi = int(ranging_block_index) & ((1 << 128) - 1)
    key = seed.to_bytes(16, byteorder="big", signed=False)
    data = rbi.to_bytes(16, byteorder="big", signed=False)
    ct = _aes128_ecb_encrypt_block(key, data)
    v128 = int.from_bytes(ct, byteorder="big", signed=False)
    return int(v128 & 0xFFFFFFFF)


def select_nb_channel(allow_list: list[int], mac_mms_prng_seed: int, ranging_block_index: int) -> int:
    if not allow_list:
        raise ValueError("allow_list must not be empty")
    p = nba_prng_value_u32(mac_mms_prng_seed=mac_mms_prng_seed, ranging_block_index=ranging_block_index)
    return int(allow_list[p % len(allow_list)])


_NB_CHANNEL_STEP_CODE_MAP = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 16,
    5: 32,
    6: 64,
    7: 128,
}


def decode_nb_channel_step(nb_channel_step: int | None = None, nb_channel_step_code: int | None = None) -> int:
    """
    Decode NbChannelStep value.
    - If nb_channel_step is given, it is used directly (>=1 required).
    - Else, use Table-11 style code mapping (1,2,4,8,...).
    """
    if nb_channel_step is not None:
        step = int(nb_channel_step)
        if step <= 0:
            raise ValueError("nb_channel_step must be >= 1")
        return step
    if nb_channel_step_code is None:
        return 1
    code = int(nb_channel_step_code)
    if code not in _NB_CHANNEL_STEP_CODE_MAP:
        raise ValueError(f"nb_channel_step_code must be one of {sorted(_NB_CHANNEL_STEP_CODE_MAP.keys())}")
    return int(_NB_CHANNEL_STEP_CODE_MAP[code])


def build_nb_channel_allow_list(
    *,
    explicit_allow_list: list[int] | None,
    nb_channel_start: int = 0,
    nb_channel_step: int = 1,
    nb_channel_step_code: int | None = None,
    nb_channel_bitmask_hex: str | None = None,
    max_channel: int = 249,
) -> list[int]:
    """
    IEEE P802.15.4ab/D03 10.39.8.4.2 and 10.39.11.1.3.5/.6:
      mmsNbChannelAllowList = NbChannelBitmaskSet ∩ NbChannelStepSet

    Priority:
    - explicit_allow_list (if provided and non-empty) overrides map-derived list.
    """
    if explicit_allow_list is not None and len(explicit_allow_list) > 0:
        out = sorted(set(int(c) for c in explicit_allow_list if 0 <= int(c) <= max_channel))
        if not out:
            raise ValueError("explicit allow list is empty after filtering")
        return out

    start = int(nb_channel_start)
    step = decode_nb_channel_step(nb_channel_step=nb_channel_step, nb_channel_step_code=nb_channel_step_code)
    step_set = set(range(start, max_channel + 1, step))

    if nb_channel_bitmask_hex is None or nb_channel_bitmask_hex.strip() == "":
        bitmask_set = set(range(0, max_channel + 1))
    else:
        h = nb_channel_bitmask_hex.strip().lower().replace("0x", "")
        bits = int(h, 16)
        bitmask_set = {i for i in range(max_channel + 1) if ((bits >> i) & 1) == 1}

    out = sorted(step_set.intersection(bitmask_set))
    if not out:
        raise ValueError("Derived NB allow list is empty (session cannot proceed)")
    return out


@dataclass(frozen=True)
class NbChannelSwitchConfig:
    enable_switching: bool
    allow_list: tuple[int, ...]
    mms_prng_seed: int
    channel_switching_field: int = 1
    nb_channel_spacing_mhz: float = 5.0
    mms_nb_init_channel: int = 2


def spec_select_nb_channel_reference(
    allow_list: list[int],
    mac_mms_prng_seed: int,
    ranging_block_index: int,
) -> int:
    """
    Spec-reference channel selection:
    SelectedChannel = allow_list[PrngValue mod n]
    where PrngValue is bits 0..31 (LSB32) of AES128(key=seed,data=RBI).
    """
    if not allow_list:
        raise ValueError("allow_list must not be empty")
    seed = int(mac_mms_prng_seed) & 0xFF
    if seed < 0 or seed > 255:
        raise ValueError("mac_mms_prng_seed must be in [0,255]")
    rbi = int(ranging_block_index) & ((1 << 128) - 1)
    key = seed.to_bytes(16, byteorder="big", signed=False)
    data = rbi.to_bytes(16, byteorder="big", signed=False)
    ct = _aes128_ecb_encrypt_block(key, data)
    # LSB 32 bits, represented by last 4 bytes in big-endian ciphertext integer view.
    prng_value = int.from_bytes(ct[-4:], byteorder="big", signed=False)
    return int(allow_list[prng_value % len(allow_list)])


def selected_nb_channel_for_block(cfg: NbChannelSwitchConfig, ranging_block_index: int) -> int:
    """
    Select NB channel for the current ranging block.

    IEEE P802.15.4ab/D03 related behavior:
    - 10.39.8.4.3 channel switching enabled: PRNG-based allow-list selection.
    - Channel Switching field == 0: use the lowest channel in allow list
      for control/report phase behavior.
    """
    if not cfg.allow_list:
        raise ValueError("allow_list must not be empty")
    allow = list(cfg.allow_list)
    if int(cfg.channel_switching_field) == 0:
        return int(min(allow))
    if not bool(cfg.enable_switching) or len(allow) <= 1:
        return int(min(allow))
    return int(
        select_nb_channel(
            allow_list=allow,
            mac_mms_prng_seed=int(cfg.mms_prng_seed),
            ranging_block_index=int(ranging_block_index),
        )
    )


def selected_nb_channel_for_phase(
    cfg: NbChannelSwitchConfig,
    phase: str,
    ranging_block_index: int,
) -> int:
    """
    Phase-aware selection:
    - init: use mmsNbInitChannel (default 2), independent of hopping.
    - ctrl/report: use selected_nb_channel_for_block rules.
    """
    ph = str(phase).strip().lower()
    if ph == "init":
        return int(cfg.mms_nb_init_channel)
    if ph in ("ctrl", "report"):
        return int(selected_nb_channel_for_block(cfg, ranging_block_index=ranging_block_index))
    raise ValueError("phase must be one of {'init','ctrl','report'}")
