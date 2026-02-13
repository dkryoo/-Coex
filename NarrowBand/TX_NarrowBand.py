# TX_NarrowBand.py
import numpy as np

def generate_mseq32():
    """Simple 32-bit m-seq from 5-bit LFSR taps x^5 + x^2 + 1."""
    reg = [1, 1, 1, 1, 1]
    seq = []
    for _ in range(32):
        seq.append(reg[-1])
        new = reg[-1] ^ reg[-3]
        reg = [new] + reg[:-1]
    return np.array(seq, dtype=int)

class OQPSK_SF32_Tx:
    """
    Frame (Config#1 style):
      SHR = SYNC(32 bits m-seq) + SFD(8 bits)
      PHR = 8 bits: length(bytes)[7 bits LSB-first] + reserved(0)
      PSDU = byte padded
    Modulation:
      4 bits (LSB-first) -> symbol (0..15)
      symbol -> 32 chips (Table 13-1 codebook)
      chips -> OQPSK baseband (I=even chips, Q=odd chips), Q delayed by Tc/2
      half-sine pulse, osr samples/chip
    """

    def __init__(self, chip_rate_hz=2e6, osr=8, chip_map=None):
        self.chip_rate_hz = float(chip_rate_hz)
        self.osr = int(osr)
        if self.osr < 2 or self.osr % 2 != 0:
            raise ValueError("osr must be even and >=2")
        self.fs = self.chip_rate_hz * self.osr
        self.SF = 32

        self.chip_map = chip_map if chip_map is not None else self._default_chip_map()
        self._validate_chip_map()

        t = np.arange(self.osr) / self.osr
        self.pulse = np.sin(np.pi * t)

        # SHR bits
        self.SYNC_BITS = generate_mseq32()
        self.SFD_BITS = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=int)
        self.SHR_BITS = np.concatenate([self.SYNC_BITS, self.SFD_BITS])  # 40 bits = 10 symbols

    def _validate_chip_map(self):
        cm = np.asarray(self.chip_map, dtype=int)
        if cm.shape != (16, 32):
            raise ValueError("chip_map must be shape (16,32)")
        if not np.all((cm == 0) | (cm == 1)):
            raise ValueError("chip_map entries must be 0/1")
        self.chip_map = cm

    def _default_chip_map(self):
        # Table 13-1 (your 16x32 codebook)
        return np.array([
            [1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0],
            [1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0],
            [0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0],
            [0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1],
            [0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1],
            [0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0],
            [1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1],
            [1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1],
            [1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1],
            [1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1],
            [0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1],
            [0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,0],
            [0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1],
            [1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0],
            [1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0]
        ], dtype=int)

    # ---------- FIR shaping (optional) ----------
    @staticmethod
    def design_lowpass_fir(fs: float, cutoff_hz: float, taps: int = 1601) -> np.ndarray:
        if taps % 2 == 0:
            raise ValueError("taps must be odd")
        fc = cutoff_hz / fs
        if not (0.0 < fc < 0.5):
            raise ValueError("cutoff must be within (0, fs/2)")

        n = np.arange(taps) - (taps - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * n)
        w = np.hanning(taps)
        h = h * w
        h = h / (np.sum(h) + 1e-30)
        return h.astype(float)

    @staticmethod
    def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(x, h, mode="same").astype(np.complex128)

    # ---------- Frame builder ----------
    def build_frame_bits(self, psdu_bits: np.ndarray) -> tuple[np.ndarray, int]:
        psdu_bits = np.asarray(psdu_bits, dtype=int).flatten()
        psdu_len_bytes = int(np.ceil(psdu_bits.size / 8))
        if not (1 <= psdu_len_bytes <= 127):
            raise ValueError("PSDU length(bytes) must be 1..127")

        pad = psdu_len_bytes * 8 - psdu_bits.size
        psdu_padded = np.concatenate([psdu_bits, np.zeros(pad, dtype=int)])

        # PHR: length[0..6] LSB-first + reserved(0)
        phr_bits = np.array([(psdu_len_bytes >> i) & 1 for i in range(7)] + [0], dtype=int)

        frame_bits = np.concatenate([self.SHR_BITS, phr_bits, psdu_padded])
        if frame_bits.size % 4 != 0:
            raise RuntimeError("Frame bits must be multiple of 4 (byte padded PSDU should ensure this).")

        return frame_bits, psdu_len_bytes

    # ---------- Modulator ----------
    def bits_to_baseband(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=int).flatten()
        if bits.size % 4 != 0:
            bits = bits[: (bits.size // 4) * 4]

        # nibble(LSB-first) -> symbol
        syms = []
        for i in range(0, bits.size, 4):
            nib = bits[i:i+4]
            syms.append(int(nib[0] | (nib[1] << 1) | (nib[2] << 2) | (nib[3] << 3)))
        syms = np.asarray(syms, dtype=int)

        chips01 = self.chip_map[syms, :].reshape(-1)
        chips_pm = chips01 * 2 - 1

        I = chips_pm[0::2].astype(float)
        Q = chips_pm[1::2].astype(float)

        I_wave = np.outer(self.pulse, I).reshape(-1, order="F")
        Q_wave = np.outer(self.pulse, Q).reshape(-1, order="F")

        half = self.osr // 2
        I_wave = np.concatenate([I_wave, np.zeros(half)])
        Q_wave = np.concatenate([np.zeros(half), Q_wave])

        L = max(I_wave.size, Q_wave.size)
        I_wave = np.pad(I_wave, (0, L - I_wave.size))
        Q_wave = np.pad(Q_wave, (0, L - Q_wave.size))

        wf = I_wave + 1j * Q_wave
        wf = wf / np.sqrt(np.mean(np.abs(wf) ** 2) + 1e-30)
        return wf.astype(np.complex128)


if __name__ == "__main__":
    # Simple TX self-test
    tx = OQPSK_SF32_Tx(chip_rate_hz=2e6, osr=8)

    psdu_bits = np.random.randint(0, 2, 122).astype(int)
    frame_bits, Lb = tx.build_frame_bits(psdu_bits)
    wf_bb = tx.bits_to_baseband(frame_bits)

    # Optional shaping to 2.5 MHz channel
    h_tx = tx.design_lowpass_fir(tx.fs, cutoff_hz=1.25e6, taps=1601)
    wf_tx = tx.apply_fir(wf_bb, h_tx)
    wf_tx = wf_tx / np.sqrt(np.mean(np.abs(wf_tx)**2) + 1e-30)

    print("[TX] fs =", tx.fs, "samples/s")
    print("[TX] PSDU bytes =", Lb)
    print("[TX] wf_bb len =", len(wf_bb), "wf_tx len =", len(wf_tx))
