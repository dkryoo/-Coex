import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Utilities: dB, thermal noise, path loss, Rician channel
# =========================================================
C0 = 299_792_458.0  # speed of light (m/s)
K_BOLTZ = 1.380649e-23  # J/K

def dbm_to_w(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)

def w_to_dbm(w: float) -> float:
    return 10.0 * np.log10(w) + 30.0

def friis_pathloss_db(fc_hz: float, d_m: float) -> float:
    """Free-space path loss PL(dB) = 20log10(4πd/λ)."""
    d_m = max(d_m, 1e-9)
    lam = C0 / fc_hz
    return 20.0 * np.log10(4.0 * np.pi * d_m / lam)

def log_distance_pathloss_db(fc_hz: float, d_m: float, n: float = 2.0, d0: float = 1.0) -> float:
    """
    Log-distance path loss:
      PL(d) = PL(d0) + 10 n log10(d/d0)
    """
    d0 = max(d0, 1e-9)
    d_m = max(d_m, d0)
    pl0 = friis_pathloss_db(fc_hz, d0)
    return pl0 + 10.0 * n * np.log10(d_m / d0)

def scale_waveform_to_psd(
    wf: np.ndarray,
    psd_dbm_per_mhz: float,
    bw_hz: float,
    ant_gain_tx_db: float = 0.0,
    cable_loss_tx_db: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """
    Scale complex baseband waveform so that conducted Tx power corresponds to integrated PSD target.

    psd_dbm_per_mhz: target/limit (EIRP PSD) in dBm/MHz
    bw_hz: bandwidth over which PSD is integrated to total power (Hz)

    EIRP_total(dBm) = PSD(dBm/MHz) + 10log10(bw_hz/1e6)
    Conducted(dBm) = EIRP_total - Gt + Lcable_tx

    We interpret mean(|wf|^2) as conducted power in watts (1-ohm convention).
    """
    if bw_hz <= 0:
        raise ValueError("bw_hz must be > 0")

    p_eirp_dbm = psd_dbm_per_mhz + 10.0 * np.log10(bw_hz / 1e6)
    p_cond_dbm = p_eirp_dbm - ant_gain_tx_db + cable_loss_tx_db
    p_cond_w = dbm_to_w(p_cond_dbm)

    p_now = float(np.mean(np.abs(wf) ** 2)) + 1e-30
    scale = np.sqrt(p_cond_w / p_now)
    return wf * scale, p_eirp_dbm, p_cond_dbm

def rician_multipath_channel(
    fs: float,
    delays_s: list[float],
    powers_db: list[float],
    K_db: float = 8.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Create a tapped-delay complex baseband channel impulse response h[n].
      - delays_s: per-tap delays (seconds)
      - powers_db: per-tap average power weights (dB, relative)
      - K_db: Rician K-factor (dB) for first tap (LOS). Other taps Rayleigh.
    """
    if len(delays_s) != len(powers_db):
        raise ValueError("delays_s and powers_db must have same length")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    rng = np.random.default_rng(seed)
    delays_s = np.asarray(delays_s, dtype=float)
    powers_lin = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)

    # Convert delays to integer sample offsets
    d_samp = np.round(delays_s * fs).astype(int)
    if np.any(d_samp < 0):
        raise ValueError("delays_s must be non-negative")

    L = int(d_samp.max()) + 1
    h = np.zeros(L, dtype=np.complex128)

    # Normalize relative tap powers to sum=1 for small-scale fading shape
    powers_lin = powers_lin / (powers_lin.sum() + 1e-30)

    K = 10.0 ** (K_db / 10.0)

    for i, (ds, p) in enumerate(zip(d_samp, powers_lin)):
        if i == 0:
            # LOS + scattered component
            phi = rng.uniform(0.0, 2.0 * np.pi)
            los = np.sqrt(K / (K + 1.0)) * np.sqrt(p) * np.exp(1j * phi)
            sca = np.sqrt(1.0 / (K + 1.0)) * np.sqrt(p / 2.0) * (
                rng.standard_normal() + 1j * rng.standard_normal()
            )
            h[ds] += los + sca
        else:
            # Rayleigh taps
            h[ds] += np.sqrt(p / 2.0) * (rng.standard_normal() + 1j * rng.standard_normal())

    return h

def apply_pathloss_and_channel(
    wf: np.ndarray,
    h: np.ndarray,
    pl_db: float,
    ant_gain_rx_db: float = 0.0,
    cable_loss_rx_db: float = 0.0,
) -> np.ndarray:
    """
    Apply multipath channel (convolution) + large-scale path loss + RX gain/loss.
    Power scaling: Pr = Pt + Gr - PL - Lrx
    Amplitude scaling: sqrt(power scaling)
    """
    power_scale_db = ant_gain_rx_db - pl_db - cable_loss_rx_db
    amp = 10.0 ** (power_scale_db / 20.0)
    y = np.convolve(wf, h, mode="full") * amp
    return y

def add_thermal_noise_white(
    wf: np.ndarray,
    fs: float,
    nf_db: float = 6.0,
    T_k: float = 290.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add complex AWGN with two-sided PSD = kT*F (W/Hz), where F=10^(NF/10).
    For discrete-time complex white noise at sample rate fs:
      E[|n|^2] = kT*F*fs
    This is the correct "thermal PSD" injection BEFORE you apply any explicit RX filtering.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    rng = np.random.default_rng(seed)
    F = 10.0 ** (nf_db / 10.0)
    sigma2 = K_BOLTZ * T_k * F * fs  # E[|n|^2]
    n = (rng.standard_normal(len(wf)) + 1j * rng.standard_normal(len(wf))) * np.sqrt(sigma2 / 2.0)
    return wf + n

# =========================================================
# NBA Config #1 O-QPSK (SF=32) modem (TX/RX) -- improved version
# =========================================================
class NBAConfig1System:
    """
    Config #1 (Table 69 style):
    - SHR: SYNC 8 symbols + SFD 2 symbols, SF=32
    - PHR: 1 octet (8 bits) = 2 symbols, uncoded, SF=32
      * bits0..6: PSDU length in BYTES, bits7: reserved(0)
      * for an octet, bit order is LSB-first in the stored bit array here
    - PSDU: uncoded, byte-aligned via zero padding, SF=32
    - Pulse shaping: half-sine (simulation choice)
    - OQPSK half-chip delay (Q delayed by Tc/2)
    """

    def __init__(self, osr=8, chip_rate_hz=2e6):
        self.osr = int(osr)
        if self.osr < 2 or self.osr % 2 != 0:
            raise ValueError("osr should be even and >= 2 for half-chip shift with osr//2")
        self.SF = 32
        self.chip_rate_hz = float(chip_rate_hz)
        self.fs = self.chip_rate_hz * self.osr

        self.chip_map = self._get_chip_map()  # 16 x 32, assumed standard Table 13-1

        # SFD bits (8 bits) as provided by user
        self.SFD_BITS = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=int)

        # SHR bits: 8 SYNC symbols => 32 bits zeros, + SFD (2 symbols => 8 bits)
        self.SHR_BITS = np.concatenate([np.zeros(32, dtype=int), self.SFD_BITS])
        assert self.SHR_BITS.size == 40  # 10 symbols * 4 bits

        # Symbol/chip counts
        self.SHR_SYMS = 10
        self.PHR_SYMS = 2
        self.SHR_CHIPS = self.SHR_SYMS * self.SF
        self.PHR_CHIPS = self.PHR_SYMS * self.SF

    def _get_chip_map(self) -> np.ndarray:
        # Assumed Table 13-1 compliant (user confirmed).
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

    # ---------- waveform generation ----------
    def _gen_wf_sf32(self, bits: np.ndarray) -> np.ndarray:
        """
        bits: LSB-first per 4-bit nibble. Each nibble -> symbol 0..15 -> 32 chips -> OQPSK pulse stream.
        """
        bits = np.array(bits, dtype=int).flatten()
        if bits.size % 4 != 0:
            raise ValueError("bits length must be multiple of 4")

        syms = []
        for i in range(0, bits.size, 4):
            nib = bits[i:i+4]  # [b0,b1,b2,b3] LSB-first
            sym = int(nib[0] | (nib[1] << 1) | (nib[2] << 2) | (nib[3] << 3))
            syms.append(sym)

        chips = self.chip_map[np.array(syms, dtype=int)].T  # (32, nSym)

        # half-sine pulse shape
        t = np.arange(self.osr) / self.osr
        pulse = np.sin(np.pi * t)

        # OQPSK: odd chips -> I, even chips -> Q, Q delayed by Tc/2 (osr//2 samples)
        o_c = (chips[0::2, :].flatten(order='F') * 2 - 1).astype(float)
        e_c = (chips[1::2, :].flatten(order='F') * 2 - 1).astype(float)

        i_part = np.concatenate([np.outer(pulse, o_c).flatten(order='F'), np.zeros(self.osr // 2)])
        q_part = np.concatenate([np.zeros(self.osr // 2), np.outer(pulse, e_c).flatten(order='F')])

        return i_part + 1j * q_part

    # ---------- RX helpers ----------
    def _matched_corr(self, rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return np.convolve(rx, np.conj(ref[::-1]), mode="valid")

    def _chip_integrate_dump_with_offset(self, synced: np.ndarray, offset: int) -> np.ndarray:
        x = synced[offset:]
        re = np.real(x)
        im = np.imag(x)
        q_shift = self.osr // 2

        n_i = len(re) // self.osr
        i_vals = np.array([np.sum(re[k*self.osr:(k+1)*self.osr]) for k in range(n_i)])

        n_q = max(0, (len(im) - q_shift) // self.osr)
        q_vals = np.array([np.sum(im[q_shift + k*self.osr : q_shift + (k+1)*self.osr]) for k in range(n_q)])

        l = min(len(i_vals), len(q_vals))
        rx_c = np.zeros(l * 2, dtype=int)
        rx_c[0::2] = (i_vals[:l] > 0).astype(int)
        rx_c[1::2] = (q_vals[:l] > 0).astype(int)
        return rx_c

    def _chips_to_bits_sf32(self, chips_32n: np.ndarray) -> np.ndarray:
        chips_32n = np.array(chips_32n, dtype=int).flatten()
        if chips_32n.size % 32 != 0:
            raise ValueError("chips length must be multiple of 32")

        out = []
        for i in range(chips_32n.size // 32):
            blk = chips_32n[i*32:(i+1)*32]
            best_sym = int(np.argmin(np.sum(blk[None, :] != self.chip_map, axis=1)))
            out.extend([(best_sym >> k) & 1 for k in range(4)])  # LSB-first
        return np.array(out, dtype=int)

    def _fine_timing_search(self, synced: np.ndarray) -> tuple[int, int]:
        """
        Search sample offset 0..osr-1 minimizing SFD bit errors.
        (No chip inversion hack; rely on complex gain correction from coarse correlation.)
        """
        best_score = 10**9
        best_off = 0

        for off in range(self.osr):
            rx_c = self._chip_integrate_dump_with_offset(synced, off)
            if rx_c.size < self.SHR_CHIPS:
                continue

            sfd_chips = rx_c[self.SHR_CHIPS - 64 : self.SHR_CHIPS]
            sfd_hat = self._chips_to_bits_sf32(sfd_chips)[:8]
            score = int(np.sum(sfd_hat != self.SFD_BITS))

            if score < best_score:
                best_score = score
                best_off = off

        return best_off, best_score

    # ---------- PPDU build / decode ----------
    def build_ppdu(self, psdu_bits: np.ndarray, verbose: bool = False):
        """
        PHR = 1 octet (8 bits) => 2 symbols.
          bits0..6 = PSDU length in bytes (LSB-first),
          bit7 = reserved 0
        PSDU is padded to full bytes.
        """
        psdu_bits = np.array(psdu_bits, dtype=int).flatten()

        psdu_len_bytes = int(np.ceil(psdu_bits.size / 8))
        if psdu_len_bytes <= 0 or psdu_len_bytes > 127:
            raise ValueError("PSDU length bytes must be 1..127 for 7-bit length field")

        pad_bits = psdu_len_bytes * 8 - psdu_bits.size
        psdu_padded = np.concatenate([psdu_bits, np.zeros(pad_bits, dtype=int)])

        # PHR bits (LSB-first): length[0..6] + reserved[7]=0
        phr_bits = np.array([(psdu_len_bytes >> i) & 1 for i in range(7)] + [0], dtype=int)

        full_bits = np.concatenate([self.SHR_BITS, phr_bits, psdu_padded])
        if full_bits.size % 4 != 0:
            raise RuntimeError("Internal error: total bits must be multiple of 4")

        if verbose:
            total_syms = full_bits.size // 4
            psdu_syms = psdu_padded.size // 4
            print(f"[TX] PSDU bits={psdu_bits.size}, PSDU bytes={psdu_len_bytes}, pad_bits={pad_bits}")
            print(f"[TX] SHR bits={self.SHR_BITS.size} (syms={self.SHR_SYMS}), PHR bits=8 (syms={self.PHR_SYMS}), PSDU padded bits={psdu_padded.size} (syms={psdu_syms})")
            print(f"[TX] total symbols={total_syms}, total chips={total_syms*self.SF}, fs={self.fs/1e6:.3f} Msps")

        wf = self._gen_wf_sf32(full_bits)
        wf = wf / np.sqrt(np.mean(np.abs(wf) ** 2) + 1e-30)
        return wf, psdu_len_bytes, pad_bits

    def rx_ppdu(self, rx_wf: np.ndarray, max_len_bytes: int = 127, verbose: bool = False):
        """
        Detect + decode:
          - coarse sync via SHR correlation
          - complex gain correction from correlation peak
          - fine timing search
          - decode PHR length
          - decode PSDU bits (byte-aligned)
        """
        ref_wf = self._gen_wf_sf32(self.SHR_BITS)
        corr = self._matched_corr(rx_wf, ref_wf)

        sync_idx = int(np.argmax(np.abs(corr)))
        synced = rx_wf[sync_idx:]

        # complex gain correction from correlation peak (phase+sign)
        g = corr[sync_idx] / (np.vdot(ref_wf, ref_wf) + 1e-30)
        synced = synced / (g + 1e-30)

        best_off, sfd_score = self._fine_timing_search(synced)
        rx_c = self._chip_integrate_dump_with_offset(synced, best_off)

        phr_start = self.SHR_CHIPS
        phr_end = self.SHR_CHIPS + self.PHR_CHIPS
        if rx_c.size < phr_end:
            raise RuntimeError("No PHR (frame too short).")

        phr_bits = self._chips_to_bits_sf32(rx_c[phr_start:phr_end])[:8]
        det_len_bytes = int(sum(int(phr_bits[i]) << i for i in range(7)))  # bits0..6 LSB-first

        if det_len_bytes <= 0 or det_len_bytes > max_len_bytes:
            raise RuntimeError(f"Bad PHR length: {det_len_bytes}")

        payload_bits_len = det_len_bytes * 8
        payload_chips_need = (payload_bits_len // 4) * 32  # SF=32
        payload_start = phr_end
        payload_end = payload_start + payload_chips_need

        if rx_c.size < payload_end:
            raise RuntimeError("Payload truncated.")

        rx_payload_bits = self._chips_to_bits_sf32(rx_c[payload_start:payload_end])[:payload_bits_len]

        if verbose:
            print(f"[RX] sync_idx={sync_idx}, best_off={best_off}, sfd_score={sfd_score}, det_len_bytes={det_len_bytes}")

        dbg = dict(best_off=best_off, sfd_score=sfd_score, gain=g)
        return rx_payload_bits, det_len_bytes, sync_idx, dbg

# =========================================================
# End-to-end simulation: PSD scaling + pathloss+Rician + thermal noise
# =========================================================
def simulate(
    osr=8,
    chip_rate_hz=2e6,
    fc_hz=6.5e9,
    psd_dbm_per_mhz=-41.3,
    bw_hz=2e6,
    d_m=5.0,
    n_exp=2.0,
    K_db=8.0,
    delays_s=(0.0, 50e-9, 120e-9),
    powers_db=(0.0, -6.0, -10.0),
    nf_db=6.0,
    snr_axis_db=None,  # optional legacy axis; not used in thermal-noise mode
    iters=500,
    bit_len=122,
    delay_samples=50,
    seed_base=1,
):
    """
    This runs with ABSOLUTE thermal noise (kTB + NF) injected as white noise at fs.
    NOTE: Without an explicit RX filter, "noise_bw_hz" is not enforced; we inject correct PSD at fs.

    For comparisons across distances/PSD/NF/channel params, vary those instead of sweeping SNR.
    """

    nba = NBAConfig1System(osr=osr, chip_rate_hz=chip_rate_hz)

    # Precompute large-scale path loss
    pl_db = log_distance_pathloss_db(fc_hz, d_m, n=n_exp, d0=1.0)

    ber_results = []
    ok_results = []
    sfd_avg_results = []
    detlen_avg_results = []

    # If you still want a sweep axis, use distance or NF or PSD externally.
    if snr_axis_db is None:
        snr_axis_db = [0]  # dummy single point

    print("=== NBA Config#1 + PSD scaling + Pathloss/Rician + Thermal Noise ===")
    print(f"fs={nba.fs/1e6:.3f} Msps, chip_rate={chip_rate_hz/1e6:.3f} Mcps, osr={osr}")
    print(f"fc={fc_hz/1e9:.3f} GHz, d={d_m} m, n={n_exp}, PL={pl_db:.2f} dB")
    print(f"PSD={psd_dbm_per_mhz} dBm/MHz over BW={bw_hz/1e6:.3f} MHz => EIRP_total={psd_dbm_per_mhz + 10*np.log10(bw_hz/1e6):.2f} dBm")
    print(f"K={K_db} dB, taps delays(ns)={[1e9*x for x in delays_s]}, taps powers(dB)={list(powers_db)}")
    print(f"NF={nf_db} dB, bit_len={bit_len}, iters={iters}")
    print("------------------------------------------------------------")

    for axis_val in snr_axis_db:
        errs = 0
        ok = 0
        sfd_sum = 0
        len_sum = 0

        # (Optional) could use axis_val to vary something (e.g., nf_db), here we keep it fixed.
        for i in range(iters):
            rng = np.random.default_rng(seed_base + i)
            tx_bits = rng.integers(0, 2, bit_len, dtype=int)

            # TX build
            wf, psdu_len_bytes, pad_bits = nba.build_ppdu(tx_bits)

            # (1) Scale to PSD target
            wf_scaled, p_eirp_dbm, p_cond_dbm = scale_waveform_to_psd(
                wf, psd_dbm_per_mhz=psd_dbm_per_mhz, bw_hz=bw_hz
            )

            # (2) Channel: distance loss + Rician multipath
            h = rician_multipath_channel(
                fs=nba.fs,
                delays_s=list(delays_s),
                powers_db=list(powers_db),
                K_db=K_db,
                seed=seed_base + 10_000 + i
            )
            rx = apply_pathloss_and_channel(wf_scaled, h, pl_db=pl_db)

            # Add leading zeros (packet arrival delay)
            rx = np.concatenate([np.zeros(delay_samples, dtype=complex), rx])

            # (3) Thermal noise
            rx = add_thermal_noise_white(rx, fs=nba.fs, nf_db=nf_db, T_k=290.0, seed=seed_base + 20_000 + i)

            # Decode
            try:
                rx_bits_padded, det_len_bytes, sync_idx, dbg = nba.rx_ppdu(rx, max_len_bytes=127)

                # Compare only original bit_len (ignore TX padding bits)
                errs += int(np.sum(tx_bits != rx_bits_padded[:bit_len]))
                ok += 1
                sfd_sum += dbg["sfd_score"]
                len_sum += det_len_bytes
            except Exception:
                errs += bit_len

        ber = errs / (iters * bit_len)
        ber_results.append(ber)
        ok_results.append(100.0 * ok / iters)
        sfd_avg_results.append(sfd_sum / ok if ok > 0 else np.nan)
        detlen_avg_results.append(len_sum / ok if ok > 0 else np.nan)

        print(f"Axis={axis_val} | BER={ber:.6e} | OK%={ok_results[-1]:.1f} | SFDscore(avg)={sfd_avg_results[-1]:.2f} | DetLenB(avg)={detlen_avg_results[-1]:.2f}")

    return np.array(snr_axis_db), np.array(ber_results), np.array(ok_results), np.array(sfd_avg_results), np.array(detlen_avg_results)

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # NOTE:
    # - "SNR sweep" is not meaningful once you use absolute kTB noise unless you vary PSD/NF/BW/distance.
    # - So we run a single point by default. You can create a sweep by varying d_m or nf_db externally.

    # --- 거리 sweep: 1~50m, 5m 간격 ---
    distances = np.arange(1, 51, 5)  # 1,6,11,...,46
    ber_list = []
    ok_list = []

    for d in distances:
        axis, ber, ok_pct, sfd_avg, detlen_avg = simulate(
            osr=8,
            chip_rate_hz=2e6,
            fc_hz=6.5e9,
            psd_dbm_per_mhz=-41.3,
            bw_hz=2e6,
            d_m=float(d),
            n_exp=2.0,
            K_db=8.0,
            delays_s=(0.0, 50e-9, 120e-9),
            powers_db=(0.0, -6.0, -10.0),
            nf_db=6.0,
            snr_axis_db=None,      # single point
            iters=200,             # 거리 sweep이면 500은 너무 오래 걸릴 수 있어. 필요시 500으로 올리면 됨.
            bit_len=122,
            delay_samples=50,
            seed_base=123,
        )
        ber_list.append(float(ber[0]))
        ok_list.append(float(ok_pct[0]))

    # --- Plot: BER vs distance ---
    plt.figure(figsize=(8,5))
    plt.semilogy(distances, ber_list, "o-", linewidth=2, markersize=6, label="BER")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Distance (m)")
    plt.ylabel("BER")
    plt.title("NBA Config#1: BER vs Distance (PSD+Rician+Thermal)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot: OK% vs distance ---
    plt.figure(figsize=(8,5))
    plt.plot(distances, ok_list, "o-", linewidth=2, markersize=6, label="Decode success (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Distance (m)")
    plt.ylabel("OK (%)")
    plt.title("NBA Config#1: Decode Success vs Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()
