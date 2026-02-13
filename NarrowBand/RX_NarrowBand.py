# RX_NarrowBand.py
import numpy as np

if __package__:
    from . import TX_NarrowBand as txnb
else:
    import TX_NarrowBand as txnb


class OQPSK_SF32_Rx:
    """
    RX for TX_NarrowBand.py (NO channel):
      - Sync by correlating with SHR waveform
      - Fine timing by SFD score over sample offsets 0..osr-1
      - Decode PHR and PSDU

    IMPORTANT:
      If your received waveform is TX-shaped (wf_tx),
      pass tx_fir=h_tx so reference SHR is filtered the same way.
    """

    def __init__(self, chip_rate_hz=2e6, osr=8):
        self.chip_rate_hz = float(chip_rate_hz)
        self.osr = int(osr)
        if self.osr < 2 or self.osr % 2 != 0:
            raise ValueError("osr must be even and >=2")
        self.fs = self.chip_rate_hz * self.osr
        self.SF = 32
        self.half = self.osr // 2

        tx0 = txnb.OQPSK_SF32_Tx(chip_rate_hz=self.chip_rate_hz, osr=self.osr)
        self.chip_map = tx0.chip_map
        self.SHR_BITS = tx0.SHR_BITS
        self.SFD_BITS = tx0.SFD_BITS

        self.SHR_CHIPS = 10 * 32
        self.PHR_CHIPS = 2 * 32

    @staticmethod
    def _matched_corr(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return np.convolve(rx, np.conj(ref[::-1]), mode="valid")

    @staticmethod
    def _apply_fir_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(x, h, mode="same").astype(np.complex128)

    def _chip_integrate_dump(self, wf: np.ndarray, offset: int = 0) -> np.ndarray:
        # generous tail pad (prevents boundary truncation)
        wf = np.concatenate([wf, np.zeros(200 * self.osr, dtype=np.complex128)])

        x = wf[offset:]
        re = np.real(x)
        im = np.imag(x)

        n_i = len(re) // self.osr
        i_vals = np.array([np.sum(re[k*self.osr:(k+1)*self.osr]) for k in range(n_i)])

        q_shift = self.half
        n_q = max(0, (len(im) - q_shift) // self.osr)
        q_vals = np.array([np.sum(im[q_shift + k*self.osr : q_shift + (k+1)*self.osr]) for k in range(n_q)])

        L = min(len(i_vals), len(q_vals))
        chips = np.empty(2 * L, dtype=int)
        chips[0::2] = (i_vals[:L] > 0).astype(int)
        chips[1::2] = (q_vals[:L] > 0).astype(int)
        return chips

    def _chips_to_bits(self, chips: np.ndarray) -> np.ndarray:
        chips = np.asarray(chips, dtype=int).flatten()
        chips = chips[: (len(chips)//32) * 32]

        bits = []
        for i in range(len(chips)//32):
            blk = chips[i*32:(i+1)*32]
            best = int(np.argmin(np.sum(blk[None, :] != self.chip_map, axis=1)))
            bits.extend([(best >> k) & 1 for k in range(4)])
        return np.asarray(bits, dtype=int)

    def _fine_timing(self, synced: np.ndarray) -> tuple[int, int]:
        best_off = 0
        best_score = 10**9
        for off in range(self.osr):
            chips = self._chip_integrate_dump(synced, off)
            if len(chips) < self.SHR_CHIPS:
                continue
            sfd_chips = chips[self.SHR_CHIPS - 64 : self.SHR_CHIPS]
            sfd_hat = self._chips_to_bits(sfd_chips)[:8]
            score = int(np.sum(sfd_hat != self.SFD_BITS))
            if score < best_score:
                best_score = score
                best_off = off
        return best_off, best_score

    def decode(self, rx_wf: np.ndarray, tx_fir: np.ndarray | None = None, verbose: bool = False):
        """
        rx_wf: wf_tx (already TX-shaped) or wf_bb (unshaped).
        tx_fir: if rx_wf is TX-shaped, pass the SAME h_tx so reference is shaped too.
        """
        rx = rx_wf.astype(np.complex128)

        # Build reference SHR waveform using TX modulator (exact match)
        tx1 = txnb.OQPSK_SF32_Tx(chip_rate_hz=self.chip_rate_hz, osr=self.osr)
        ref = tx1.bits_to_baseband(self.SHR_BITS)

        if tx_fir is not None:
            ref = self._apply_fir_same(ref, tx_fir)

        corr = self._matched_corr(rx, ref)
        corr_abs = np.abs(corr)
        candidate_count = min(64, len(corr_abs))
        sync_candidates = np.argpartition(corr_abs, -candidate_count)[-candidate_count:]
        sync_candidates = sync_candidates[np.argsort(corr_abs[sync_candidates])[::-1]]

        phr_start = self.SHR_CHIPS
        phr_end = self.SHR_CHIPS + self.PHR_CHIPS

        best = None
        for sync_idx in sync_candidates:
            synced = rx[sync_idx:]

            # phase/gain correction
            g = corr[sync_idx] / (np.vdot(ref, ref) + 1e-30)
            synced = synced / (g + 1e-30)

            best_off, sfd_score = self._fine_timing(synced)
            chips = self._chip_integrate_dump(synced, best_off)

            if len(chips) < phr_end:
                continue

            phr_bits = self._chips_to_bits(chips[phr_start:phr_end])[:8]
            psdu_len_bytes = int(sum((int(phr_bits[i]) << i) for i in range(7)))
            if not (1 <= psdu_len_bytes <= 127):
                continue

            psdu_bits_len = psdu_len_bytes * 8
            psdu_chips_need = (psdu_bits_len // 4) * 32

            psdu_start = phr_end
            psdu_end = psdu_start + psdu_chips_need
            if len(chips) < psdu_end:
                continue

            psdu_bits = self._chips_to_bits(chips[psdu_start:psdu_end])[:psdu_bits_len]
            metric = (sfd_score, -corr_abs[sync_idx])
            cand = (metric, int(sync_idx), int(best_off), int(sfd_score), int(psdu_len_bytes), psdu_bits)

            if best is None or cand[0] < best[0]:
                best = cand

                if sfd_score == 0:
                    break

        if best is None:
            raise RuntimeError("Unable to find valid SHR/PHR candidate")

        _, sync_idx, best_off, sfd_score, psdu_len_bytes, psdu_bits = best
        psdu_bits_len = psdu_len_bytes * 8

        if verbose:
            print(f"[RX] sync_idx={sync_idx}, best_off={best_off}, sfd_score={sfd_score}")
            print(f"[RX] psdu_len_bytes={psdu_len_bytes}, psdu_bits_len={psdu_bits_len}")

        return psdu_bits, psdu_len_bytes, sfd_score

if __name__ == "__main__":
    chip_rate_hz = 2e6
    osr = 8

    # TX
    tx = txnb.OQPSK_SF32_Tx(chip_rate_hz=chip_rate_hz, osr=osr)
    psdu_bits = np.random.randint(0, 2, 122).astype(int)

    frame_bits, psdu_len_bytes = tx.build_frame_bits(psdu_bits)
    wf_bb = tx.bits_to_baseband(frame_bits)

    # TX shaping (optional)
    h_tx = tx.design_lowpass_fir(tx.fs, cutoff_hz=1.25e6, taps=1601)
    wf_tx = tx.apply_fir(wf_bb, h_tx)
    wf_tx = wf_tx / np.sqrt(np.mean(np.abs(wf_tx)**2) + 1e-30)

    # RX (no channel) - first, verify pure modulation/demodulation compatibility.
    rx = OQPSK_SF32_Rx(chip_rate_hz=chip_rate_hz, osr=osr)
    rx_psdu_bits, rx_len_bytes, sfd_score = rx.decode(wf_bb, tx_fir=None, verbose=True)

    rx_trim = rx_psdu_bits[: len(psdu_bits)]
    ber = float(np.mean(rx_trim != psdu_bits))
    print(f"[TEST no shaping] BER={ber:.6e}, len_bytes={rx_len_bytes}, sfd_score={sfd_score}")

    # NOTE:
    # wf_tx includes pulse-shaping FIR. With the current simple hard-decision receiver,
    # additional matched filtering/equalization would be needed for robust decoding.
