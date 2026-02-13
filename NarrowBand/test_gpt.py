import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate  # (안 써도 되지만 유지)

class NBASystemAutonomous:
    def __init__(self, osr=8):
        self.osr = osr
        self.chip_map = self._get_chip_map()
        self.G = [0o133, 0o171]  # K=7 FEC
        self.num_states = 64
        self._precompute_trellis()

        # SHR/SFD (네가 쓰던 그대로)
        self.ref_sync_bits = np.concatenate([np.zeros(32, dtype=int),
                                             np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=int)])
        # SHR = 10 symbols = 40 bits = 320 chips (sf=32)
        self.SHR_CHIPS = 10 * 32
        self.PHR_CHIPS = 2 * 32  # 네 코드: PHR 2 symbols

    def _get_chip_map(self):
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

    def _precompute_trellis(self):
        self.next_states = np.zeros((self.num_states, 2), dtype=int)
        self.outputs = np.zeros((self.num_states, 2, 2), dtype=int)
        for s in range(self.num_states):
            for b in [0, 1]:
                ns = (s >> 1) | (b << 5)
                reg = (b << 6) | s
                self.next_states[s, b] = ns
                self.outputs[s, b] = [
                    bin(reg & self.G[0]).count('1') % 2,
                    bin(reg & self.G[1]).count('1') % 2
                ]

    # ---------- FEC ----------
    def encode(self, bits):
        state, encoded = 0, []
        for b in np.concatenate([bits, np.zeros(6, dtype=int)]):
            reg = (b << 6) | state
            encoded.extend([
                bin(reg & self.G[0]).count('1') % 2,
                bin(reg & self.G[1]).count('1') % 2
            ])
            state = (state >> 1) | (b << 5)
        return np.array(encoded, dtype=int)

    def viterbi(self, rx_fec):
        n_steps = len(rx_fec) // 2
        metrics = np.full(self.num_states, np.inf)
        metrics[0] = 0
        paths = np.zeros((self.num_states, n_steps), dtype=int)

        for t in range(n_steps):
            new_m = np.full(self.num_states, np.inf)
            new_p = np.zeros((self.num_states, n_steps), dtype=int)
            r = rx_fec[2*t:2*t+2]

            for s in range(self.num_states):
                if metrics[s] == np.inf:
                    continue
                for b in [0, 1]:
                    ns = self.next_states[s, b]
                    cost = np.sum(r != self.outputs[s, b])
                    cand = metrics[s] + cost
                    if cand < new_m[ns]:
                        new_m[ns] = cand
                        new_p[ns, :t] = paths[s, :t]
                        new_p[ns, t] = b

            metrics, paths = new_m, new_p

        best = np.argmin(metrics)
        return paths[best, :-6]  # tail 제거

    # ---------- Waveform ----------
    def _gen_wf(self, bits):
        bits = np.array(bits, dtype=int).flatten()
        assert len(bits) % 4 == 0, "bits length must be multiple of 4"

        syms = [int(''.join(map(str, bits[i:i+4][::-1])), 2) for i in range(0, len(bits), 4)]
        chips = self.chip_map[np.array(syms, dtype=int)].T  # (32, nSym)

        t = np.arange(self.osr) / self.osr
        pulse = np.sin(np.pi * t)

        o_c = (chips[0::2, :].flatten(order='F') * 2 - 1)
        e_c = (chips[1::2, :].flatten(order='F') * 2 - 1)

        # 너 기존 O-QPSK half-chip delay 구조 그대로
        i_part = np.concatenate([np.outer(pulse, o_c).flatten(order='F'), np.zeros(self.osr // 2)])
        q_part = np.concatenate([np.zeros(self.osr // 2), np.outer(pulse, e_c).flatten(order='F')])
        return i_part + 1j * q_part

    def generate_ppdu(self, info_bits):
        info_bits = np.array(info_bits, dtype=int).flatten()
        fec = self.encode(info_bits)

        # SHR(10 symbols): SYNC(8 symbols = 32 bits zeros) + SFD(2 symbols = 8 bits)
        shr = self.ref_sync_bits.copy()

        # PHR: 너 방식 유지 (len(fec)//8 를 7bit LSB-first로 + reserved 0)
        phr_val = len(fec) // 8
        phr = np.array([int(b) for b in bin(phr_val)[2:].zfill(7)[::-1]] + [0], dtype=int)

        wf = self._gen_wf(np.concatenate([shr, phr, fec]))
        wf = wf / np.sqrt(np.mean(np.abs(wf) ** 2))
        return wf, len(fec)

    # ---------- Improved RX ----------
    def _matched_corr(self, rx, ref):
        # corr[k] = sum rx[k+n] * conj(ref[n])
        return np.convolve(rx, np.conj(ref[::-1]), mode='valid')

    def _chip_integrate_dump(self, synced):
        """
        Robust hard chip decision:
        - integrate over each chip interval of length osr
        - Q branch is shifted by osr/2 (because you build waveform with q leading zeros osr/2)
        """
        re = np.real(synced)
        im = np.imag(synced)

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

    def autonomous_sync_demod(self, rx_wf, max_len_bytes=64):
        # 1) Sync via matched filter (complex conj)
        ref_wf = self._gen_wf(self.ref_sync_bits)
        corr = self._matched_corr(rx_wf, ref_wf)
        sync_idx = int(np.argmax(np.abs(corr)))
        synced = rx_wf[sync_idx:]

        # 2) Phase correction from correlation peak
        phi = np.angle(corr[sync_idx])
        synced = synced * np.exp(-1j * phi)

        # 3) Chip hard decision (integrate & dump)
        rx_c = self._chip_integrate_dump(synced)

        # --- PHR decode (PHR starts at 320 chips) ---
        phr_start = self.SHR_CHIPS
        phr_end = self.SHR_CHIPS + self.PHR_CHIPS
        if len(rx_c) < phr_end:
            raise RuntimeError("Frame too short: no PHR.")

        phr_chips = rx_c[phr_start:phr_end]
        phr_bits = []
        for i in range(2):
            blk = phr_chips[i*32:(i+1)*32]
            best_sym = np.argmin(np.sum(blk != self.chip_map, axis=1))
            phr_bits.extend([int(b) for b in bin(best_sym)[2:].zfill(4)[::-1]])

        det_len_bytes = int(''.join(map(str, phr_bits[:7][::-1])), 2)

        # sanity check (PHR 깨지면 payload slicing이 랜덤이 됨)
        if det_len_bytes <= 0 or det_len_bytes > max_len_bytes:
            raise RuntimeError(f"Bad PHR length: {det_len_bytes} bytes")

        det_fec_bits = det_len_bytes * 8

        # --- Payload demod ---
        payload_start = phr_end
        payload_need_chips = (det_fec_bits // 4) * 32
        payload_end = payload_start + payload_need_chips
        if len(rx_c) < payload_end:
            raise RuntimeError("Frame too short: payload truncated.")

        payload_chips = rx_c[payload_start:payload_end]
        fec_bits = []
        for i in range(len(payload_chips) // 32):
            blk = payload_chips[i*32:(i+1)*32]
            best = np.argmin(np.sum(blk != self.chip_map, axis=1))
            fec_bits.extend([int(b) for b in bin(best)[2:].zfill(4)[::-1]])

        if len(fec_bits) == 0:
            raise RuntimeError("Empty payload demod.")

        rx_bits = self.viterbi(np.array(fec_bits, dtype=int))
        return rx_bits, det_fec_bits, sync_idx


if __name__ == "__main__":
    nba = NBASystemAutonomous(osr=8)

    snr_range = np.arange(9, -16, -2)
    iters, bit_len = 1000, 122
    ber_results = []

    print(f"NBA System Simulation (Improved RX): {bit_len} bits per packet")
    print(f"{'SNR(dB)':>7} | {'BER':>12} | {'Sync_Idx':>8} | {'Det_Len(bits)':>12}")
    print("-" * 52)

    for snr in snr_range:
        errs = 0
        det_sum = 0
        ok_cnt = 0

        for _ in range(iters):
            tx_bits = np.random.randint(0, 2, bit_len)
            delay = 50

            wf, act_fec_len = nba.generate_ppdu(tx_bits)
            rx_wf = np.concatenate([np.zeros(delay), wf])

            # NOTE: 여기 SNR은 "샘플당" 기준이라 절대값 해석은 애매하지만,
            # 개선 RX로 곡선 형태는 훨씬 정상화될 것.
            noise = (np.random.randn(len(rx_wf)) + 1j*np.random.randn(len(rx_wf))) * \
                    np.sqrt((1/(10**(snr/10)))/2)

            try:
                rx_bits, det_l, s_idx = nba.autonomous_sync_demod(rx_wf + noise, max_len_bytes=64)
                errs += np.sum(tx_bits != rx_bits[:bit_len])
                det_sum += det_l
                ok_cnt += 1
            except:
                # 프레임 실패는 bit_len 전체 에러로 처리(기존 방식 유지)
                errs += bit_len

        ber = errs / (iters * bit_len)
        ber_results.append(ber)
        det_avg = (det_sum / ok_cnt) if ok_cnt > 0 else 0
        print(f"{snr:7.1f} | {ber:12.6e} | {delay:8.1f} | {det_avg:12.1f}")

    # plot
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_range, ber_results, 'o-', linewidth=2, markersize=6, label='Improved RX (I&D + phase corr)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('NBA System Performance (Improved RX): SNR vs BER')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nba_ber_plot_improved.png')
    print("\n[Result] Plot saved as 'nba_ber_plot_improved.png'")
    plt.show()
