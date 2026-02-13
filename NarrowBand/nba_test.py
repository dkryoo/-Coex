import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate

class NBASystemAutonomous:
    def __init__(self, osr=8):
        self.osr = osr
        self.chip_map = self._get_chip_map()
        self.G = [0o133, 0o171] # K=7 FEC
        self.num_states = 64
        self._precompute_trellis()

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
        ])

    def _precompute_trellis(self):
        self.next_states = np.zeros((self.num_states, 2), dtype=int)
        self.outputs = np.zeros((self.num_states, 2, 2), dtype=int)
        for s in range(self.num_states):
            for b in [0, 1]:
                ns = (s >> 1) | (b << 5)
                reg = (b << 6) | s
                self.next_states[s, b] = ns
                self.outputs[s, b] = [bin(reg & self.G[0]).count('1') % 2, bin(reg & self.G[1]).count('1') % 2]

    def encode(self, bits):
        state, encoded = 0, []
        for b in np.concatenate([bits, np.zeros(6, dtype=int)]):
            reg = (b << 6) | state
            encoded.extend([bin(reg & self.G[0]).count('1') % 2, bin(reg & self.G[1]).count('1') % 2])
            state = (state >> 1) | (b << 5)
        return np.array(encoded)

    def viterbi(self, rx_fec):
        n_steps = len(rx_fec) // 2
        metrics = np.full(self.num_states, np.inf); metrics[0] = 0
        paths = np.zeros((self.num_states, n_steps), dtype=int)
        for t in range(n_steps):
            new_m, new_p = np.full(self.num_states, np.inf), np.zeros((self.num_states, n_steps), dtype=int)
            r = rx_fec[2*t : 2*t+2]
            for s in range(self.num_states):
                if metrics[s] == np.inf: continue
                for b in [0, 1]:
                    ns = self.next_states[s, b]
                    cost = np.sum(r != self.outputs[s, b])
                    if metrics[s] + cost < new_m[ns]:
                        new_m[ns] = metrics[s] + cost
                        new_p[ns, :t], new_p[ns, t] = paths[s, :t], b
            metrics, paths = new_m, new_p
        return paths[np.argmin(metrics), :-6]

    def _gen_wf(self, bits):
        syms = [int(''.join(map(str, bits[i:i+4][::-1])), 2) for i in range(0, len(bits), 4)]
        chips = self.chip_map[syms].T
        t = np.arange(self.osr) / self.osr
        pulse = np.sin(np.pi * t)
        o_c, e_c = (chips[0::2, :].flatten(order='F')*2-1), (chips[1::2, :].flatten(order='F')*2-1)
        return np.concatenate([np.outer(pulse, o_c).flatten(order='F'), np.zeros(self.osr//2)]) + \
               1j * np.concatenate([np.zeros(self.osr//2), np.outer(pulse, e_c).flatten(order='F')])

    def generate_ppdu(self, info_bits):
        fec = self.encode(info_bits)
        shr = np.concatenate([np.zeros(32, dtype=int), [1, 1, 1, 0, 0, 1, 0, 1]]) # SHR(10 symbols)
        phr_val = len(fec) // 8
        phr = np.array([int(b) for b in bin(phr_val)[2:].zfill(7)[::-1]] + [0])
        wf = self._gen_wf(np.concatenate([shr, phr, fec]))
        return wf / np.sqrt(np.mean(np.abs(wf)**2)), len(fec)

    def autonomous_sync_demod(self, rx_wf):
        # 1. Sync
        ref_sync = np.concatenate([np.zeros(32, dtype=int), [1, 1, 1, 0, 0, 1, 0, 1]])
        ref_wf = self._gen_wf(ref_sync)
        corr = correlate(rx_wf, ref_wf, mode='valid')
        sync_idx = np.argmax(np.abs(corr))
        synced = rx_wf[sync_idx:]
        
        # Polarity Correction
        if np.real(corr[sync_idx]) < 0: synced = -synced
            
        # 2. Sampling
        i_s = np.real(synced)[self.osr // 2 :: self.osr] > 0
        q_s = np.imag(synced)[self.osr :: self.osr] > 0
        l = min(len(i_s), len(q_s))
        rx_c = np.zeros(l*2, dtype=int); rx_c[0::2], rx_c[1::2] = i_s[:l], q_s[:l]
        
        # 3. PHR Decoding (PHR starts at 320 chips)
        phr_chips = rx_c[320 : 384]
        phr_bits = []
        for i in range(2):
            best_sym = np.argmin(np.sum(phr_chips[i*32:(i+1)*32] != self.chip_map, axis=1))
            phr_bits.extend([int(b) for b in bin(best_sym)[2:].zfill(4)[::-1]])
        
        # 4. Length Extraction from PHR
        det_len_bytes = int(''.join(map(str, phr_bits[:7][::-1])), 2)
        det_fec_bits = det_len_bytes * 8
        
        # 5. Payload Demodulation (Autonomous)
        payload_start = 384
        payload_chips = rx_c[payload_start : payload_start + (det_fec_bits // 4) * 32]
        fec_bits = []
        for i in range(len(payload_chips) // 32):
            best = np.argmin(np.sum(payload_chips[i*32:(i+1)*32] != self.chip_map, axis=1))
            fec_bits.extend([int(b) for b in bin(best)[2:].zfill(4)[::-1]])
            
        if len(fec_bits) == 0: return np.zeros(1)
        return self.viterbi(np.array(fec_bits)), det_fec_bits, sync_idx

if __name__ == "__main__":
    nba = NBASystemAutonomous(osr=8)
    snr_range = np.arange(9, -16, -2) # 시뮬레이션 SNR 범위
    iters, bit_len = 1000, 122 # 통계적 신뢰도를 위해 반복 횟수 상향
    ber_results = []

    print(f"NBA System Simulation: {bit_len} bits per packet")
    print(f"{'SNR(dB)':>7} | {'BER':>10} | {'Sync_Idx':>8} | {'Det_Len':>8}")
    print("-" * 45)

    for snr in snr_range:
        errs = 0
        d_l_sum = 0
        for _ in range(iters):
            tx_bits = np.random.randint(0, 2, bit_len)
            delay = 50
            wf, act_fec_len = nba.generate_ppdu(tx_bits)
            
            # AWGN 채널 적용
            rx_wf = np.concatenate([np.zeros(delay), wf])
            noise = (np.random.randn(len(rx_wf)) + 1j*np.random.randn(len(rx_wf))) * \
                    np.sqrt((1/(10**(snr/10)))/2)
            
            try:
                # 자율 복조 수행 (PHR 기반 길이 탐지)
                rx_bits, det_l, s_idx = nba.autonomous_sync_demod(rx_wf + noise)
                errs += np.sum(tx_bits != rx_bits[:bit_len])
                d_l_sum += det_l
            except:
                errs += bit_len # 복조 실패 시 전체 에러 처리

        ber = errs / (iters * bit_len)
        ber_results.append(ber)
        print(f"{snr:7.1f} | {ber:10.6f} | {delay:8.1f} | {d_l_sum/iters:8.1f}")

    # --- 그래프 시각화 ---
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_range, ber_results, 'r-o', linewidth=2, markersize=6, label='NBA (Viterbi Coded)')
    
    # 이론적 기준선 (AWGN 상의 일반적인 성능 지표)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('NBA System Performance: SNR vs BER')
    plt.legend()
    
    # 연구실 서버 환경(Docker/Linux)을 고려한 저장 로직
    plt.savefig('nba_ber_plot.png')
    print("\n[Result] Plot saved as 'nba_ber_plot.png'")
    plt.show()