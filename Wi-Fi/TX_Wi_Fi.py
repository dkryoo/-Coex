import numpy as np
from dataclasses import dataclass
import os
import sys
from pathlib import Path

try:
    from Channel.Rician import apply_distance_rician_channel_with_thermal_noise
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Channel.Rician import apply_distance_rician_channel_with_thermal_noise


@dataclass(frozen=True)
class MCSProfile:
    index: int
    modulation: str
    code_rate: float
    n_bpsc: int


class WiFiOFDMTx:
    """
    Wi-Fi 6E / 7 style OFDM TX (single spatial stream, GI=0.8 us).

    Supported:
    - Standards: wifi6e (MCS 0..11), wifi7 (MCS 0..13)
    - Channel bonding bandwidths: 20 / 40 / 80 / 160 MHz

    Simplifications for simulation:
    - No preamble/pilots/scrambler/FEC/interleaver modeling.
    - Data subcarriers are populated directly with random QAM symbols.
    """

    # 11ax/11be common numerology: subcarrier spacing 78.125 kHz
    SUBCARRIER_SPACING_HZ = 78_125.0
    T_FFT_S = 12.8e-6
    GI_S = 0.8e-6
    T_SYM_S = T_FFT_S + GI_S

    BANDWIDTH_OPTIONS_MHZ = (20, 40, 80, 160)

    # Full-band data subcarriers (Nsd) for HE/EHT (NSS=1 rate basis)
    NSD_BY_BW = {
        20: 234,
        40: 468,
        80: 980,
        160: 1960,
    }

    # Base MCS set (0..11) for Wi-Fi 6E (11ax)
    MCS_TABLE_11AX = [
        MCSProfile(0, "BPSK", 1 / 2, 1),
        MCSProfile(1, "QPSK", 1 / 2, 2),
        MCSProfile(2, "QPSK", 3 / 4, 2),
        MCSProfile(3, "16-QAM", 1 / 2, 4),
        MCSProfile(4, "16-QAM", 3 / 4, 4),
        MCSProfile(5, "64-QAM", 2 / 3, 6),
        MCSProfile(6, "64-QAM", 3 / 4, 6),
        MCSProfile(7, "64-QAM", 5 / 6, 6),
        MCSProfile(8, "256-QAM", 3 / 4, 8),
        MCSProfile(9, "256-QAM", 5 / 6, 8),
        MCSProfile(10, "1024-QAM", 3 / 4, 10),
        MCSProfile(11, "1024-QAM", 5 / 6, 10),
    ]

    # Wi-Fi 7 (11be) extension
    MCS_TABLE_11BE_EXTRA = [
        MCSProfile(12, "4096-QAM", 3 / 4, 12),
        MCSProfile(13, "4096-QAM", 5 / 6, 12),
    ]

    def __init__(self, rng_seed: int | None = None, center_freq_hz: float = 6.5e9):
        self.rng = np.random.default_rng(rng_seed)
        if center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be > 0")
        self.center_freq_hz = float(center_freq_hz)

    def set_center_frequency(self, center_freq_hz: float) -> None:
        if center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be > 0")
        self.center_freq_hz = float(center_freq_hz)

    @staticmethod
    def db_to_w(db: float) -> float:
        return 10.0 ** (db / 10.0)

    @staticmethod
    def w_to_db(watt: float) -> float:
        return 10.0 * np.log10(max(watt, 1e-30))

    def scale_waveform_to_tx_power_dbw(self, wf: np.ndarray, tx_power_dbw: float) -> np.ndarray:
        """
        Scale waveform so mean(|wf|^2) equals tx_power_dbw in watts-domain.
        """
        p_target_w = self.db_to_w(tx_power_dbw)
        p_now_w = float(np.mean(np.abs(wf) ** 2)) + 1e-30
        return (wf * np.sqrt(p_target_w / p_now_w)).astype(np.complex128)

    def _get_mcs_table(self, standard: str) -> list[MCSProfile]:
        s = standard.lower().strip()
        if s == "wifi6e":
            return self.MCS_TABLE_11AX
        if s == "wifi7":
            return self.MCS_TABLE_11AX + self.MCS_TABLE_11BE_EXTRA
        raise ValueError("standard must be 'wifi6e' or 'wifi7'")

    def _channel_params(self, channel_bw_mhz: int) -> tuple[float, int, int, int]:
        if channel_bw_mhz not in self.BANDWIDTH_OPTIONS_MHZ:
            raise ValueError(f"channel_bw_mhz must be one of {self.BANDWIDTH_OPTIONS_MHZ}")

        fs_hz = float(channel_bw_mhz) * 1e6
        # 20 MHz -> 256 FFT in 11ax/11be numerology
        fft_n = 256 * (channel_bw_mhz // 20)
        cp_n = int(round(self.GI_S * fs_hz))
        nsd = self.NSD_BY_BW[channel_bw_mhz]
        return fs_hz, fft_n, cp_n, nsd

    @staticmethod
    def _required_phy_rate_mbps(
        target_rx_throughput_mbps: float,
        rx_success_prob: float,
        mac_efficiency: float,
    ) -> float:
        if target_rx_throughput_mbps <= 0:
            raise ValueError("target_rx_throughput_mbps must be > 0")
        if not (0 < rx_success_prob <= 1):
            raise ValueError("rx_success_prob must be in (0, 1]")
        if not (0 < mac_efficiency <= 1):
            raise ValueError("mac_efficiency must be in (0, 1]")
        return target_rx_throughput_mbps / (rx_success_prob * mac_efficiency)

    def _phy_rate_mbps(self, mcs: MCSProfile, nsd: int) -> float:
        # Single-stream PHY rate approximation: Nsd * Nbpsc * R / Tsym
        bits_per_symbol = nsd * mcs.n_bpsc * mcs.code_rate
        return bits_per_symbol / self.T_SYM_S / 1e6

    def select_mcs(
        self,
        target_rx_throughput_mbps: float,
        channel_bw_mhz: int = 20,
        standard: str = "wifi6e",
        rx_success_prob: float = 0.9,
        mac_efficiency: float = 0.8,
    ) -> tuple[MCSProfile, float, float]:
        req_phy = self._required_phy_rate_mbps(
            target_rx_throughput_mbps=target_rx_throughput_mbps,
            rx_success_prob=rx_success_prob,
            mac_efficiency=mac_efficiency,
        )
        _, _, _, nsd = self._channel_params(channel_bw_mhz)

        table = self._get_mcs_table(standard)
        for m in table:
            phy = self._phy_rate_mbps(m, nsd)
            if phy >= req_phy:
                return m, req_phy, phy

        m = table[-1]
        return m, req_phy, self._phy_rate_mbps(m, nsd)

    @staticmethod
    def _gray_to_binary(gray: np.ndarray) -> np.ndarray:
        binary = gray.copy()
        shift = 1
        while shift < 32:
            binary ^= (binary >> shift)
            shift <<= 1
        return binary

    def _qam_map(self, bits: np.ndarray, n_bpsc: int) -> np.ndarray:
        bits = bits.astype(int).reshape(-1, n_bpsc)

        if n_bpsc == 1:
            return (2 * bits[:, 0] - 1).astype(np.complex128)

        k = n_bpsc // 2
        m_side = 2**k

        i_gray = np.zeros(bits.shape[0], dtype=np.int32)
        q_gray = np.zeros(bits.shape[0], dtype=np.int32)
        for b in range(k):
            i_gray = (i_gray << 1) | bits[:, b]
            q_gray = (q_gray << 1) | bits[:, k + b]

        i_bin = self._gray_to_binary(i_gray)
        q_bin = self._gray_to_binary(q_gray)

        i_level = 2 * i_bin - (m_side - 1)
        q_level = 2 * q_bin - (m_side - 1)

        m_order = 2**n_bpsc
        norm = np.sqrt((2.0 / 3.0) * (m_order - 1.0))
        return (i_level + 1j * q_level) / norm

    @staticmethod
    def _data_subcarrier_bins(fft_n: int, nsd: int) -> np.ndarray:
        half = nsd // 2
        neg = np.arange(-half, 0)
        pos = np.arange(1, half + 1)
        sc = np.concatenate([neg, pos])
        return (sc % fft_n).astype(int)

    def _gen_ofdm_symbol(self, mcs: MCSProfile, fft_n: int, cp_n: int, nsd: int) -> np.ndarray:
        n_cbps = nsd * mcs.n_bpsc
        coded_bits = self.rng.integers(0, 2, n_cbps, dtype=int)
        data_syms = self._qam_map(coded_bits, n_bpsc=mcs.n_bpsc)

        X = np.zeros(fft_n, dtype=np.complex128)
        bins = self._data_subcarrier_bins(fft_n=fft_n, nsd=nsd)
        X[bins] = data_syms

        x = np.fft.ifft(X, n=fft_n)
        cp = x[-cp_n:]
        return np.concatenate([cp, x])

    def generate_for_target_rx_throughput(
        self,
        target_rx_throughput_mbps: float,
        duration_s: float = 0.01,
        channel_bw_mhz: int = 20,
        standard: str = "wifi6e",
        rx_success_prob: float = 0.9,
        mac_efficiency: float = 0.8,
        force_mcs: int | None = None,
        center_freq_hz: float | None = None,
        tx_power_dbw: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate Wi-Fi 6E/7 OFDM baseband for a target RX throughput.

        Returns:
          waveform: complex baseband samples
          info: selected PHY parameters and throughput summary
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        if center_freq_hz is None:
            center_freq_hz = self.center_freq_hz
        if center_freq_hz <= 0:
            raise ValueError("center_freq_hz must be > 0")

        fs_hz, fft_n, cp_n, nsd = self._channel_params(channel_bw_mhz)
        table = self._get_mcs_table(standard)

        if force_mcs is None:
            mcs, req_phy, sel_phy = self.select_mcs(
                target_rx_throughput_mbps=target_rx_throughput_mbps,
                channel_bw_mhz=channel_bw_mhz,
                standard=standard,
                rx_success_prob=rx_success_prob,
                mac_efficiency=mac_efficiency,
            )
        else:
            idx_map = {m.index: m for m in table}
            if force_mcs not in idx_map:
                max_idx = max(idx_map.keys())
                raise ValueError(f"force_mcs must be within supported set for {standard} (max {max_idx})")
            mcs = idx_map[force_mcs]
            req_phy = self._required_phy_rate_mbps(
                target_rx_throughput_mbps=target_rx_throughput_mbps,
                rx_success_prob=rx_success_prob,
                mac_efficiency=mac_efficiency,
            )
            sel_phy = self._phy_rate_mbps(mcs, nsd)

        n_symbols = max(1, int(np.floor(duration_s / self.T_SYM_S)))
        symbols = [self._gen_ofdm_symbol(mcs, fft_n=fft_n, cp_n=cp_n, nsd=nsd) for _ in range(n_symbols)]

        wf = np.concatenate(symbols)
        wf = wf / np.sqrt(np.mean(np.abs(wf) ** 2) + 1e-30)
        if tx_power_dbw is not None:
            wf = self.scale_waveform_to_tx_power_dbw(wf, tx_power_dbw=tx_power_dbw)

        expected_rx_tp = sel_phy * rx_success_prob * mac_efficiency

        info = {
            "standard": standard,
            "center_freq_hz": float(center_freq_hz),
            "channel_bw_mhz": channel_bw_mhz,
            "sample_rate_hz": fs_hz,
            "fft_n": fft_n,
            "cp_n": cp_n,
            "n_data_subcarriers": nsd,
            "mcs_index": mcs.index,
            "modulation": mcs.modulation,
            "code_rate": mcs.code_rate,
            "phy_rate_mbps": sel_phy,
            "required_phy_rate_mbps": req_phy,
            "target_rx_throughput_mbps": target_rx_throughput_mbps,
            "expected_rx_throughput_mbps": expected_rx_tp,
            "duration_s": duration_s,
            "n_ofdm_symbols": n_symbols,
            "n_samples": int(wf.size),
            "tx_power_dbw": self.w_to_db(float(np.mean(np.abs(wf) ** 2))),
        }
        return wf.astype(np.complex128), info

    @staticmethod
    def save_waveform_preview_png(
        wf: np.ndarray,
        fs_hz: float,
        out_path: str = "Wi-Fi/tx_wifi_preview.png",
        max_time_samples: int = 4000,
        welch_nfft: int = 4096,
        welch_seg_len: int = 2048,
        welch_overlap: float = 0.5,
    ) -> str:
        """
        Save quick-look plots (time-domain IQ + Welch PSD) to PNG.

        Spectrum unit:
        - dBFS/Hz, where FS reference is complex-signal power 1.0 (mean |x|^2 = 1).
        """
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        n = min(max_time_samples, len(wf))
        t_us = np.arange(n) / fs_hz * 1e6

        f_mhz, psd_dbfs_hz = WiFiOFDMTx._welch_psd_dbfs_per_hz(
            wf=wf,
            fs_hz=fs_hz,
            nfft=welch_nfft,
            seg_len=welch_seg_len,
            overlap=welch_overlap,
        )

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(t_us, np.real(wf[:n]), label="I", lw=1.0)
        ax[0].plot(t_us, np.imag(wf[:n]), label="Q", lw=1.0, alpha=0.9)
        ax[0].set_title("Wi-Fi TX Waveform (Time Domain)")
        ax[0].set_xlabel("Time [us]")
        ax[0].set_ylabel("Amplitude")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        ax[1].plot(f_mhz, psd_dbfs_hz, lw=1.0)
        ax[1].set_title("Wi-Fi TX Welch PSD")
        ax[1].set_xlabel("Frequency Offset [MHz]")
        ax[1].set_ylabel("PSD [dBFS/Hz]")
        ax[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    @staticmethod
    def _welch_psd_dbfs_per_hz(
        wf: np.ndarray,
        fs_hz: float,
        nfft: int = 4096,
        seg_len: int = 2048,
        overlap: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Welch PSD estimate for complex baseband.
        Returns (frequency_mhz, psd_dbfs_per_hz).
        """
        if fs_hz <= 0:
            raise ValueError("fs_hz must be > 0")
        if seg_len <= 0 or nfft <= 0:
            raise ValueError("seg_len and nfft must be > 0")
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1)")

        x = np.asarray(wf, dtype=np.complex128).flatten()
        if x.size < seg_len:
            x = np.pad(x, (0, seg_len - x.size))

        hop = max(1, int(seg_len * (1.0 - overlap)))
        w = np.hanning(seg_len)
        w_pow = np.sum(w**2) + 1e-30

        psd_acc = np.zeros(nfft, dtype=float)
        count = 0
        for start in range(0, x.size - seg_len + 1, hop):
            seg = x[start : start + seg_len] * w
            X = np.fft.fft(seg, n=nfft)
            # Two-sided PSD [power/Hz] for complex baseband.
            psd = (np.abs(X) ** 2) / (fs_hz * w_pow)
            psd_acc += psd
            count += 1

        if count == 0:
            seg = np.pad(x, (0, max(0, seg_len - x.size)))[:seg_len] * w
            X = np.fft.fft(seg, n=nfft)
            psd_acc = (np.abs(X) ** 2) / (fs_hz * w_pow)
            count = 1

        psd_avg = psd_acc / count
        psd_shift = np.fft.fftshift(psd_avg)
        f_mhz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_hz)) / 1e6
        psd_db = 10.0 * np.log10(psd_shift + 1e-30)
        return f_mhz, psd_db

    @staticmethod
    def save_waveform_preview_csv(
        wf: np.ndarray,
        fs_hz: float,
        out_time_csv: str = "Wi-Fi/tx_wifi_time.csv",
        out_spec_csv: str = "Wi-Fi/tx_wifi_spectrum.csv",
        max_time_samples: int = 4000,
        welch_nfft: int = 4096,
        welch_seg_len: int = 2048,
        welch_overlap: float = 0.5,
    ) -> tuple[str, str]:
        os.makedirs(os.path.dirname(out_time_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(out_spec_csv) or ".", exist_ok=True)
        n = min(max_time_samples, len(wf))
        t_us = np.arange(n) / fs_hz * 1e6
        time_mat = np.column_stack([t_us, np.real(wf[:n]), np.imag(wf[:n])])
        np.savetxt(out_time_csv, time_mat, delimiter=",", header="time_us,I,Q", comments="")

        f_mhz, psd_dbfs_hz = WiFiOFDMTx._welch_psd_dbfs_per_hz(
            wf=wf,
            fs_hz=fs_hz,
            nfft=welch_nfft,
            seg_len=welch_seg_len,
            overlap=welch_overlap,
        )
        spec_mat = np.column_stack([f_mhz, psd_dbfs_hz])
        np.savetxt(
            out_spec_csv,
            spec_mat,
            delimiter=",",
            header="freq_offset_mhz,psd_dbfs_per_hz",
            comments="",
        )
        return out_time_csv, out_spec_csv


if __name__ == "__main__":
    tx = WiFiOFDMTx(rng_seed=7, center_freq_hz=6.5e9)

    wf, info = tx.generate_for_target_rx_throughput(
        target_rx_throughput_mbps=600.0,
        duration_s=0.005,
        channel_bw_mhz=160,
        standard="wifi7",
        rx_success_prob=0.9,
        mac_efficiency=0.8,
        center_freq_hz=6.875e9,
        tx_power_dbw=-20.0,
    )

    print("[Wi-Fi TX] Generated waveform")
    print(
        f"  Standard={info['standard']}, Fc={info['center_freq_hz']/1e9:.3f} GHz, "
        f"BW={info['channel_bw_mhz']} MHz"
    )
    print(f"  MCS={info['mcs_index']} ({info['modulation']}, R={info['code_rate']:.3f})")
    print(f"  PHY={info['phy_rate_mbps']:.1f} Mbps, Required PHY={info['required_phy_rate_mbps']:.1f} Mbps")
    print(f"  Target RX TP={info['target_rx_throughput_mbps']:.1f} Mbps")
    print(f"  Expected RX TP={info['expected_rx_throughput_mbps']:.1f} Mbps")
    print(f"  TX power={info['tx_power_dbw']:.2f} dBW")
    print(f"  Samples={info['n_samples']}, Fs={info['sample_rate_hz']/1e6:.1f} Msps")

    try:
        png_path = tx.save_waveform_preview_png(wf, fs_hz=info["sample_rate_hz"])
        print(f"  Preview saved: {png_path}")
    except Exception as exc:
        print(f"  Preview PNG skipped: {exc}")

    t_csv, s_csv = tx.save_waveform_preview_csv(wf, fs_hz=info["sample_rate_hz"])
    print(f"  Preview CSV saved: {t_csv}, {s_csv}")

    print("\n[Wi-Fi TX -> Distance Rician Channel Demo]")
    demo_distances_m = [5, 20, 50]
    for d_m in demo_distances_m:
        rx_wf, ch_info = apply_distance_rician_channel_with_thermal_noise(
            tx_wf=wf,
            fs_hz=info["sample_rate_hz"],
            fc_hz=info["center_freq_hz"],
            distance_m=float(d_m),
            tx_eirp_db=-20.0,  # Example absolute EIRP in dBW
            pathloss_exp=2.0,
            ref_distance_m=1.0,
            delays_s=(0.0, 30e-9, 80e-9),
            powers_db=(0.0, -6.0, -10.0),
            k_factor_db=6.0,
            nf_db=6.0,
            temperature_k=290.0,
            noise_ref_bw_hz=info["channel_bw_mhz"] * 1e6,
            rx_ant_gain_db=0.0,
            rx_cable_loss_db=0.0,
            rx_lead_zeros=64,
            channel_seed=1000 + d_m,
            noise_seed=2000 + d_m,
        )
        print(
            f"  d={d_m:3d} m | PL={ch_info['pathloss_db']:.2f} dB | "
            f"Pr={ch_info['pr_dbw']:.2f} dBW | N={ch_info['noise_dbw_ref_bw']:.2f} dBW | "
            f"SNR={ch_info['snr_db_ref_bw']:.2f} dB | rx_samples={len(rx_wf)}"
        )

    # Save one RX preview example for visualization.
    rx_wf_ex, _ = apply_distance_rician_channel_with_thermal_noise(
        tx_wf=wf,
        fs_hz=info["sample_rate_hz"],
        fc_hz=info["center_freq_hz"],
        distance_m=20.0,
        tx_eirp_db=-20.0,
        pathloss_exp=2.0,
        delays_s=(0.0, 30e-9, 80e-9),
        powers_db=(0.0, -6.0, -10.0),
        k_factor_db=6.0,
        nf_db=6.0,
        temperature_k=290.0,
        noise_ref_bw_hz=info["channel_bw_mhz"] * 1e6,
        rx_lead_zeros=64,
        channel_seed=1020,
        noise_seed=2020,
    )
    try:
        rx_png = tx.save_waveform_preview_png(
            rx_wf_ex,
            fs_hz=info["sample_rate_hz"],
            out_path="Wi-Fi/rx_wifi_after_rician_preview.png",
        )
        print(f"  RX preview saved: {rx_png}")
    except Exception as exc:
        print(f"  RX preview PNG skipped: {exc}")
    rx_t_csv, rx_s_csv = tx.save_waveform_preview_csv(
        rx_wf_ex,
        fs_hz=info["sample_rate_hz"],
        out_time_csv="Wi-Fi/rx_wifi_after_rician_time.csv",
        out_spec_csv="Wi-Fi/rx_wifi_after_rician_spectrum.csv",
    )
    print(f"  RX preview CSV saved: {rx_t_csv}, {rx_s_csv}")
