from __future__ import annotations

import math
import numpy as np

from Channel.Rician import apply_distance_rician_channel_with_thermal_noise
from UWB.mms_uwb_packet_mode import MmsUwbConfig, build_mms_uwb_fragments, RSTU_S, C_MPS, get_rsf_base_sequence


def _ds_twr_tof_rstu(
    i_tx1: float,
    i_rx1: float,
    r_rx1: float,
    r_tx1: float,
    i_tx2: float,
    i_rx2: float,
    r_rx2: float,
    r_tx2: float,
) -> float:
    ra = i_rx1 - i_tx1
    rb = i_rx2 - i_tx2
    da = r_tx1 - r_rx1
    db = r_tx2 - r_rx2
    denom = ra + rb + da + db
    if denom <= 1e-12:
        return 0.0
    return max(0.0, (ra * rb - da * db) / denom)


def _ber_bpsk_awgn(snr_db: float) -> float:
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(max(snr_lin, 1e-18)))


def _build_rsf_template(cfg: MmsUwbConfig) -> np.ndarray:
    base = get_rsf_base_sequence(code_index=cfg.phy_uwb_mms_rsf_code_index).astype(np.float64)
    if len(base) == 0:
        raise ValueError("Empty RSF base sequence")
    n_chips = max(1, cfg.phy_uwb_mms_rsf_length_units512 * 512)
    reps = int(np.ceil(n_chips / len(base)))
    seq = np.tile(base, reps)[:n_chips]
    # Include RSF repetition count in template energy.
    seq = np.tile(seq, max(1, cfg.phy_uwb_mms_rsf_reps))
    seq = seq / (np.sqrt(np.mean(np.abs(seq) ** 2)) + 1e-30)
    return seq.astype(np.complex128)


def _first_path_legacy(
    corr: np.ndarray,
    rx_lead_zeros: int,
    fp_alpha: float,
) -> tuple[float, float]:
    peak = int(np.argmax(corr))
    peak_val = float(corr[peak])
    n_noise = max(32, min(256, len(corr) // 8))
    noise_floor = float(np.median(corr[:n_noise]))
    search_back = min(2048, peak + 1)
    left = max(rx_lead_zeros, peak - search_back + 1)
    local = corr[left : peak + 1]
    thr = noise_floor + fp_alpha * max(peak_val - noise_floor, 1e-12)
    idx_rel = np.where(local >= thr)[0]
    fp = int(left + idx_rel[0]) if len(idx_rel) > 0 else peak

    fp_f = float(fp)
    if 1 <= fp < (len(corr) - 1):
        y1 = corr[fp - 1]
        y2 = corr[fp]
        y3 = corr[fp + 1]
        den = (y1 - 2.0 * y2 + y3)
        if abs(den) > 1e-18:
            delta = 0.5 * (y1 - y3) / den
            fp_f = fp_f + float(np.clip(delta, -0.5, 0.5))
    return fp_f, float(peak)


def _first_path_cfar_leading_edge(
    corr: np.ndarray,
    rx_lead_zeros: int,
    fp_alpha: float,
    cfar_guard: int,
    cfar_train: int,
    cfar_mult: float,
) -> tuple[float, float]:
    peak = int(np.argmax(corr))
    peak_val = float(corr[peak])
    n_noise = max(32, min(256, len(corr) // 8))
    noise_floor = float(np.median(corr[:n_noise]))

    search_back = min(2048, peak + 1)
    left = max(rx_lead_zeros, peak - search_back + 1)
    right = peak
    det_idx = None
    g = max(1, int(cfar_guard))
    t = max(4, int(cfar_train))
    mult = max(1.0, float(cfar_mult))
    for i in range(left + t + g, right - t - g + 1):
        l0 = i - g - t
        l1 = i - g
        r0 = i + g + 1
        r1 = i + g + 1 + t
        noise_l = float(np.mean(corr[l0:l1]))
        noise_r = float(np.mean(corr[r0:r1]))
        noise_hat = 0.5 * (noise_l + noise_r)
        thr_i = max(noise_floor * (1.0 + 0.5 * fp_alpha), mult * noise_hat)
        if corr[i] >= thr_i and corr[i] >= corr[i - 1]:
            det_idx = i
            break
    fp = int(det_idx) if det_idx is not None else peak

    # leading-edge fit
    fp_f = float(fp)
    if fp < (len(corr) - 4):
        win = corr[fp : min(fp + 24, len(corr))]
        local_peak = float(np.max(win)) if len(win) > 0 else peak_val
        if local_peak > 1e-18:
            yn = win / local_peak
            xs = np.arange(len(win), dtype=float)
            mask = (yn >= 0.10) & (yn <= 0.60)
            if int(np.sum(mask)) >= 3:
                a, b = np.polyfit(xs[mask], yn[mask], 1)
                if abs(a) > 1e-12:
                    x_cross = float(np.clip((0.5 - b) / a, 0.0, len(win) - 1))
                    fp_f = float(fp) + x_cross
    return fp_f, float(peak)


def _estimate_rmarker_arrival_rstu(
    tx_template: np.ndarray,
    fs_hz: float,
    fc_hz: float,
    distance_m: float,
    tx_eirp_dbw: float,
    nf_db: float,
    temperature_k: float,
    channel_seed: int,
    noise_seed: int,
    fp_alpha: float = 0.35,
    nlos_alpha: float = 0.0,
    nlos_cap_ratio: float = 0.2,
    detector_mode: str = "legacy",
    cfar_guard: int = 8,
    cfar_train: int = 32,
    cfar_mult: float = 3.7,
) -> tuple[float, float]:
    rx_lead_zeros = 32
    rx_wf, info = apply_distance_rician_channel_with_thermal_noise(
        tx_wf=tx_template,
        fs_hz=499.2e6,
        fc_hz=fc_hz,
        distance_m=distance_m,
        tx_eirp_db=tx_eirp_dbw,
        pathloss_exp=2.0,
        delays_s=(0.0, 6e-9, 12e-9),
        powers_db=(0.0, -6.0, -10.0),
        k_factor_db=8.0,
        nf_db=nf_db,
        temperature_k=temperature_k,
        noise_ref_bw_hz=20e6,
        rx_lead_zeros=rx_lead_zeros,
        channel_seed=channel_seed,
        noise_seed=noise_seed,
    )
    ref = tx_template.astype(np.complex128)
    corr = np.abs(np.convolve(rx_wf, np.conj(ref[::-1]), mode="valid")).astype(np.float64)
    if len(corr) == 0:
        return 0.0, float(info["snr_db_ref_bw"])

    if detector_mode == "cfar":
        fp_f, peak_f = _first_path_cfar_leading_edge(
            corr=corr,
            rx_lead_zeros=rx_lead_zeros,
            fp_alpha=fp_alpha,
            cfar_guard=cfar_guard,
            cfar_train=cfar_train,
            cfar_mult=cfar_mult,
        )
    else:
        fp_f, peak_f = _first_path_legacy(corr=corr, rx_lead_zeros=rx_lead_zeros, fp_alpha=fp_alpha)

    # Excess delay proxy and simple NLOS bias correction.
    excess_samples = max(0.0, peak_f - fp_f)
    fp_f = max(fp_f, float(rx_lead_zeros))
    raw_delay_samples = max(0.0, fp_f - float(rx_lead_zeros))
    # Conservative NLOS correction: bounded so it cannot collapse ToA to zero.
    nlos_bias_samples = min(nlos_alpha * excess_samples, nlos_cap_ratio * raw_delay_samples)
    delay_samples = max(0.0, raw_delay_samples - nlos_bias_samples)

    samp_per_rstu = fs_hz * RSTU_S
    return float(delay_samples / samp_per_rstu), float(info["snr_db_ref_bw"])


def simulate_mms_performance(
    distances_m: list[float] | tuple[float, ...] = (5.0, 10.0, 20.0, 40.0),
    n_trials: int = 200,
    rif_payload_bits: int = 256,
    fc_hz: float = 6.5e9,
    tx_eirp_dbw: float = -20.0,
    nf_db: float = 6.0,
    temperature_k: float = 290.0,
    seed: int = 1234,
    fp_alpha: float = 0.35,
    nlos_alpha: float = 0.0,
    nlos_cap_ratio: float = 0.2,
    detector_mode: str = "legacy",
    cfar_guard: int = 8,
    cfar_train: int = 32,
    cfar_mult: float = 3.7,
    external_interference_penalty_db: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    cfg = MmsUwbConfig(phy_uwb_mms_rsf_number_frags=2, phy_uwb_mms_rif_number_frags=1)
    fs_hz = 499.2e6
    rsf_template = _build_rsf_template(cfg)
    phase_start = 100_000
    i_frags = build_mms_uwb_fragments("initiator", phase_start, cfg)
    r_frags = build_mms_uwb_fragments("responder", phase_start, cfg)
    i_rsf = [f.rmarker_rstu for f in i_frags if f.kind == "RSF"][:2]
    r_rsf = [f.rmarker_rstu for f in r_frags if f.kind == "RSF"][:2]

    out: list[dict] = []
    for d in distances_m:
        bit_err_sum = 0
        bits_sum = n_trials * rif_payload_bits
        frame_err = 0
        dist_err_sq_sum = 0.0
        snr_acc = 0.0

        for k in range(n_trials):
            trial_seed = seed + 10000 * int(d) + k
            d_f = float(d)
            # Four legs with independent fading/noise, RMarker estimated from received waveform.
            a2b_1_rstu, snr1 = _estimate_rmarker_arrival_rstu(
                tx_template=rsf_template,
                fs_hz=fs_hz,
                fc_hz=fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                channel_seed=trial_seed + 100,
                noise_seed=trial_seed + 200,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                detector_mode=detector_mode,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
            )
            b2a_1_rstu, snr2 = _estimate_rmarker_arrival_rstu(
                tx_template=rsf_template,
                fs_hz=fs_hz,
                fc_hz=fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                channel_seed=trial_seed + 300,
                noise_seed=trial_seed + 400,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                detector_mode=detector_mode,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
            )
            a2b_2_rstu, snr3 = _estimate_rmarker_arrival_rstu(
                tx_template=rsf_template,
                fs_hz=fs_hz,
                fc_hz=fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                channel_seed=trial_seed + 500,
                noise_seed=trial_seed + 600,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                detector_mode=detector_mode,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
            )
            b2a_2_rstu, snr4 = _estimate_rmarker_arrival_rstu(
                tx_template=rsf_template,
                fs_hz=fs_hz,
                fc_hz=fc_hz,
                distance_m=d_f,
                tx_eirp_dbw=tx_eirp_dbw,
                nf_db=nf_db,
                temperature_k=temperature_k,
                channel_seed=trial_seed + 700,
                noise_seed=trial_seed + 800,
                fp_alpha=fp_alpha,
                nlos_alpha=nlos_alpha,
                nlos_cap_ratio=nlos_cap_ratio,
                detector_mode=detector_mode,
                cfar_guard=cfar_guard,
                cfar_train=cfar_train,
                cfar_mult=cfar_mult,
            )
            snr_db = 0.25 * (snr1 + snr2 + snr3 + snr4) - float(external_interference_penalty_db)
            snr_acc += snr_db

            # Keep BER model simple but use link SNR from actual channel realizations.
            eff_snr_db = snr_db - 3.0
            ber = _ber_bpsk_awgn(eff_snr_db)
            p_sync = 1.0 / (1.0 + math.exp(-(eff_snr_db - 2.0) / 1.5))
            frame_success = (rng.random() < p_sync)
            if not frame_success:
                bit_err = rif_payload_bits
            else:
                bit_err = int(rng.binomial(rif_payload_bits, min(max(ber, 0.0), 1.0)))
            bit_err_sum += bit_err
            if bit_err > 0:
                frame_err += 1

            # RMarker-wise timing from channel-estimated arrivals.
            i_tx1 = float(i_rsf[0])
            i_tx2 = float(i_rsf[1])
            r_tx1 = float(r_rsf[0])
            r_tx2 = float(r_rsf[1])
            r_rx1 = i_tx1 + a2b_1_rstu
            r_rx2 = i_tx2 + a2b_2_rstu
            i_rx1 = r_tx1 + b2a_1_rstu
            i_rx2 = r_tx2 + b2a_2_rstu

            tof_est_rstu = _ds_twr_tof_rstu(i_tx1, i_rx1, r_rx1, r_tx1, i_tx2, i_rx2, r_rx2, r_tx2)
            d_est = tof_est_rstu * RSTU_S * C_MPS
            dist_err_sq_sum += (d_est - d_f) ** 2

        out.append(
            {
                "distance_m": float(d),
                "snr_db_avg": snr_acc / n_trials,
                "ber": bit_err_sum / max(bits_sum, 1),
                "fer": frame_err / max(n_trials, 1),
                "ranging_rmse_m": math.sqrt(dist_err_sq_sum / max(n_trials, 1)),
            }
        )
    return out
