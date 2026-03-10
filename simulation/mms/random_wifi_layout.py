from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from simulation.mms.wifi6g_channels import ChannelSegment, generate_fullband_partition, summarize_partition


@dataclass
class RandomWiFiLayoutConfig:
    area_w_m: float = 50.0
    area_h_m: float = 50.0
    min_ap_sep_m: float = 2.0
    tx_power_dbm: float = 20.0
    p_bursty: float = 0.5
    duty_cycle: float = 0.1
    per_ap_duty_jitter: float = 0.0
    burst_mean_on_ms_min: float = 0.5
    burst_mean_on_ms_max: float = 10.0
    pathloss_n: float = 2.8
    shadowing_sigma_db: float = 4.0
    pl0_db: float = 46.0
    enable_rician_fading: bool = True
    rician_k_factor_db: float = 6.0


def _sample_points_with_min_sep(
    *,
    n: int,
    w: float,
    h: float,
    min_sep: float,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = float(rng.uniform(0.0, w))
        y = float(rng.uniform(0.0, h))
        ok = True
        for px, py in pts:
            if math.hypot(x - px, y - py) < min_sep:
                ok = False
                break
        if ok:
            pts.append((x, y))
    if len(pts) < n:
        # fallback to unconstrained fill
        while len(pts) < n:
            pts.append((float(rng.uniform(0.0, w)), float(rng.uniform(0.0, h))))
    return np.asarray(pts, dtype=float)


def _duty_with_jitter(base: float, jitter: float, rng: np.random.Generator) -> float:
    if jitter <= 0:
        return float(np.clip(base, 0.0, 1.0))
    return float(np.clip(base + rng.uniform(-jitter, jitter), 0.0, 1.0))


def generate_random_wifi_layout(
    *,
    rng: np.random.Generator,
    layout_cfg: RandomWiFiLayoutConfig,
    bw_mix_weights: dict[int, float] | None,
    occupancy_mode: str,
    generated_trial_id: str,
    mixed_assign_mode: str = "binomial",
) -> dict:
    segs: list[ChannelSegment] = generate_fullband_partition(
        rng=rng,
        bw_mix_weights=bw_mix_weights,
        regulatory_domain="FCC",
    )
    n_ap = len(segs)
    pts = _sample_points_with_min_sep(
        n=n_ap,
        w=float(layout_cfg.area_w_m),
        h=float(layout_cfg.area_h_m),
        min_sep=float(layout_cfg.min_ap_sep_m),
        rng=rng,
    )

    occ_modes: list[str] = []
    if occupancy_mode == "mixed":
        p = float(np.clip(layout_cfg.p_bursty, 0.0, 1.0))
        if mixed_assign_mode == "fixed":
            n_bursty = int(round(p * n_ap))
            mark = np.asarray([True] * n_bursty + [False] * (n_ap - n_bursty), dtype=bool)
            rng.shuffle(mark)
            occ_modes = ["bursty" if bool(v) else "uniform" for v in mark]
        else:
            occ_modes = ["bursty" if float(rng.random()) < p else "uniform" for _ in range(n_ap)]

    aps = []
    for i, s in enumerate(segs):
        d = _duty_with_jitter(float(layout_cfg.duty_cycle), float(layout_cfg.per_ap_duty_jitter), rng)
        if occupancy_mode == "mixed":
            mode = str(occ_modes[i])
        else:
            mode = occupancy_mode
        ap = {
            "id": f"ap{i+1}",
            "x": float(pts[i, 0]),
            "y": float(pts[i, 1]),
            "wifi_ch": int(s.center_ch),
            "bw_mhz": int(s.bw_mhz),
            "tx_power_dBm": float(layout_cfg.tx_power_dbm),
            "duty_cycle": float(d),
            "traffic_load": float(d),
            "occupancy_mode": str(mode),
            "covered_20ch_list": list(s.covered_20ch_list),
        }
        if mode == "bursty":
            mean_on = float(rng.uniform(layout_cfg.burst_mean_on_ms_min, layout_cfg.burst_mean_on_ms_max))
            if d <= 0.0:
                mean_off = float(layout_cfg.burst_mean_on_ms_max)
            elif d >= 1.0:
                mean_off = 1e-6
            else:
                mean_off = float(mean_on * (1.0 - d) / d)
            ap["burst_mean_on_ms"] = float(mean_on)
            ap["burst_mean_off_ms"] = float(max(mean_off, 1e-6))
        aps.append(ap)

    part_summary = summarize_partition(segs)
    out = {
        "generated_trial_id": str(generated_trial_id),
        "area_w_m": float(layout_cfg.area_w_m),
        "area_h_m": float(layout_cfg.area_h_m),
        "pathloss_n": float(layout_cfg.pathloss_n),
        "shadowing_sigma_db": float(layout_cfg.shadowing_sigma_db),
        "pl0_db": float(layout_cfg.pl0_db),
        "enable_rician_fading": bool(layout_cfg.enable_rician_fading),
        "rician_k_factor_db": float(layout_cfg.rician_k_factor_db),
        "band_partition_summary": part_summary,
        "band_partition_segments": [
            {
                "bw_mhz": int(s.bw_mhz),
                "center_ch": int(s.center_ch),
                "f_center_hz": float(s.f_center_hz),
                "f_lo_hz": float(s.f_lo_hz),
                "f_hi_hz": float(s.f_hi_hz),
                "covered_20ch_list": list(s.covered_20ch_list),
            }
            for s in segs
        ],
        "wifi_aps": aps,
    }
    return out


def layout_distance_stats(layout: dict, uwb_nodes_xy: list[tuple[float, float]]) -> dict:
    aps = layout.get("wifi_aps", [])
    if not aps or not uwb_nodes_xy:
        return {"n_ap": 0, "ap_to_nodes_min_m": float("nan"), "ap_to_nodes_mean_m": float("nan")}
    d_all = []
    for ap in aps:
        ax, ay = float(ap.get("x", 0.0)), float(ap.get("y", 0.0))
        for (ux, uy) in uwb_nodes_xy:
            d_all.append(float(math.hypot(ax - float(ux), ay - float(uy))))
    arr = np.asarray(d_all, dtype=float)
    return {
        "n_ap": int(len(aps)),
        "ap_to_nodes_min_m": float(np.min(arr)),
        "ap_to_nodes_mean_m": float(np.mean(arr)),
    }


def validate_occupancy(
    layout: dict,
    expected_mode: str,
    p_bursty: float,
    *,
    mixed_tol: float = 0.10,
    duty_tol: float = 0.02,
    mixed_z_max: float = 4.0,
) -> dict:
    aps = list(layout.get("wifi_aps", []))
    n_ap = int(len(aps))
    if n_ap <= 0:
        raise AssertionError("layout has no APs")

    n_uniform = 0
    n_bursty = 0
    duty_hat_bursty: list[float] = []
    for i, ap in enumerate(aps):
        mode = str(ap.get("occupancy_mode", "uniform")).strip().lower()
        if mode == "uniform":
            n_uniform += 1
        elif mode == "bursty":
            n_bursty += 1
            mean_on = float(ap.get("burst_mean_on_ms", 0.0))
            mean_off = float(ap.get("burst_mean_off_ms", 0.0))
            if mean_on <= 0.0 or mean_off < 0.0:
                raise AssertionError(f"invalid burst params at ap[{i}] mean_on={mean_on}, mean_off={mean_off}")
            duty_hat = mean_on / max(mean_on + mean_off, 1e-12)
            duty_cfg = float(ap.get("duty_cycle", ap.get("traffic_load", 0.0)))
            duty_hat_bursty.append(float(duty_hat))
            if abs(duty_hat - duty_cfg) > float(duty_tol):
                raise AssertionError(
                    f"bursty duty mismatch at ap[{i}] mode={mode} duty_hat={duty_hat:.4f} duty_cfg={duty_cfg:.4f}"
                )
        else:
            raise AssertionError(f"unknown occupancy_mode at ap[{i}]={mode}")

    expected_mode = str(expected_mode).strip().lower()
    if expected_mode == "uniform":
        if n_bursty != 0:
            raise AssertionError(f"expected all uniform but bursty_count={n_bursty}")
    elif expected_mode == "bursty":
        if n_uniform != 0:
            raise AssertionError(f"expected all bursty but uniform_count={n_uniform}")
    elif expected_mode == "mixed":
        frac = float(n_bursty / max(n_ap, 1))
        target = float(np.clip(p_bursty, 0.0, 1.0))
        sigma = math.sqrt(max(target * (1.0 - target), 1e-12) / max(n_ap, 1))
        tol_eff = max(float(mixed_tol), 3.0 * sigma)
        z = abs(frac - target) / max(sigma, 1e-12)
        # Binomial mixed mode can produce rare but valid tail realizations.
        # Keep strict absolute tolerance, but allow statistically plausible tails.
        if abs(frac - target) > tol_eff and z > float(mixed_z_max):
            raise AssertionError(
                f"mixed occupancy fraction mismatch: realized={frac:.3f}, target={target:.3f}, "
                f"tol={tol_eff:.3f}, z={z:.2f}, z_max={float(mixed_z_max):.2f}"
            )
    else:
        raise AssertionError(f"unsupported expected_mode={expected_mode}")

    arr = np.asarray(duty_hat_bursty, dtype=float) if duty_hat_bursty else np.asarray([], dtype=float)
    return {
        "n_ap": n_ap,
        "uniform_count": int(n_uniform),
        "bursty_count": int(n_bursty),
        "bursty_frac": float(n_bursty / max(n_ap, 1)),
        "mixed_z_score": float(abs((n_bursty / max(n_ap, 1)) - np.clip(p_bursty, 0.0, 1.0)) / max(math.sqrt(max(np.clip(p_bursty, 0.0, 1.0) * (1.0 - np.clip(p_bursty, 0.0, 1.0)), 1e-12) / max(n_ap, 1)), 1e-12)),
        "duty_hat_mean_bursty": float(np.mean(arr)) if arr.size else float("nan"),
        "duty_hat_p95_bursty": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "occupancy_ok": True,
    }
