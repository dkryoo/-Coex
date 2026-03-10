from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load_wifi_tx_class():
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("tx_wifi_module_for_lut_build", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.WiFiOFDMTx


def _parse_list(s: str) -> list[float]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError("list is empty")
    return out


def _build_offsets(args: argparse.Namespace) -> list[float]:
    if args.offsets_mhz:
        return _parse_list(args.offsets_mhz)
    if args.offset_step_mhz is None or args.offset_range_mhz is None:
        raise ValueError("provide either --offsets-mhz or both --offset-step-mhz and --offset-range-mhz")
    step = float(args.offset_step_mhz)
    rng = float(args.offset_range_mhz)
    if step <= 0.0 or rng <= 0.0:
        raise ValueError("offset-step-mhz and offset-range-mhz must be > 0 when offsets-mhz is omitted")
    n = int(np.floor((2.0 * rng) / step)) + 1
    vals = [(-rng + i * step) for i in range(n)]
    if vals[-1] < rng:
        vals.append(rng)
    # Stable rounded values to avoid floating CSV noise.
    return [float(np.round(v, 6)) for v in vals]


def _active_signed_for_mode(*, nfft: int, nsd: int, mode: str) -> np.ndarray:
    half = nsd // 2
    full = np.concatenate([np.arange(-half, 0, dtype=int), np.arange(1, half + 1, dtype=int)])
    if mode == "full":
        return full
    if mode == "partial_center":
        h2 = max(1, half // 2)
        return np.concatenate([np.arange(-h2, 0, dtype=int), np.arange(1, h2 + 1, dtype=int)])
    if mode == "partial_edges":
        q = max(4, half // 4)
        left = np.arange(-half, -half + q, dtype=int)
        right = np.arange(half - q + 1, half + 1, dtype=int)
        return np.concatenate([left, right])
    raise ValueError(f"unknown allocation mode: {mode}")


def _integrate_band_power(f_mhz: np.ndarray, psd_w_hz: np.ndarray, f0_mhz: float, bw_mhz: float) -> float:
    lo = float(f0_mhz) - 0.5 * float(bw_mhz)
    hi = float(f0_mhz) + 0.5 * float(bw_mhz)
    m = (f_mhz >= lo) & (f_mhz <= hi)
    if int(np.sum(m)) < 2:
        return 0.0
    return float(np.trapezoid(psd_w_hz[m], f_mhz[m] * 1e6))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Wi-Fi ACI coupling LUT from waveform PSD integration.")
    ap.add_argument("--wifi-bws-mhz", type=str, default="20,40,80,160")
    ap.add_argument("--nb-bw-mhz", type=float, default=2.0)
    ap.add_argument("--offsets-mhz", type=str, default=None)
    ap.add_argument("--offset-step-mhz", type=float, default=None)
    ap.add_argument("--offset-range-mhz", type=float, default=None)
    ap.add_argument("--allocation-mode", type=str, default="full", choices=["full", "partial_center", "partial_edges"])
    ap.add_argument("--standard", type=str, default="wifi7", choices=["wifi6e", "wifi7"])
    ap.add_argument("--duration-s", type=float, default=0.03)
    ap.add_argument("--aclr-db", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="simulation/mms/results/wifi_aci_lut")
    args = ap.parse_args()

    wifi_bws = [int(round(v)) for v in _parse_list(args.wifi_bws_mhz)]
    offsets = _build_offsets(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    WiFiOFDMTx = _load_wifi_tx_class()
    tx = WiFiOFDMTx(rng_seed=int(args.seed), center_freq_hz=6.5e9)

    rows: list[dict] = []
    sanity_rows: list[dict] = []
    for bw in wifi_bws:
        if args.allocation_mode == "full":
            wf, info = tx.generate_for_target_rx_throughput(
                target_rx_throughput_mbps=800.0,
                duration_s=float(args.duration_s),
                channel_bw_mhz=int(bw),
                standard=str(args.standard),
                tx_power_dbw=-20.0,
                center_freq_hz=6.5e9,
            )
        else:
            dbg0 = tx.build_debug_ofdm_grid(channel_bw_mhz=int(bw), standard=str(args.standard), force_mcs=0)
            nfft = int(dbg0["fft_n"])
            nsd = int(dbg0["n_data_subcarriers_nominal"])
            active = _active_signed_for_mode(nfft=nfft, nsd=nsd, mode=str(args.allocation_mode))
            dbg = tx.build_debug_ofdm_grid(
                channel_bw_mhz=int(bw),
                standard=str(args.standard),
                force_mcs=0,
                active_signed_subcarriers=active,
            )
            sym = np.asarray(dbg["symbol_td"], dtype=np.complex128)
            t_sym = float(tx.T_SYM_S)
            n_sym = max(1, int(np.floor(float(args.duration_s) / t_sym)))
            wf = np.tile(sym, n_sym)
            wf = wf / np.sqrt(float(np.mean(np.abs(wf) ** 2)) + 1e-30)
            info = {"sample_rate_hz": float(bw) * 1e6}

        wf = tx.apply_tx_emission_mask(
            wf,
            fs_hz=float(info["sample_rate_hz"]),
            channel_bw_hz=float(bw) * 1e6,
            aclr_db=float(args.aclr_db),
            seed=int(args.seed) + 17 + int(bw),
        )
        p_total = float(np.mean(np.abs(wf) ** 2)) + 1e-30
        f_mhz, psd_dbfs_hz = tx._welch_psd_dbfs_per_hz(
            wf=wf,
            fs_hz=float(info["sample_rate_hz"]),
            nfft=8192,
            seg_len=4096,
            overlap=0.5,
        )
        psd_w_hz = 10.0 ** (psd_dbfs_hz / 10.0)
        p_psd_total = float(np.trapezoid(psd_w_hz, f_mhz * 1e6))
        delta_db = 10.0 * np.log10((p_psd_total + 1e-30) / (p_total + 1e-30))
        sanity_rows.append(
            {
                "wifi_bw_mhz": int(bw),
                "allocation_mode": str(args.allocation_mode),
                "p_total_w": float(p_total),
                "p_psd_total_w": float(p_psd_total),
                "psd_total_delta_db": float(delta_db),
            }
        )

        for off in offsets:
            p_v = _integrate_band_power(f_mhz, psd_w_hz, f0_mhz=float(off), bw_mhz=float(args.nb_bw_mhz))
            coupling = float(np.clip(p_v / p_total, 1e-15, 1.0))
            rows.append(
                {
                    "wifi_bw_mhz": int(bw),
                    "nb_bw_mhz": float(args.nb_bw_mhz),
                    "allocation_mode": str(args.allocation_mode),
                    "offset_mhz": float(off),
                    "coupling_linear": coupling,
                    "coupling_db": float(10.0 * np.log10(coupling + 1e-30)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["wifi_bw_mhz", "offset_mhz"]).reset_index(drop=True)
    sane = pd.DataFrame(sanity_rows).sort_values("wifi_bw_mhz").reset_index(drop=True)
    df.to_csv(out_dir / "wifi_aci_lut.csv", index=False)
    sane.to_csv(out_dir / "psd_power_sanity.csv", index=False)
    ok = bool(np.all(np.abs(sane["psd_total_delta_db"].to_numpy(dtype=float)) <= 0.5))
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Wi-Fi ACI LUT Build",
                "",
                f"- allocation_mode: `{args.allocation_mode}`",
                f"- standard: `{args.standard}`",
                f"- nb_bw_mhz: `{args.nb_bw_mhz}`",
                f"- offsets_count: `{len(offsets)}`",
                f"- offsets_min_max_mhz: `{float(np.min(offsets)):.3f}, {float(np.max(offsets)):.3f}`",
                f"- PSD power sanity (|delta|<=0.5 dB): `{ok}`",
                "",
                "Files:",
                f"- `{out_dir / 'wifi_aci_lut.csv'}`",
                f"- `{out_dir / 'psd_power_sanity.csv'}`",
            ]
        )
    )
    print(f"saved: {out_dir}")
    print(f"lut_rows={len(df)} psd_power_sanity_ok={ok}")


if __name__ == "__main__":
    main()
