from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from simulation.mms.wifi_spatial_model import WiFiACILUT
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.wifi_spatial_model import WiFiACILUT


def _frac_overlap(fc_a_hz: float, bw_a_hz: float, fc_b_hz: float, bw_b_hz: float) -> float:
    a0 = float(fc_a_hz) - 0.5 * float(bw_a_hz)
    a1 = float(fc_a_hz) + 0.5 * float(bw_a_hz)
    b0 = float(fc_b_hz) - 0.5 * float(bw_b_hz)
    b1 = float(fc_b_hz) + 0.5 * float(bw_b_hz)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return float(inter / max(float(bw_a_hz), 1.0))


def _load_wifi_tx_class():
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("tx_wifi_module_for_aci_check", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.WiFiOFDMTx


def _parse_offsets(s: str) -> list[float]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError("offset list is empty")
    return out


def _integrate_band_power(
    f_mhz: np.ndarray,
    psd_w_hz: np.ndarray,
    f0_mhz: float,
    bw_mhz: float,
) -> float:
    lo = float(f0_mhz) - 0.5 * float(bw_mhz)
    hi = float(f0_mhz) + 0.5 * float(bw_mhz)
    m = (f_mhz >= lo) & (f_mhz <= hi)
    if int(np.sum(m)) < 2:
        return 0.0
    return float(np.trapezoid(psd_w_hz[m], f_mhz[m] * 1e6))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate ACI coupling model vs waveform PSD integration.")
    ap.add_argument("--wifi-bw-mhz", type=float, default=160.0)
    ap.add_argument("--nb-bw-mhz", type=float, default=2.0)
    ap.add_argument("--offsets-mhz", type=str, required=True)
    ap.add_argument("--standard", type=str, default="wifi7", choices=["wifi6e", "wifi7"])
    ap.add_argument("--duration-s", type=float, default=0.02)
    ap.add_argument("--aclr-db", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", type=str, default="overlap", choices=["overlap", "lut"])
    ap.add_argument("--lut-path", type=str, default=None)
    ap.add_argument("--aci-lut-path", type=str, default=None)
    ap.add_argument("--psd-floor-rel", type=float, default=1e-15)
    ap.add_argument("--out", type=str, default="simulation/mms/results/wifi_aci_mask_check")
    args = ap.parse_args()

    offsets = _parse_offsets(args.offsets_mhz)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    WiFiOFDMTx = _load_wifi_tx_class()
    tx = WiFiOFDMTx(rng_seed=int(args.seed), center_freq_hz=6.5e9)
    wf, info = tx.generate_for_target_rx_throughput(
        target_rx_throughput_mbps=800.0,
        duration_s=float(args.duration_s),
        channel_bw_mhz=int(round(float(args.wifi_bw_mhz))),
        standard=str(args.standard),
        tx_power_dbw=-20.0,
        center_freq_hz=6.5e9,
    )
    wf = tx.apply_tx_emission_mask(
        wf,
        fs_hz=float(info["sample_rate_hz"]),
        channel_bw_hz=float(args.wifi_bw_mhz) * 1e6,
        aclr_db=float(args.aclr_db),
        seed=int(args.seed) + 17,
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
    psd_total_delta_db = float(10.0 * np.log10((p_psd_total + 1e-30) / (p_total + 1e-30)))
    psd_power_sanity_ok = bool(abs(psd_total_delta_db) <= 0.5)

    lut_path = args.lut_path if args.lut_path else args.aci_lut_path
    lut = None
    if str(args.model).lower() == "lut":
        if not lut_path:
            raise ValueError("--model lut requires --lut-path (or --aci-lut-path)")
        lut = WiFiACILUT.from_csv(lut_path)

    rows: list[dict] = []
    for off in offsets:
        if str(args.model).lower() == "lut":
            frac = float(
                lut.coupling_linear(
                    wifi_bw_mhz=float(args.wifi_bw_mhz),
                    offset_mhz=float(off),
                )
            )
        else:
            frac = float(
                _frac_overlap(
                    fc_a_hz=0.0,
                    bw_a_hz=float(args.wifi_bw_mhz) * 1e6,
                    fc_b_hz=float(off) * 1e6,
                    bw_b_hz=float(args.nb_bw_mhz) * 1e6,
                )
            )
        p_model = p_total * max(frac, 0.0)
        p_psd_raw = _integrate_band_power(f_mhz, psd_w_hz, f0_mhz=float(off), bw_mhz=float(args.nb_bw_mhz))
        p_psd_floor = float(p_total * max(float(args.psd_floor_rel), 0.0))
        p_psd = float(max(p_psd_raw, p_psd_floor))
        model_dbm = 10.0 * np.log10(p_model + 1e-30) + 30.0
        psd_dbm = 10.0 * np.log10(p_psd + 1e-30) + 30.0
        psd_raw_dbm = 10.0 * np.log10(p_psd_raw + 1e-30) + 30.0
        rows.append(
            {
                "offset_mhz": float(off),
                "coupling_frac_model": float(frac),
                "model_type": str(args.model),
                "I_model_dbm": float(model_dbm),
                "I_psd_dbm": float(psd_dbm),
                "I_psd_raw_dbm": float(psd_raw_dbm),
                "delta_db_model_minus_psd": float(model_dbm - psd_dbm),
            }
        )

    df = pd.DataFrame(rows).sort_values("offset_mhz").reset_index(drop=True)
    df.to_csv(out_dir / "aci_validation.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(df["offset_mhz"], df["I_model_dbm"], marker="o", label="Model (rect overlap)")
    ax.plot(df["offset_mhz"], df["I_psd_dbm"], marker="x", label="Waveform PSD integral")
    ax.set_xlabel("Victim center offset [MHz]")
    ax.set_ylabel("Interference power in victim band [dBm]")
    ax.set_title("ACI: model vs waveform PSD")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_aci_model_vs_psd.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.0))
    ax.plot(df["offset_mhz"], df["delta_db_model_minus_psd"], marker="o")
    ax.axhline(0.0, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("Victim center offset [MHz]")
    ax.set_ylabel("Delta [dB] (model - PSD)")
    ax.set_title("ACI model mismatch")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_aci_delta.png", dpi=150)
    plt.close(fig)

    close_mask = df["coupling_frac_model"] > 0.0
    mae_close = float(np.mean(np.abs(df.loc[close_mask, "delta_db_model_minus_psd"]))) if close_mask.any() else float("nan")
    mae_all = float(np.mean(np.abs(df["delta_db_model_minus_psd"])))
    pass_close = bool(mae_close <= 2.0) if np.isfinite(mae_close) else False
    pass_all = bool(mae_all <= 3.0)
    pass_overall = bool(pass_close and pass_all and psd_power_sanity_ok)

    report = [
        "# Wi-Fi ACI Model Validation",
        "",
        f"- output_dir: `{out_dir}`",
        f"- wifi_bw_mhz: `{args.wifi_bw_mhz}`",
        f"- nb_bw_mhz: `{args.nb_bw_mhz}`",
        f"- standard: `{args.standard}`",
        f"- aclr_db: `{args.aclr_db}`",
        f"- model: `{args.model}`",
        f"- lut_path: `{args.lut_path}`",
        f"- psd_floor_rel: `{args.psd_floor_rel}`",
        "",
        "Model compared:",
        "- `overlap`: rectangular overlap fraction (legacy).",
        "- `lut`: PSD-derived LUT interpolation from `build_wifi_aci_lut.py` output.",
        "- Waveform truth uses `Wi-Fi/TX_Wi_Fi.py` OFDM waveform + Welch PSD integration over victim band.",
        "",
        "PSD absolute-power sanity:",
        f"- p_total(time): `{p_total:.6e}` W",
        f"- p_total(PSD integral): `{p_psd_total:.6e}` W",
        f"- delta_db: `{psd_total_delta_db:.3f}` dB",
        f"- sanity PASS(|delta|<=0.5 dB): `{psd_power_sanity_ok}`",
        "",
        f"- MAE(all offsets): `{mae_all:.2f} dB`",
        f"- MAE(overlap-only offsets): `{mae_close:.2f} dB`",
        f"- PASS(overlap-only, <=2 dB): `{pass_close}`",
        f"- PASS(all, <=3 dB): `{pass_all}`",
        f"- OVERALL PASS: `{pass_overall}`",
        "",
        "Files:",
        f"- `{out_dir / 'aci_validation.csv'}`",
        f"- `{out_dir / 'plot_aci_model_vs_psd.png'}`",
        f"- `{out_dir / 'plot_aci_delta.png'}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"saved: {out_dir}")
    print(
        " ".join(
            [
                f"mae_all={mae_all:.2f}dB",
                f"mae_overlap={mae_close:.2f}dB",
                f"psd_sanity_db={psd_total_delta_db:.3f}",
                f"pass={pass_overall}",
            ]
        )
    )


if __name__ == "__main__":
    main()
