from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_wifi_tx_class():
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "Wi-Fi" / "TX_Wi_Fi.py"
    spec = importlib.util.spec_from_file_location("tx_wifi_module_for_grid_check", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.WiFiOFDMTx


def _signed_axis(nfft: int) -> np.ndarray:
    return np.arange(-(nfft // 2), nfft // 2, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Wi-Fi OFDM/OFDMA frequency grid occupancy.")
    ap.add_argument("--bw-mhz", type=int, default=160, choices=[20, 40, 80, 160])
    ap.add_argument("--standard", type=str, default="wifi6e", choices=["wifi6e", "wifi7"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="simulation/mms/results/wifi_ofdma_grid_check")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    WiFiOFDMTx = _load_wifi_tx_class()
    tx = WiFiOFDMTx(rng_seed=int(args.seed), center_freq_hz=6.5e9)
    dbg = tx.build_debug_ofdm_grid(channel_bw_mhz=int(args.bw_mhz), standard=str(args.standard), force_mcs=0)

    X = np.asarray(dbg["grid"], dtype=np.complex128)
    nfft = int(dbg["fft_n"])
    signed = _signed_axis(nfft)
    Xs = np.fft.fftshift(X)

    # Guard checks in signed-k view (fftshift) for stable interpretation.
    occ = np.flatnonzero(np.abs(Xs) > 1e-12).astype(int)
    min_occ = int(np.min(occ))
    max_occ = int(np.max(occ))
    left = Xs[:min_occ]
    right = Xs[max_occ + 1 :]
    left_zero_ratio = float(np.mean(np.abs(left) <= 1e-12)) if left.size > 0 else float("nan")
    right_zero_ratio = float(np.mean(np.abs(right) <= 1e-12)) if right.size > 0 else float("nan")

    summary = {
        "bw_mhz": int(args.bw_mhz),
        "standard": str(args.standard),
        "fft_n": nfft,
        "dc_abs": float(np.abs(X[int(dbg["dc_index"])])),
        "n_occupied_bins": int(occ.size),
        "left_guard_len": int(left.size),
        "right_guard_len": int(right.size),
        "left_guard_zero_ratio": left_zero_ratio,
        "right_guard_zero_ratio": right_zero_ratio,
        "indexing": str(dbg.get("indexing", "")),
    }

    (out_dir / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(
        out_dir / "grid_bins.csv",
        np.column_stack([signed, np.abs(Xs), np.real(Xs), np.imag(Xs)]),
        delimiter=",",
        header="signed_k,absX,realX,imagX",
        comments="",
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.plot(signed, np.abs(Xs), lw=1.2)
    ax.axvline(0, color="black", linestyle="--", lw=0.8, label="DC")
    ax.set_title(f"Wi-Fi OFDM Grid |X[k]| ({args.standard}, {args.bw_mhz} MHz)")
    ax.set_xlabel("Signed subcarrier index k")
    ax.set_ylabel("|X[k]|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_grid_abs.png", dpi=150)
    plt.close(fig)

    report = [
        "# Wi-Fi OFDM Grid Check",
        "",
        f"- output_dir: `{out_dir}`",
        f"- standard: `{args.standard}`",
        f"- bw_mhz: `{args.bw_mhz}`",
        f"- fft_n: `{nfft}`",
        f"- dc_abs: `{summary['dc_abs']:.3e}`",
        f"- occupied_bins: `{summary['n_occupied_bins']}`",
        f"- left_guard_len/zero_ratio: `{summary['left_guard_len']}` / `{summary['left_guard_zero_ratio']:.3f}`",
        f"- right_guard_len/zero_ratio: `{summary['right_guard_len']}` / `{summary['right_guard_zero_ratio']:.3f}`",
        "",
        "Files:",
        f"- `{out_dir / 'grid_summary.json'}`",
        f"- `{out_dir / 'grid_bins.csv'}`",
        f"- `{out_dir / 'plot_grid_abs.png'}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report))
    print(f"saved: {out_dir}")
    print(f"dc_abs={summary['dc_abs']:.3e} left_guard_zero_ratio={left_zero_ratio:.3f} right_guard_zero_ratio={right_zero_ratio:.3f}")


if __name__ == "__main__":
    main()
