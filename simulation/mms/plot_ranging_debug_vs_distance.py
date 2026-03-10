from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MMS ranging debug diagnostics vs distance")
    ap.add_argument("--trial-csv", type=str, required=True)
    ap.add_argument("--aggregate-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    trial = _load_csv(args.trial_csv)
    agg = _load_csv(args.aggregate_csv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.aggregate_csv).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    files: list[str] = []

    # 1) bias/rmse/mae vs distance
    if {"distance_m", "range_bias_m", "range_rmse_m", "range_mae_m"}.issubset(set(agg.columns)):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = agg["distance_m"].to_numpy(dtype=float)
        ax.plot(x, agg["range_bias_m"], marker="o", linewidth=2.0, label="bias")
        ax.plot(x, agg["range_rmse_m"], marker="s", linewidth=2.0, label="RMSE")
        ax.plot(x, agg["range_mae_m"], marker="^", linewidth=2.0, label="MAE")
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title("Range Error Metrics vs Distance")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Error [m]")
        ax.legend()
        fig.tight_layout()
        fn = "plot_range_error_metrics_vs_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 2) first path / peak index vs distance
    if {"distance_m", "first_path_index_mean", "peak_index_mean"}.issubset(set(agg.columns)):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = agg["distance_m"].to_numpy(dtype=float)
        ax.plot(x, agg["first_path_index_mean"], marker="o", linewidth=2.0, label="k_fp mean")
        ax.plot(x, agg["peak_index_mean"], marker="s", linewidth=2.0, label="k_peak mean")
        ax.set_title("First-path / Peak Index vs Distance")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Index [samples]")
        ax.legend()
        fig.tight_layout()
        fn = "plot_fp_peak_index_vs_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 3) bias equivalent ticks
    if {"distance_m", "range_bias_equiv_ticks"}.issubset(set(agg.columns)):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = agg["distance_m"].to_numpy(dtype=float)
        ax.plot(x, agg["range_bias_equiv_ticks"], marker="o", linewidth=2.0, label="bias [ticks]")
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title("Range Bias Equivalent ToF Ticks vs Distance")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Ticks")
        ax.legend()
        fig.tight_layout()
        fn = "plot_bias_ticks_vs_distance.png"
        fig.savefig(out_dir / fn, dpi=170)
        plt.close(fig)
        files.append(fn)

    # 4) noise window margin distribution
    if {"distance_m", "noise_win_end", "first_path_index"}.issubset(set(trial.columns)):
        td = trial.copy()
        td["noise_margin"] = pd.to_numeric(td["noise_win_end"], errors="coerce") - pd.to_numeric(td["first_path_index"], errors="coerce")
        td = td[np.isfinite(td["noise_margin"])]
        if not td.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            dvals = sorted(td["distance_m"].unique())
            data = [td[td["distance_m"] == d]["noise_margin"].to_numpy(dtype=float) for d in dvals]
            ax.boxplot(data, tick_labels=[f"{d:g}" for d in dvals], showfliers=False)
            ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
            ax.set_title("Noise Window End - First Path Index")
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("noise_win_end - k_fp [samples]")
            fig.tight_layout()
            fn = "plot_noise_window_margin_vs_distance.png"
            fig.savefig(out_dir / fn, dpi=170)
            plt.close(fig)
            files.append(fn)

    print(f"saved plots to: {out_dir}")
    for fn in files:
        print(f"- {fn}")


if __name__ == "__main__":
    main()
