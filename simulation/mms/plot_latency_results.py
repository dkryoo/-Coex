from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


def _p95_valid(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 95))


def _load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No input CSV files")
    df = pd.concat(frames, ignore_index=True)
    return df


def _save_plot(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def _plot_core(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    # 1) success rate vs distance (off vs dense)
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in sorted(df["wifi_mode"].dropna().unique()):
        dfg = df[df["wifi_mode"] == mode]
        grp = dfg.groupby("distance_m", dropna=False)["success"].mean().reset_index()
        ax.plot(grp["distance_m"], grp["success"], marker="o", label=str(mode))
    ax.set_title("Success Rate vs Distance")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Success Rate")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    _save_plot(fig, out_dir / "success_vs_distance.png")
    plt.close(fig)

    # 2) latency median/p95 vs distance (success only)
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in sorted(df["wifi_mode"].dropna().unique()):
        dfg = df[(df["wifi_mode"] == mode) & (df["success"] == True) & (df["latency_to_success_ms"].notna())]
        if dfg.empty:
            continue
        g = dfg.groupby("distance_m")["latency_to_success_ms"]
        med = g.median()
        p95 = g.quantile(0.95)
        x = med.index.to_numpy(dtype=float)
        ax.plot(x, med.values, marker="o", label=f"{mode} median")
        ax.plot(x, p95.values, marker="x", linestyle="--", label=f"{mode} p95")
    ax.set_title("Latency (Success) vs Distance")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    _save_plot(fig, out_dir / "latency_vs_distance.png")
    plt.close(fig)

    # 3) offset impact at fixed distance/ch if available
    if "wifi_offset_mhz_req" in df.columns:
        f = df.copy()
        f = f[f["wifi_offset_mhz_req"].notna()]
        if not f.empty:
            fig, ax1 = plt.subplots(figsize=(7, 4))
            gsucc = f.groupby("wifi_offset_mhz_req")["success"].mean()
            ax1.plot(gsucc.index, gsucc.values, marker="o", color="tab:blue", label="success_rate")
            ax1.set_xlabel("Wi-Fi offset [MHz]")
            ax1.set_ylabel("Success Rate", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2 = ax1.twinx()
            glat = f[(f["success"] == True) & (f["latency_to_success_ms"].notna())].groupby("wifi_offset_mhz_req")["latency_to_success_ms"].median()
            ax2.plot(glat.index, glat.values, marker="x", color="tab:red", label="median_latency")
            ax2.set_ylabel("Median Latency [ms]", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Offset Impact: Success and Latency")
            _save_plot(fig, out_dir / "offset_impact.png")
            plt.close(fig)

    # 4) uwb channel comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby("uwb_channel")["success"].mean().reset_index()
    ax.bar(g["uwb_channel"].astype(str), g["success"])
    ax.set_title("Success Rate by UWB Channel")
    ax.set_xlabel("UWB Channel")
    ax.set_ylabel("Success Rate")
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, out_dir / "success_by_uwb_channel.png")
    plt.close(fig)

    # 5) fail reason stacked bar by wifi_mode
    if "fail_reason" in df.columns:
        fr = df[df["success"] == False].copy()
        if not fr.empty:
            tab = pd.crosstab(fr["wifi_mode"], fr["fail_reason"], normalize="index")
            fig, ax = plt.subplots(figsize=(8, 4))
            tab.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Fail Reason Distribution")
            ax.set_ylabel("Fraction")
            ax.grid(True, axis="y", alpha=0.3)
            _save_plot(fig, out_dir / "fail_reason_stacked.png")
            plt.close(fig)

    # 6) success rate vs wifi_density (line per UWB channel)
    if "wifi_density" in df.columns and "uwb_channel" in df.columns:
        dfd = df[df["wifi_mode"] == "dense"].copy()
        if dfd["wifi_density"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for ch in sorted(dfd["uwb_channel"].dropna().unique()):
                g = dfd[dfd["uwb_channel"] == ch].groupby("wifi_density")["success"].mean().reset_index()
                ax.plot(g["wifi_density"], g["success"], marker="o", label=f"UWB ch {int(ch)}")
            ax.set_title("Success Rate vs Wi-Fi Density")
            ax.set_xlabel("Wi-Fi Density / Load")
            ax.set_ylabel("Success Rate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            _save_plot(fig, out_dir / "success_vs_wifi_density_by_uwb_ch.png")
            plt.close(fig)

            # 7) latency median/p95 vs wifi_density (success-only)
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ok = dfd[(dfd["success"] == True) & (dfd["latency_to_success_ms"].notna())]
            if not ok.empty:
                med = ok.groupby("wifi_density")["latency_to_success_ms"].median().reset_index()
                p95 = ok.groupby("wifi_density")["latency_to_success_ms"].quantile(0.95).reset_index()
                ax.plot(med["wifi_density"], med["latency_to_success_ms"], marker="o", label="median")
                ax.plot(p95["wifi_density"], p95["latency_to_success_ms"], marker="x", linestyle="--", label="p95")
                ax.legend()
            ax.set_title("Latency vs Wi-Fi Density (success-only)")
            ax.set_xlabel("Wi-Fi Density / Load")
            ax.set_ylabel("Latency [ms]")
            ax.grid(True, alpha=0.3)
            _save_plot(fig, out_dir / "latency_vs_wifi_density_success_only.png")
            plt.close(fig)

    # 9) switching OFF vs ON comparison against wifi density
    if "enable_nb_channel_switching" in df.columns and "wifi_density" in df.columns:
        dfx = df[df["wifi_mode"] == "dense"].copy()
        if not dfx.empty and dfx["wifi_density"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for sw in sorted(dfx["enable_nb_channel_switching"].dropna().unique()):
                g = dfx[dfx["enable_nb_channel_switching"] == sw].groupby("wifi_density")["success"].mean().reset_index()
                ax.plot(g["wifi_density"], g["success"], marker="o", label=f"switching={'ON' if int(sw)==1 else 'OFF'}")
            ax.set_title("Success Rate vs Wi-Fi Density (NB switching OFF/ON)")
            ax.set_xlabel("Wi-Fi Density / Load")
            ax.set_ylabel("Success Rate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            _save_plot(fig, out_dir / "success_vs_wifi_density_switching_off_on.png")
            plt.close(fig)

            ok = dfx[(dfx["success"] == True) & (dfx["latency_to_success_ms"].notna())]
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for sw in sorted(dfx["enable_nb_channel_switching"].dropna().unique()):
                gmed = ok[ok["enable_nb_channel_switching"] == sw].groupby("wifi_density")["latency_to_success_ms"].median().reset_index()
                gp95 = ok[ok["enable_nb_channel_switching"] == sw].groupby("wifi_density")["latency_to_success_ms"].quantile(0.95).reset_index()
                if not gmed.empty:
                    ax.plot(
                        gmed["wifi_density"],
                        gmed["latency_to_success_ms"],
                        marker="o",
                        label=f"{'ON' if int(sw)==1 else 'OFF'} median",
                    )
                if not gp95.empty:
                    ax.plot(
                        gp95["wifi_density"],
                        gp95["latency_to_success_ms"],
                        marker="x",
                        linestyle="--",
                        label=f"{'ON' if int(sw)==1 else 'OFF'} p95",
                    )
            ax.set_title("Latency vs Wi-Fi Density (NB switching OFF/ON, success-only)")
            ax.set_xlabel("Wi-Fi Density / Load")
            ax.set_ylabel("Latency [ms]")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
            _save_plot(fig, out_dir / "latency_vs_wifi_density_switching_off_on.png")
            plt.close(fig)

    # 8) success rate vs wifi_offset_mhz (line per UWB channel)
    if "wifi_offset_mhz_req" in df.columns and "uwb_channel" in df.columns:
        dfo = df[(df["wifi_mode"] == "dense") & (df["wifi_offset_mhz_req"].notna())].copy()
        if dfo["wifi_offset_mhz_req"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for ch in sorted(dfo["uwb_channel"].dropna().unique()):
                g = dfo[dfo["uwb_channel"] == ch].groupby("wifi_offset_mhz_req")["success"].mean().reset_index()
                if not g.empty:
                    ax.plot(g["wifi_offset_mhz_req"], g["success"], marker="o", label=f"UWB ch {int(ch)}")
            ax.set_title("Success Rate vs Wi-Fi Offset")
            ax.set_xlabel("Wi-Fi Offset [MHz]")
            ax.set_ylabel("Success Rate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            _save_plot(fig, out_dir / "success_vs_wifi_offset_by_uwb_ch.png")
            plt.close(fig)

    # 10) switching OFF/ON comparison vs wifi offset
    if "enable_nb_channel_switching" in df.columns and "wifi_offset_mhz_req" in df.columns:
        dfo2 = df[(df["wifi_mode"] == "dense") & (df["wifi_offset_mhz_req"].notna())].copy()
        if not dfo2.empty and dfo2["wifi_offset_mhz_req"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for sw in sorted(dfo2["enable_nb_channel_switching"].dropna().unique()):
                g = dfo2[dfo2["enable_nb_channel_switching"] == sw].groupby("wifi_offset_mhz_req")["success"].mean().reset_index()
                if not g.empty:
                    ax.plot(g["wifi_offset_mhz_req"], g["success"], marker="o", label=f"switching={'ON' if int(sw)==1 else 'OFF'}")
            ax.set_title("Success Rate vs Wi-Fi Offset (NB switching OFF/ON)")
            ax.set_xlabel("Wi-Fi Offset [MHz]")
            ax.set_ylabel("Success Rate")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
            _save_plot(fig, out_dir / "success_vs_wifi_offset_switching_off_on.png")
            plt.close(fig)

            ok = dfo2[(dfo2["success"] == True) & (dfo2["latency_to_success_ms"].notna())]
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for sw in sorted(dfo2["enable_nb_channel_switching"].dropna().unique()):
                gmed = ok[ok["enable_nb_channel_switching"] == sw].groupby("wifi_offset_mhz_req")["latency_to_success_ms"].median().reset_index()
                gp95 = ok[ok["enable_nb_channel_switching"] == sw].groupby("wifi_offset_mhz_req")["latency_to_success_ms"].quantile(0.95).reset_index()
                if not gmed.empty:
                    ax.plot(gmed["wifi_offset_mhz_req"], gmed["latency_to_success_ms"], marker="o", label=f"{'ON' if int(sw)==1 else 'OFF'} median")
                if not gp95.empty:
                    ax.plot(gp95["wifi_offset_mhz_req"], gp95["latency_to_success_ms"], marker="x", linestyle="--", label=f"{'ON' if int(sw)==1 else 'OFF'} p95")
            ax.set_title("Latency vs Wi-Fi Offset (NB switching OFF/ON, success-only)")
            ax.set_xlabel("Wi-Fi Offset [MHz]")
            ax.set_ylabel("Latency [ms]")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
            _save_plot(fig, out_dir / "latency_vs_wifi_offset_switching_off_on.png")
            plt.close(fig)

    # 11) allow-list impact vs density (switching enabled runs)
    if "mmsNbChannelAllowList_json" in df.columns and "wifi_density" in df.columns:
        dfa = df[(df["wifi_mode"] == "dense") & (df["wifi_density"].notna())].copy()
        if not dfa.empty and dfa["wifi_density"].nunique() >= 2 and dfa["mmsNbChannelAllowList_json"].nunique() >= 2:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for al in sorted(dfa["mmsNbChannelAllowList_json"].dropna().unique()):
                g = dfa[dfa["mmsNbChannelAllowList_json"] == al].groupby("wifi_density")["success"].mean().reset_index()
                if g.empty:
                    continue
                lbl = str(al)
                if len(lbl) > 24:
                    lbl = lbl[:24] + "..."
                ax.plot(g["wifi_density"], g["success"], marker="o", label=lbl)
            ax.set_title("Success Rate vs Wi-Fi Density (allow-list cases)")
            ax.set_xlabel("Wi-Fi Density / Load")
            ax.set_ylabel("Success Rate")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            _save_plot(fig, out_dir / "success_vs_wifi_density_allow_list_cases.png")
            plt.close(fig)


def _dump_timelines(df: pd.DataFrame, out_dir: Path, n: int, seed: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    cand = df[df["event_trace_json"].notna()] if "event_trace_json" in df.columns else pd.DataFrame()
    if cand.empty:
        return
    rng = random.Random(seed)
    idxs = list(cand.index)
    rng.shuffle(idxs)
    idxs = idxs[: max(0, int(n))]
    for k, idx in enumerate(idxs):
        row = cand.loc[idx]
        trace = row.get("event_trace_json", "[]")
        try:
            ev = json.loads(trace)
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(9, 2.8))
        y = 0
        cmap = {
            "nb_lbt_wait": "tab:orange",
            "nb_lbt_wait_spatial": "tab:brown",
            "nb_ssbd_deferral": "tab:brown",
            "nb_control_tx": "tab:blue",
            "wifi_cca_wait": "tab:purple",
            "uwb_gap": "tab:gray",
            "uwb_shot": "tab:green",
        }
        for e in ev:
            s = float(e.get("start_ms", 0.0))
            e_ms = float(e.get("end_ms", s))
            lbl = str(e.get("label", "evt"))
            ax.broken_barh([(s, max(0.0, e_ms - s))], (y, 8), facecolors=cmap.get(lbl, "tab:red"))
        ax.set_title(f"Trial timeline idx={idx}, success={row.get('success')}")
        ax.set_xlabel("Time [ms]")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        out = out_dir / f"timeline_{k:03d}.png"
        _save_plot(fig, out)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot MMS latency result CSVs.")
    p.add_argument("--inputs", type=str, nargs="+", required=True, help="Input trial-level CSV files.")
    p.add_argument("--out-dir", type=str, default="simulation/mms/results/plots")
    p.add_argument("--dump-timeline", type=int, default=0, help="Number of random trial timelines to dump.")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_csvs(args.inputs)
    # Save summary CSV
    summ = (
        df.groupby(["wifi_mode", "distance_m", "uwb_channel", "wifi_offset_mhz_req"], dropna=False)
        .agg(
            success_rate=("success", "mean"),
            med_latency_success_ms=("latency_to_success_ms", "median"),
            p95_latency_success_ms=("latency_to_success_ms", _p95_valid),
            med_fail_time_ms=("time_spent_ms", "median"),
            n=("success", "count"),
        )
        .reset_index()
    )
    summ.to_csv(out_dir / "summary.csv", index=False)

    _plot_core(df, out_dir)
    if args.dump_timeline > 0:
        _dump_timelines(df, out_dir, n=args.dump_timeline, seed=args.seed)
    print(f"saved plots/summary to {out_dir}")


if __name__ == "__main__":
    main()
