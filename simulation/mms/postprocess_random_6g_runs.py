from __future__ import annotations

"""
Post-process multiple random 6GHz duty-sweep runs.

Example:
python simulation/mms/postprocess_random_6g_runs.py \
  --runs \
    simulation/mms/results/random_bwpos_160heavy \
    simulation/mms/results/random_bwpos_balanced \
    simulation/mms/results/random_bwpos_20heavy \
  --labels "160-heavy,balanced,20-heavy" \
  --pick latest \
  --out-dir simulation/mms/results/postprocess
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BW_KEYS = [20, 40, 80, 160]


@dataclass
class CaseData:
    label: str
    run_dirs: list[Path]
    trial_df: pd.DataFrame
    agg_df: pd.DataFrame


def _parse_csv_list(v: str | None) -> list[str]:
    if v is None:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _parse_int_csv(v: str | None) -> list[int]:
    out = []
    for x in _parse_csv_list(v):
        out.append(int(x))
    return out


def _as_bool(v: str | int | bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {v}")


def _is_run_dir(p: Path) -> bool:
    return (p / "trial_results.csv").exists() and (p / "aggregate_results.csv").exists()


def _resolve_run_dirs(inp: Path, pick: str) -> list[Path]:
    if _is_run_dir(inp):
        return [inp]
    if not inp.exists() or not inp.is_dir():
        return []
    children = [c for c in inp.iterdir() if c.is_dir() and _is_run_dir(c)]
    if not children:
        return []
    children = sorted(children, key=lambda p: p.stat().st_mtime)
    if pick == "latest":
        return [children[-1]]
    return children


def _json_load(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_bw_hist(obj: Any) -> dict[int, int]:
    out = {20: 0, 40: 0, 80: 0, 160: 0}
    d = _json_load(obj)
    if isinstance(d, dict):
        if "bw_hist" in d and isinstance(d["bw_hist"], dict):
            d = d["bw_hist"]
        for k, v in d.items():
            try:
                kk = int(k)
                if kk in out:
                    out[kk] = int(v)
            except Exception:
                continue
    return out


def _derive_trial_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "n_ap" not in df.columns:
        vals = []
        for _, r in df.iterrows():
            n = np.nan
            a = _json_load(r.get("occ_counts_json", None))
            if isinstance(a, dict) and "n_ap" in a:
                n = float(a["n_ap"])
            if not np.isfinite(n):
                b = _json_load(r.get("wifi_ap_summary_json", None))
                if isinstance(b, dict) and "n_ap" in b:
                    n = float(b["n_ap"])
            vals.append(n)
        df["n_ap"] = vals

    if "partition_bw_hist_json" in df.columns:
        src = df["partition_bw_hist_json"]
    else:
        src = df.get("band_partition_summary_json", pd.Series([None] * len(df)))

    bw_20, bw_40, bw_80, bw_160 = [], [], [], []
    for x in src:
        h = _normalize_bw_hist(x)
        bw_20.append(h[20])
        bw_40.append(h[40])
        bw_80.append(h[80])
        bw_160.append(h[160])
    df["bw20_cnt"] = bw_20
    df["bw40_cnt"] = bw_40
    df["bw80_cnt"] = bw_80
    df["bw160_cnt"] = bw_160

    if "occ_bursty_frac" not in df.columns:
        vals = []
        for _, r in df.iterrows():
            frac = np.nan
            d = _json_load(r.get("occ_counts_json", None))
            if isinstance(d, dict):
                u = float(d.get("uniform", np.nan))
                b = float(d.get("bursty", np.nan))
                if np.isfinite(u) and np.isfinite(b) and (u + b) > 0:
                    frac = b / (u + b)
            vals.append(frac)
        df["occ_bursty_frac"] = vals

    uniq = []
    for _, r in df.iterrows():
        seq = _json_load(r.get("nb_channel_seq_first8_json", None))
        if isinstance(seq, list):
            uniq.append(int(len(set(seq))))
        else:
            uniq.append(np.nan)
    df["nb_seq_unique_count"] = uniq

    if "partition_ok" not in df.columns:
        df["partition_ok"] = np.nan
    if "coverage_ratio" not in df.columns:
        df["coverage_ratio"] = np.nan

    return df


def _load_case(label: str, run_dirs: list[Path]) -> CaseData:
    tdfs = []
    adfs = []
    for rd in run_dirs:
        t = pd.read_csv(rd / "trial_results.csv")
        a = pd.read_csv(rd / "aggregate_results.csv")
        t["case_label"] = label
        t["run_dir"] = str(rd)
        a["case_label"] = label
        a["run_dir"] = str(rd)
        tdfs.append(t)
        adfs.append(a)
    trial = _derive_trial_fields(pd.concat(tdfs, ignore_index=True)) if tdfs else pd.DataFrame()
    agg = pd.concat(adfs, ignore_index=True) if adfs else pd.DataFrame()
    return CaseData(label=label, run_dirs=run_dirs, trial_df=trial, agg_df=agg)


def _sanity_checks(cases: list[CaseData], strict: bool) -> tuple[list[str], list[str]]:
    msgs: list[str] = []
    errs: list[str] = []

    for c in cases:
        d = c.trial_df
        if d.empty:
            errs.append(f"[{c.label}] empty trial data")
            continue

        if "partition_ok" in d.columns and d["partition_ok"].notna().any():
            ok_rate = float(np.mean(d["partition_ok"].astype(bool)))
            msgs.append(f"[{c.label}] partition_ok_rate={ok_rate:.4f}")
            if strict and ok_rate < 1.0:
                errs.append(f"[{c.label}] partition_ok_rate < 1.0")
        else:
            msgs.append(f"[{c.label}] partition_ok column missing")
            if strict:
                errs.append(f"[{c.label}] partition_ok column missing")

        if "coverage_ratio" in d.columns and d["coverage_ratio"].notna().any():
            cmin = float(np.nanmin(pd.to_numeric(d["coverage_ratio"], errors="coerce")))
            cmax = float(np.nanmax(pd.to_numeric(d["coverage_ratio"], errors="coerce")))
            msgs.append(f"[{c.label}] coverage_ratio min/max={cmin:.6f}/{cmax:.6f}")
            if strict and (abs(cmin - 1.0) > 1e-9 or abs(cmax - 1.0) > 1e-9):
                errs.append(f"[{c.label}] coverage_ratio not exactly 1.0")
        else:
            msgs.append(f"[{c.label}] coverage_ratio column missing")
            if strict:
                errs.append(f"[{c.label}] coverage_ratio column missing")

        if "occupancy_mode" in d.columns and "occ_bursty_frac" in d.columns:
            for mode, g in d.groupby("occupancy_mode", dropna=False):
                frac = pd.to_numeric(g["occ_bursty_frac"], errors="coerce")
                if not frac.notna().any():
                    continue
                mn = float(np.nanmean(frac))
                msgs.append(f"[{c.label}] occupancy={mode} bursty_frac_mean={mn:.4f}")
                if strict:
                    sm = str(mode).lower()
                    if sm == "uniform" and float(np.nanmax(np.abs(frac))) > 1e-9:
                        errs.append(f"[{c.label}] uniform mode has bursty_frac != 0")
                    if sm == "bursty" and float(np.nanmin(np.abs(frac - 1.0))) > 1e-9:
                        errs.append(f"[{c.label}] bursty mode has bursty_frac != 1")
        else:
            msgs.append(f"[{c.label}] occupancy sanity columns missing")

        if "hopping" in d.columns and "nb_seq_unique_count" in d.columns:
            goff = d[d["hopping"].astype(str).str.lower() == "off"]
            gon = d[d["hopping"].astype(str).str.lower() == "on"]
            if len(goff):
                off_ratio = float(np.mean(pd.to_numeric(goff["nb_seq_unique_count"], errors="coerce") == 1))
                msgs.append(f"[{c.label}] hopping off unique==1 ratio={off_ratio:.4f}")
            if len(gon):
                on_ratio = float(np.mean(pd.to_numeric(gon["nb_seq_unique_count"], errors="coerce") > 1))
                msgs.append(f"[{c.label}] hopping on unique>1 ratio={on_ratio:.4f}")
    return msgs, errs


def _plot_distributions(trial_all: pd.DataFrame, out_dir: Path) -> list[str]:
    made: list[str] = []
    if trial_all.empty:
        return made

    if "n_ap" in trial_all.columns and pd.to_numeric(trial_all["n_ap"], errors="coerce").notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, g in trial_all.groupby("case_label", dropna=False):
            vals = pd.to_numeric(g["n_ap"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                ax.hist(vals, bins=min(30, max(5, int(np.sqrt(vals.size)))), alpha=0.4, label=str(label))
        ax.set_title("n_ap distribution")
        ax.set_xlabel("n_ap")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fn = "plot_dist_n_ap.png"
        fig.savefig(out_dir / fn, dpi=160)
        plt.close(fig)
        made.append(fn)

    if "occ_bursty_frac" in trial_all.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, g in trial_all.groupby("case_label", dropna=False):
            vals = pd.to_numeric(g["occ_bursty_frac"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                ax.hist(vals, bins=20, alpha=0.4, label=str(label), range=(0.0, 1.0))
        ax.set_title("bursty_frac distribution")
        ax.set_xlabel("bursty_frac")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fn = "plot_dist_bursty_frac.png"
        fig.savefig(out_dir / fn, dpi=160)
        plt.close(fig)
        made.append(fn)

    bw_cols = ["bw20_cnt", "bw40_cnt", "bw80_cnt", "bw160_cnt"]
    if all(c in trial_all.columns for c in bw_cols):
        labels = []
        means = {20: [], 40: [], 80: [], 160: []}
        stds = {20: [], 40: [], 80: [], 160: []}
        for label, g in trial_all.groupby("case_label", dropna=False):
            labels.append(str(label))
            for bw, col in zip(BW_KEYS, bw_cols):
                x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
                means[bw].append(float(np.mean(x)) if x.size else np.nan)
                stds[bw].append(float(np.std(x)) if x.size else np.nan)
        x = np.arange(len(labels))
        w = 0.18
        fig, ax = plt.subplots(figsize=(11, 5))
        offs = {20: -1.5 * w, 40: -0.5 * w, 80: 0.5 * w, 160: 1.5 * w}
        for bw in BW_KEYS:
            ax.bar(x + offs[bw], means[bw], width=w, yerr=stds[bw], capsize=3, label=f"{bw}MHz seg")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title("BW segment count mean/std by case")
        ax.set_ylabel("segment count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fn = "plot_bw_hist_mean_std.png"
        fig.savefig(out_dir / fn, dpi=160)
        plt.close(fig)
        made.append(fn)

    if "nb_seq_unique_count" in trial_all.columns and "hopping" in trial_all.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        idx = np.arange(len(trial_all["case_label"].unique()))
        labels = sorted(trial_all["case_label"].astype(str).unique())
        off_mean = []
        on_mean = []
        for lb in labels:
            g = trial_all[trial_all["case_label"].astype(str) == lb]
            goff = pd.to_numeric(g[g["hopping"].astype(str).str.lower() == "off"]["nb_seq_unique_count"], errors="coerce")
            gon = pd.to_numeric(g[g["hopping"].astype(str).str.lower() == "on"]["nb_seq_unique_count"], errors="coerce")
            off_mean.append(float(np.nanmean(goff)) if goff.notna().any() else np.nan)
            on_mean.append(float(np.nanmean(gon)) if gon.notna().any() else np.nan)
        w = 0.35
        ax.bar(np.arange(len(labels)) - w / 2, off_mean, width=w, label="hop off")
        ax.bar(np.arange(len(labels)) + w / 2, on_mean, width=w, label="hop on")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title("NB seq unique-count evidence")
        ax.set_ylabel("mean unique count of first8 sequence")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fn = "plot_hopping_seq_unique.png"
        fig.savefig(out_dir / fn, dpi=160)
        plt.close(fig)
        made.append(fn)

    return made


def _pick_latency_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _plot_overlays(
    agg_all: pd.DataFrame,
    out_dir: Path,
    *,
    filter_occupancy: list[str],
    filter_hopping: list[str],
    filter_max_attempts: list[int],
) -> list[str]:
    made: list[str] = []
    if agg_all.empty:
        return made

    work = agg_all.copy()
    if filter_occupancy:
        work = work[work["occupancy_mode"].astype(str).isin(filter_occupancy)]
    if filter_hopping:
        work = work[work["hopping"].astype(str).isin(filter_hopping)]
    if filter_max_attempts:
        work = work[pd.to_numeric(work["max_attempts_cfg"], errors="coerce").isin(filter_max_attempts)]

    if work.empty:
        return made

    gkeys = ["occupancy_mode", "hopping", "max_attempts_cfg"]
    for (occ, hop, ma), g3 in work.groupby(gkeys, dropna=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        has_line = False
        for label, g in g3.groupby("case_label", dropna=False):
            gg = g.sort_values("duty_cycle")
            x = pd.to_numeric(gg["duty_cycle"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(gg.get("P_overall"), errors="coerce").to_numpy(dtype=float)
            if x.size == 0:
                continue
            ax.plot(x, y, marker="o", label=str(label))
            lo = gg.get("P_overall_wilson_lo")
            hi = gg.get("P_overall_wilson_hi")
            if lo is not None and hi is not None:
                lo_v = pd.to_numeric(lo, errors="coerce").to_numpy(dtype=float)
                hi_v = pd.to_numeric(hi, errors="coerce").to_numpy(dtype=float)
                if np.isfinite(lo_v).any() and np.isfinite(hi_v).any():
                    ax.fill_between(x, lo_v, hi_v, alpha=0.15)
            has_line = True
        if has_line:
            ax.set_title(f"P_overall vs duty (occ={occ}, hop={hop}, m={int(ma)})")
            ax.set_xlabel("duty_cycle")
            ax.set_ylabel("P_overall")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fn = f"plot_overlay_p_overall_occ={occ}_hop={hop}_m={int(ma)}.png"
            fig.savefig(out_dir / fn, dpi=160)
            made.append(fn)
        plt.close(fig)

    succ_med_col = _pick_latency_col(agg_all, ["latency_to_success_median_ms", "latency_to_success_mean_ms"])
    succ_p95_col = _pick_latency_col(agg_all, ["latency_to_success_p95_ms", "latency_to_success_p90_ms"])
    all_med_col = _pick_latency_col(agg_all, ["latency_all_median_ms", "fail_time_median_ms"])
    all_p95_col = _pick_latency_col(agg_all, ["latency_all_p95_ms", "fail_time_p95_ms"])

    if not any([succ_med_col, succ_p95_col, all_med_col, all_p95_col]):
        return made

    for (occ, hop, ma), g3 in work.groupby(gkeys, dropna=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        has_line = False
        for label, g in g3.groupby("case_label", dropna=False):
            gg = g.sort_values("duty_cycle")
            x = pd.to_numeric(gg["duty_cycle"], errors="coerce").to_numpy(dtype=float)
            tag = str(label)
            if succ_med_col:
                y = pd.to_numeric(gg[succ_med_col], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(y).any():
                    ax.plot(x, y, marker="o", linestyle="--", label=f"{tag} succ_med")
                    has_line = True
            if succ_p95_col:
                y = pd.to_numeric(gg[succ_p95_col], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(y).any():
                    ax.plot(x, y, marker="^", linestyle=":", label=f"{tag} succ_p95")
                    has_line = True
            if all_med_col:
                y = pd.to_numeric(gg[all_med_col], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(y).any():
                    ax.plot(x, y, marker="x", linestyle="-", label=f"{tag} all_med")
                    has_line = True
            if all_p95_col:
                y = pd.to_numeric(gg[all_p95_col], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(y).any():
                    ax.plot(x, y, marker="s", linestyle="-.", label=f"{tag} all_p95")
                    has_line = True
        if has_line:
            ax.set_title(f"Latency vs duty (occ={occ}, hop={hop}, m={int(ma)})")
            ax.set_xlabel("duty_cycle")
            ax.set_ylabel("latency [ms]")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fn = f"plot_overlay_latency_occ={occ}_hop={hop}_m={int(ma)}.png"
            fig.savefig(out_dir / fn, dpi=160)
            made.append(fn)
        plt.close(fig)

    return made


def _make_summary(cases: list[CaseData]) -> pd.DataFrame:
    rows = []
    for c in cases:
        d = c.trial_df
        if d.empty:
            continue
        row: dict[str, Any] = {
            "label": c.label,
            "n_trials_total": int(len(d)),
            "duty_min": float(np.nanmin(pd.to_numeric(d.get("duty_cycle"), errors="coerce"))),
            "duty_max": float(np.nanmax(pd.to_numeric(d.get("duty_cycle"), errors="coerce"))),
            "n_ap_mean": float(np.nanmean(pd.to_numeric(d.get("n_ap"), errors="coerce"))),
            "n_ap_std": float(np.nanstd(pd.to_numeric(d.get("n_ap"), errors="coerce"))),
            "n_ap_min": float(np.nanmin(pd.to_numeric(d.get("n_ap"), errors="coerce"))),
            "n_ap_max": float(np.nanmax(pd.to_numeric(d.get("n_ap"), errors="coerce"))),
            "bursty_frac_mean": float(np.nanmean(pd.to_numeric(d.get("occ_bursty_frac"), errors="coerce"))),
            "bursty_frac_std": float(np.nanstd(pd.to_numeric(d.get("occ_bursty_frac"), errors="coerce"))),
            "partition_ok_rate": float(np.nanmean(pd.to_numeric(d.get("partition_ok"), errors="coerce"))),
            "coverage_ratio_min": float(np.nanmin(pd.to_numeric(d.get("coverage_ratio"), errors="coerce"))),
            "coverage_ratio_max": float(np.nanmax(pd.to_numeric(d.get("coverage_ratio"), errors="coerce"))),
        }
        for bw, col in zip(BW_KEYS, ["bw20_cnt", "bw40_cnt", "bw80_cnt", "bw160_cnt"]):
            row[f"bw{bw}_mean"] = float(np.nanmean(pd.to_numeric(d.get(col), errors="coerce")))
            row[f"bw{bw}_std"] = float(np.nanstd(pd.to_numeric(d.get(col), errors="coerce")))
        off = d[d.get("hopping", "").astype(str).str.lower() == "off"]
        on = d[d.get("hopping", "").astype(str).str.lower() == "on"]
        row["hop_off_unique1_ratio"] = float(
            np.nanmean(pd.to_numeric(off.get("nb_seq_unique_count"), errors="coerce") == 1)
        ) if len(off) else np.nan
        row["hop_on_unique_gt1_ratio"] = float(
            np.nanmean(pd.to_numeric(on.get("nb_seq_unique_count"), errors="coerce") > 1)
        ) if len(on) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    resolved: list[tuple[str, list[Path]]],
    sanity_msgs: list[str],
    sanity_errs: list[str],
    summary_df: pd.DataFrame,
    plot_files: list[str],
) -> None:
    rep = out_dir / "report.md"
    lines = ["# Postprocess Report", ""]
    lines.append("## Inputs")
    lines.append(f"- pick: {args.pick}")
    lines.append(f"- strict: {bool(args.strict)}")
    lines.append(f"- filter_occupancy: {args.filter_occupancy}")
    lines.append(f"- filter_hopping: {args.filter_hopping}")
    lines.append(f"- filter_max_attempts: {args.filter_max_attempts}")
    lines.append("")
    lines.append("## Resolved runs")
    for lb, dirs in resolved:
        lines.append(f"- {lb}: {', '.join(str(d) for d in dirs)}")
    lines.append("")
    lines.append("## Sanity checks")
    lines.append(f"- status: {'PASS' if not sanity_errs else 'FAIL'}")
    for m in sanity_msgs:
        lines.append(f"- {m}")
    if sanity_errs:
        lines.append("- errors:")
        for e in sanity_errs:
            lines.append(f"  - {e}")
    lines.append("")
    lines.append("## Summary table")
    if summary_df.empty:
        lines.append("- empty")
    else:
        try:
            lines.append(summary_df.to_markdown(index=False))
        except Exception:
            lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("## Generated plots")
    for f in plot_files:
        lines.append(f"- {f}")

    rep.write_text("\n".join(lines))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Postprocess random 6GHz duty-sweep runs")
    ap.add_argument("--runs", nargs="+", required=True, help="Run dirs or parent dirs")
    ap.add_argument("--labels", type=str, default=None, help="Comma labels matching runs")
    ap.add_argument("--pick", choices=["latest", "all"], default="latest")
    ap.add_argument("--out-dir", type=str, default="simulation/mms/results/postprocess")
    ap.add_argument("--filter-occupancy", type=str, default="mixed")
    ap.add_argument("--filter-max-attempts", type=str, default="16")
    ap.add_argument("--filter-hopping", type=str, default="off,on")
    ap.add_argument("--strict", type=int, default=1)
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    run_paths = [Path(p) for p in args.runs]
    labels = _parse_csv_list(args.labels) if args.labels else [p.name for p in run_paths]
    if len(labels) != len(run_paths):
        raise ValueError("--labels count must match --runs count")

    resolved: list[tuple[str, list[Path]]] = []
    for p, lb in zip(run_paths, labels):
        r = _resolve_run_dirs(p, args.pick)
        if not r:
            raise FileNotFoundError(f"no run dirs resolved for input={p}")
        resolved.append((lb, r))

    cases: list[CaseData] = []
    for lb, dirs in resolved:
        cases.append(_load_case(lb, dirs))

    trial_all = pd.concat([c.trial_df for c in cases], ignore_index=True) if cases else pd.DataFrame()
    agg_all = pd.concat([c.agg_df for c in cases], ignore_index=True) if cases else pd.DataFrame()

    strict = _as_bool(args.strict)
    sanity_msgs, sanity_errs = _sanity_checks(cases, strict=strict)
    if sanity_errs and strict:
        msg = "\n".join(sanity_errs)
        raise RuntimeError(f"strict sanity check failed:\n{msg}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = out_dir / "trial_merged.csv"
    agg_csv = out_dir / "aggregate_merged.csv"
    trial_all.to_csv(trial_csv, index=False)
    agg_all.to_csv(agg_csv, index=False)

    summary_df = _make_summary(cases)
    summary_csv = out_dir / "summary_cases.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_files = []
    plot_files.extend(_plot_distributions(trial_all, out_dir))
    plot_files.extend(
        _plot_overlays(
            agg_all,
            out_dir,
            filter_occupancy=_parse_csv_list(args.filter_occupancy),
            filter_hopping=_parse_csv_list(args.filter_hopping),
            filter_max_attempts=_parse_int_csv(args.filter_max_attempts),
        )
    )

    # Canonical output aliases requested by users/docs.
    p_over = [f for f in plot_files if f.startswith("plot_overlay_p_overall_")]
    p_lat = [f for f in plot_files if f.startswith("plot_overlay_latency_")]
    if p_over:
        shutil.copy2(out_dir / p_over[0], out_dir / "plot_overlay_p_overall.png")
        plot_files.append("plot_overlay_p_overall.png")
    if p_lat:
        shutil.copy2(out_dir / p_lat[0], out_dir / "plot_overlay_latency.png")
        plot_files.append("plot_overlay_latency.png")

    _write_report(
        out_dir,
        args=args,
        resolved=resolved,
        sanity_msgs=sanity_msgs,
        sanity_errs=sanity_errs,
        summary_df=summary_df,
        plot_files=plot_files,
    )

    print(f"saved: {out_dir}")
    print(f"- {trial_csv.name}")
    print(f"- {agg_csv.name}")
    print(f"- {summary_csv.name}")
    print(f"- report.md")
    for f in plot_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
