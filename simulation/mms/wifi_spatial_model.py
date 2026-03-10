from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_160MHZ_CH_SET = [32, 64, 96, 128, 160, 192]


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _lin_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))


def _frac_overlap(fc_a_hz: float, bw_a_hz: float, fc_b_hz: float, bw_b_hz: float) -> float:
    a0 = float(fc_a_hz) - 0.5 * float(bw_a_hz)
    a1 = float(fc_a_hz) + 0.5 * float(bw_a_hz)
    b0 = float(fc_b_hz) - 0.5 * float(bw_b_hz)
    b1 = float(fc_b_hz) + 0.5 * float(bw_b_hz)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return float(inter / max(float(bw_a_hz), 1.0))


def wifi_ch_to_fc_hz(ch: int) -> float:
    # Existing simulator convention: fc_MHz = 5950 + 5*ch
    return float((5950.0 + 5.0 * float(ch)) * 1e6)


@dataclass
class WiFiSpatialConfig:
    area_size_m: float = 100.0
    n_ap: int = 20
    n_sta_per_ap: int = 0
    ap_tx_power_dbm: float = 20.0
    pathloss_n: float = 2.4
    pl0_db: float = 46.0
    shadowing_sigma_db: float = 4.0
    duty_cycle: float = 0.75
    txop_ms: float = 0.9
    gap_jitter_ms: float = 0.3
    seed: int = 1
    cca_threshold_dbm: float = -82.0
    cca_duration_ms: float = 0.05
    backoff_slot_ms: float = 0.009
    cw_min_slots: int = 15
    default_center_hz: float = 6.5e9
    default_bw_hz: float = 160e6
    # Per-AP occupancy simulation quantum
    time_quantum_ms: float = 0.05
    default_burst_mean_on_ms: float = 2.0
    default_burst_mean_off_ms: float = 2.0
    default_160mhz_ch_set: tuple[int, ...] = tuple(DEFAULT_160MHZ_CH_SET)
    # Optional small-scale fading on Wi-Fi interference links.
    enable_rician_fading: bool = True
    rician_k_factor_db: float = 6.0
    # ACI model: "overlap" (legacy rectangular overlap) or "lut" (PSD-derived coupling LUT).
    aci_model: str = "overlap"
    aci_lut_path: str | None = None
    aps: list[dict] | None = None


class WiFiACILUT:
    def __init__(self, table: dict[int, tuple[np.ndarray, np.ndarray]], *, nb_bw_mhz: float | None = None):
        self.table = table
        self.nb_bw_mhz = nb_bw_mhz

    @classmethod
    def from_csv(cls, path: str | Path) -> "WiFiACILUT":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ACI LUT not found: {p}")
        df = pd.read_csv(p)
        req = {"wifi_bw_mhz", "offset_mhz"}
        if not req.issubset(set(df.columns)):
            raise ValueError(f"ACI LUT missing required columns: {req}")
        if "coupling_linear" not in df.columns and "coupling_frac_model" not in df.columns:
            raise ValueError("ACI LUT requires coupling_linear (or coupling_frac_model) column")
        ccol = "coupling_linear" if "coupling_linear" in df.columns else "coupling_frac_model"
        tbl: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for bw, g in df.groupby("wifi_bw_mhz"):
            gg = g.sort_values("offset_mhz")
            x = np.asarray(np.abs(gg["offset_mhz"].to_numpy(dtype=float)), dtype=float)
            y = np.asarray(gg[ccol].to_numpy(dtype=float), dtype=float)
            y = np.clip(y, 1e-15, 1.0)
            # Symmetric model in |offset| and monotonic-ish smoothing is left to source LUT.
            xu, idx = np.unique(x, return_index=True)
            yu = y[idx]
            tbl[int(round(float(bw)))] = (xu, yu)
        nb_bw = float(df["nb_bw_mhz"].iloc[0]) if "nb_bw_mhz" in df.columns else None
        if not tbl:
            raise ValueError(f"empty ACI LUT: {p}")
        return cls(tbl, nb_bw_mhz=nb_bw)

    def has_bw(self, wifi_bw_mhz: float) -> bool:
        bw_req = int(round(float(wifi_bw_mhz)))
        return bool(bw_req in self.table)

    def offset_in_range(self, wifi_bw_mhz: float, offset_mhz: float) -> bool:
        bw_req = int(round(float(wifi_bw_mhz)))
        if bw_req not in self.table:
            return False
        x, _ = self.table[bw_req]
        if x.size == 0:
            return False
        off = abs(float(offset_mhz))
        return bool(float(x[0]) <= off <= float(x[-1]))

    def coupling_linear(self, wifi_bw_mhz: float, offset_mhz: float) -> float:
        if not self.table:
            return 0.0
        bw_req = int(round(float(wifi_bw_mhz)))
        if bw_req not in self.table:
            raise KeyError(f"wifi_bw_mhz={bw_req} not present in LUT keys={sorted(self.table.keys())}")
        x, y = self.table[bw_req]
        off = abs(float(offset_mhz))
        if x.size == 1:
            return float(np.clip(y[0], 1e-15, 1.0))
        val = float(np.interp(off, x, y, left=y[0], right=y[-1]))
        return float(np.clip(val, 1e-15, 1.0))


class WiFiSpatialModel:
    """
    Layout/spatial Wi-Fi model with per-AP duty_cycle and occupancy_mode.

    occupancy_mode:
    - uniform: iid Bernoulli(duty_cycle) each time-quantum slot
    - bursty: 2-state Markov ON/OFF with mean ON/OFF durations
    """

    def __init__(self, cfg: WiFiSpatialConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._init_aps()
        self.slot_ms = float(max(1e-3, cfg.time_quantum_ms))
        self.slot_cache: dict[int, np.ndarray] = {}
        self._rx_power_cache: dict[tuple[int, int, int, int], float] = {}
        self.current_slot = -1
        self._init_states()
        self._aci_model = str(getattr(cfg, "aci_model", "overlap")).strip().lower()
        self._aci_lut: WiFiACILUT | None = None
        self._aci_lookup_total = 0
        self._aci_lut_hits = 0
        self._aci_lut_missing_offsets = 0
        self._aci_lut_missing_bws: set[int] = set()
        lut_path = getattr(cfg, "aci_lut_path", None)
        if self._aci_model == "lut":
            if not lut_path:
                raise ValueError("aci_model='lut' requires aci_lut_path")
            self._aci_lut = WiFiACILUT.from_csv(lut_path)
        elif lut_path:
            # Allow explicit LUT path to force LUT model.
            self._aci_lut = WiFiACILUT.from_csv(lut_path)
            self._aci_model = "lut"

    def _init_aps(self) -> None:
        aps = self.cfg.aps
        if aps:
            xy = []
            tx = []
            fc = []
            bw = []
            duty = []
            mode = []
            mon = []
            mof = []
            rr_ch = list(self.cfg.default_160mhz_ch_set) or DEFAULT_160MHZ_CH_SET
            for i, a in enumerate(aps):
                xy.append([float(a.get("x", 0.0)), float(a.get("y", 0.0))])
                tx.append(float(a.get("tx_power_dBm", self.cfg.ap_tx_power_dbm)))
                ch = a.get("wifi_ch", None)
                if "channel_center_Hz" in a:
                    fc_i = float(a["channel_center_Hz"])
                elif ch is not None:
                    fc_i = wifi_ch_to_fc_hz(int(ch))
                else:
                    fc_i = wifi_ch_to_fc_hz(int(rr_ch[i % len(rr_ch)]))
                fc.append(float(fc_i))
                if "bw_Hz" in a:
                    bw_i = float(a["bw_Hz"])
                elif "bw_mhz" in a:
                    bw_i = float(a["bw_mhz"]) * 1e6
                else:
                    bw_i = float(self.cfg.default_bw_hz)
                bw.append(float(bw_i))
                duty_i = float(a.get("duty_cycle", a.get("traffic_load", self.cfg.duty_cycle)))
                duty.append(float(np.clip(duty_i, 0.0, 1.0)))
                mode_i = str(a.get("occupancy_mode", "uniform")).strip().lower()
                if mode_i not in {"uniform", "bursty"}:
                    mode_i = "uniform"
                mode.append(mode_i)
                mon.append(float(a.get("burst_mean_on_ms", self.cfg.default_burst_mean_on_ms)))
                mof.append(float(a.get("burst_mean_off_ms", self.cfg.default_burst_mean_off_ms)))
            self.ap_xy = np.asarray(xy, dtype=float)
            self.ap_tx_dbm = np.asarray(tx, dtype=float)
            self.ap_fc_hz = np.asarray(fc, dtype=float)
            self.ap_bw_hz = np.asarray(bw, dtype=float)
            self.ap_duty = np.asarray(duty, dtype=float)
            self.ap_mode = np.asarray(mode, dtype=object)
            self.ap_mean_on_ms = np.asarray(mon, dtype=float)
            self.ap_mean_off_ms = np.asarray(mof, dtype=float)
        else:
            self.ap_xy = self.rng.uniform(0.0, self.cfg.area_size_m, size=(self.cfg.n_ap, 2))
            self.ap_tx_dbm = np.full(self.cfg.n_ap, float(self.cfg.ap_tx_power_dbm), dtype=float)
            self.ap_fc_hz = np.full(self.cfg.n_ap, float(self.cfg.default_center_hz), dtype=float)
            self.ap_bw_hz = np.full(self.cfg.n_ap, float(self.cfg.default_bw_hz), dtype=float)
            self.ap_duty = np.full(self.cfg.n_ap, float(np.clip(self.cfg.duty_cycle, 0.0, 1.0)), dtype=float)
            self.ap_mode = np.full(self.cfg.n_ap, "uniform", dtype=object)
            self.ap_mean_on_ms = np.full(self.cfg.n_ap, float(self.cfg.default_burst_mean_on_ms), dtype=float)
            self.ap_mean_off_ms = np.full(self.cfg.n_ap, float(self.cfg.default_burst_mean_off_ms), dtype=float)
        self.n_ap = int(self.ap_xy.shape[0])

    def _init_states(self) -> None:
        self.state_on = self.rng.random(self.n_ap) < self.ap_duty
        self.p_on_to_off = np.zeros(self.n_ap, dtype=float)
        self.p_off_to_on = np.zeros(self.n_ap, dtype=float)
        for i in range(self.n_ap):
            p = float(self.ap_duty[i])
            if self.ap_mode[i] == "uniform":
                self.p_on_to_off[i] = 0.0
                self.p_off_to_on[i] = 0.0
                continue
            if p <= 0.0:
                self.state_on[i] = False
                self.p_on_to_off[i] = 1.0
                self.p_off_to_on[i] = 0.0
            elif p >= 1.0:
                self.state_on[i] = True
                self.p_on_to_off[i] = 0.0
                self.p_off_to_on[i] = 1.0
            else:
                on_slots = max(1.0, float(self.ap_mean_on_ms[i]) / self.slot_ms)
                off_slots_cfg = max(1.0, float(self.ap_mean_off_ms[i]) / self.slot_ms)
                # Keep requested long-term duty priority; if config off_ms inconsistent, recompute off from duty.
                off_slots_from_duty = on_slots * (1.0 - p) / p
                off_slots = max(1.0, off_slots_from_duty if np.isfinite(off_slots_from_duty) else off_slots_cfg)
                self.p_on_to_off[i] = float(np.clip(1.0 / on_slots, 0.0, 1.0))
                self.p_off_to_on[i] = float(np.clip(1.0 / off_slots, 0.0, 1.0))

    def _slot_idx(self, t_ms: float) -> int:
        return int(math.floor(float(t_ms) / self.slot_ms))

    def _advance_to_slot(self, target_slot: int) -> None:
        if target_slot <= self.current_slot:
            return
        for s in range(self.current_slot + 1, target_slot + 1):
            busy = np.empty(self.n_ap, dtype=bool)
            for i in range(self.n_ap):
                p = float(self.ap_duty[i])
                if self.ap_mode[i] == "uniform":
                    busy[i] = bool(self.rng.random() < p)
                    continue
                # Bursty Markov
                if p <= 0.0:
                    self.state_on[i] = False
                elif p >= 1.0:
                    self.state_on[i] = True
                elif self.state_on[i]:
                    if self.rng.random() < self.p_on_to_off[i]:
                        self.state_on[i] = False
                else:
                    if self.rng.random() < self.p_off_to_on[i]:
                        self.state_on[i] = True
                busy[i] = bool(self.state_on[i])
            self.slot_cache[s] = busy
        self.current_slot = target_slot

    def _busy_vec_at(self, t_ms: float) -> np.ndarray:
        s = self._slot_idx(t_ms)
        self._advance_to_slot(s)
        return self.slot_cache[s]

    def _rx_power_from_ap_dbm(self, ap_idx: int, rx_xy: np.ndarray) -> float:
        # Cache per (slot, ap, rx position) so same slot queries are consistent.
        # rx_xy in this simulator is typically fixed small set: NB RX / UWB RX.
        slot = max(0, int(self.current_slot))
        k = (slot, int(ap_idx), int(round(float(rx_xy[0]) * 1000.0)), int(round(float(rx_xy[1]) * 1000.0)))
        if k in self._rx_power_cache:
            return float(self._rx_power_cache[k])

        d = float(np.linalg.norm(self.ap_xy[ap_idx] - rx_xy))
        d = max(d, 1.0)
        shadow = float(self.rng.normal(0.0, self.cfg.shadowing_sigma_db))
        pl_db = self.cfg.pl0_db + 10.0 * self.cfg.pathloss_n * math.log10(d) + shadow
        p_dbm = float(self.ap_tx_dbm[ap_idx] - pl_db)

        if bool(self.cfg.enable_rician_fading):
            k_lin = 10.0 ** (float(self.cfg.rician_k_factor_db) / 10.0)
            # Narrowband-equivalent flat fading gain per slot.
            # E[|h|^2]=1 so average link budget remains unchanged.
            h_los = math.sqrt(k_lin / (k_lin + 1.0))
            sigma = math.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
            n1 = float(self.rng.normal(0.0, 1.0))
            n2 = float(self.rng.normal(0.0, 1.0))
            h = (h_los + sigma * n1) + 1j * (sigma * n2)
            g_db = 10.0 * math.log10(max(abs(h) ** 2, 1e-12))
            p_dbm += float(g_db)

        self._rx_power_cache[k] = float(p_dbm)
        return float(p_dbm)

    def occupancy_defer_ms(self, t_ms: float) -> float:
        s0 = self._slot_idx(float(t_ms))
        self._advance_to_slot(s0 + 1)
        busy0 = self.slot_cache[s0]
        if not bool(np.any(busy0)):
            return 0.0
        s = s0 + 1
        max_scan = 2000
        for _ in range(max_scan):
            self._advance_to_slot(s)
            if not bool(np.any(self.slot_cache[s])):
                return float((s - s0) * self.slot_ms)
            s += 1
        return float(max_scan * self.slot_ms)

    def estimate_interference_dbm(
        self,
        t_ms: float,
        rx_xy: tuple[float, float],
        rx_center_hz: float | None = None,
        rx_bw_hz: float | None = None,
    ) -> float:
        busy = self._busy_vec_at(float(t_ms))
        rx = np.asarray(rx_xy, dtype=float)
        center = float(self.cfg.default_center_hz if rx_center_hz is None else rx_center_hz)
        bw = float(self.cfg.default_bw_hz if rx_bw_hz is None else rx_bw_hz)
        p_lin = 0.0
        for i in range(self.n_ap):
            if not bool(busy[i]):
                continue
            frac = self._coupling_frac(int(i), center_hz=center, victim_bw_hz=bw)
            if frac <= 0.0:
                continue
            p_dbm = self._rx_power_from_ap_dbm(i, rx) + 10.0 * math.log10(max(frac, 1e-12))
            p_lin += _db_to_lin(p_dbm)
        return _lin_to_db(p_lin) if p_lin > 0.0 else -200.0

    def estimate_interference_avg_dbm(
        self,
        t_start_ms: float,
        t_end_ms: float,
        rx_xy: tuple[float, float],
        rx_center_hz: float | None = None,
        rx_bw_hz: float | None = None,
        n_samples: int = 5,
    ) -> float:
        if t_end_ms <= t_start_ms:
            return self.estimate_interference_dbm(
                t_ms=t_start_ms,
                rx_xy=rx_xy,
                rx_center_hz=rx_center_hz,
                rx_bw_hz=rx_bw_hz,
            )
        ts = np.linspace(float(t_start_ms), float(t_end_ms), num=max(2, int(n_samples)))
        vals = [
            _db_to_lin(
                self.estimate_interference_dbm(
                    t_ms=float(t),
                    rx_xy=rx_xy,
                    rx_center_hz=rx_center_hz,
                    rx_bw_hz=rx_bw_hz,
                )
            )
            for t in ts
        ]
        return _lin_to_db(float(np.mean(vals)))

    def realized_duty_per_ap(
        self,
        t_start_ms: float,
        t_end_ms: float,
    ) -> np.ndarray:
        if t_end_ms <= t_start_ms:
            return np.zeros(self.n_ap, dtype=float)
        s0 = self._slot_idx(float(t_start_ms))
        s1 = self._slot_idx(float(t_end_ms))
        self._advance_to_slot(s1)
        mats = [self.slot_cache[s] for s in range(s0, s1 + 1) if s in self.slot_cache]
        if not mats:
            return np.zeros(self.n_ap, dtype=float)
        m = np.vstack(mats).astype(float)
        return np.mean(m, axis=0)

    def aggregate_busy_fraction(
        self,
        t_start_ms: float,
        t_end_ms: float,
        rx_center_hz: float | None = None,
        rx_bw_hz: float | None = None,
    ) -> float:
        if t_end_ms <= t_start_ms:
            return 0.0
        center = float(self.cfg.default_center_hz if rx_center_hz is None else rx_center_hz)
        bw = float(self.cfg.default_bw_hz if rx_bw_hz is None else rx_bw_hz)
        s0 = self._slot_idx(float(t_start_ms))
        s1 = self._slot_idx(float(t_end_ms))
        self._advance_to_slot(s1)
        n = 0
        b = 0
        overlap = np.asarray(
            [self._coupling_frac(i, center_hz=center, victim_bw_hz=bw) for i in range(self.n_ap)],
            dtype=float,
        )
        for s in range(s0, s1 + 1):
            if s not in self.slot_cache:
                continue
            busy = self.slot_cache[s]
            any_busy = bool(np.any(np.logical_and(busy, overlap > 0.0)))
            b += int(any_busy)
            n += 1
        if n <= 0:
            return 0.0
        return float(b / n)

    def summarize_aps(
        self,
        *,
        uwb_nodes_xy: list[tuple[float, float]] | None = None,
    ) -> dict:
        bw_hist: dict[int, int] = {}
        mode_hist: dict[str, int] = {}
        for i in range(self.n_ap):
            bw = int(round(float(self.ap_bw_hz[i]) / 1e6))
            bw_hist[bw] = int(bw_hist.get(bw, 0) + 1)
            m = str(self.ap_mode[i])
            mode_hist[m] = int(mode_hist.get(m, 0) + 1)

        out = {
            "n_ap": int(self.n_ap),
            "bw_hist_mhz": bw_hist,
            "occupancy_hist": mode_hist,
            "duty_mean": float(np.mean(self.ap_duty)) if self.n_ap > 0 else float("nan"),
            "duty_p95": float(np.percentile(self.ap_duty, 95)) if self.n_ap > 0 else float("nan"),
        }
        if uwb_nodes_xy:
            dvals: list[float] = []
            for i in range(self.n_ap):
                ax, ay = float(self.ap_xy[i, 0]), float(self.ap_xy[i, 1])
                for ux, uy in uwb_nodes_xy:
                    dvals.append(float(math.hypot(ax - float(ux), ay - float(uy))))
            if dvals:
                arr = np.asarray(dvals, dtype=float)
                out["ap_to_uwb_min_m"] = float(np.min(arr))
                out["ap_to_uwb_mean_m"] = float(np.mean(arr))
                out["ap_to_uwb_max_m"] = float(np.max(arr))
        return out

    def aci_stats(self) -> dict:
        total = int(self._aci_lookup_total)
        hits = int(self._aci_lut_hits)
        hit_rate = float(hits / total) if total > 0 else (1.0 if self._aci_model == "overlap" else float("nan"))
        return {
            "aci_model": str(self._aci_model),
            "aci_lookup_total": total,
            "aci_lut_hits": hits,
            "aci_lut_hit_rate": float(hit_rate),
            "aci_lut_missing_offsets_count": int(self._aci_lut_missing_offsets),
            "aci_lut_missing_bws": sorted(int(v) for v in self._aci_lut_missing_bws),
        }

    def _coupling_frac(self, ap_idx: int, *, center_hz: float, victim_bw_hz: float) -> float:
        self._aci_lookup_total += 1
        # Legacy overlap coupling.
        overlap = _frac_overlap(float(self.ap_fc_hz[ap_idx]), float(self.ap_bw_hz[ap_idx]), center_hz, victim_bw_hz)
        if self._aci_model != "lut":
            self._aci_lut_hits += 1
            return float(overlap)
        if self._aci_lut is None:
            raise RuntimeError("aci_model='lut' selected but LUT is unavailable")
        off_mhz = (float(self.ap_fc_hz[ap_idx]) - float(center_hz)) / 1e6
        wifi_bw_mhz = float(self.ap_bw_hz[ap_idx]) / 1e6
        bw_req = int(round(wifi_bw_mhz))
        if not self._aci_lut.has_bw(bw_req):
            self._aci_lut_missing_bws.add(int(bw_req))
            raise RuntimeError(f"ACI LUT missing wifi_bw_mhz={bw_req}")
        self._aci_lut_hits += 1
        return float(self._aci_lut.coupling_linear(wifi_bw_mhz=wifi_bw_mhz, offset_mhz=off_mhz))


def load_wifi_layout(path: str | Path) -> WiFiSpatialConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"layout file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML layout requires pyyaml; use JSON or install pyyaml") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    aps = list(data.get("wifi_aps", []))
    # Backward compatibility + default 160MHz round-robin assignment when channel not provided.
    default_set = list(data.get("default_160mhz_ch_set", DEFAULT_160MHZ_CH_SET))
    if aps:
        for i, ap in enumerate(aps):
            if "channel_center_Hz" not in ap and "wifi_ch" not in ap:
                ap["wifi_ch"] = int(default_set[i % len(default_set)])
            if "bw_Hz" not in ap and "bw_mhz" not in ap:
                ap["bw_mhz"] = 160.0
            if "duty_cycle" not in ap and "traffic_load" in ap:
                ap["duty_cycle"] = float(ap["traffic_load"])
            if "traffic_load" not in ap and "duty_cycle" in ap:
                ap["traffic_load"] = float(ap["duty_cycle"])

    area_w = float(data.get("area_w_m", data.get("area_size_m", 100.0)))
    area_h = float(data.get("area_h_m", data.get("area_size_m", 100.0)))
    area_size = float(max(area_w, area_h))
    return WiFiSpatialConfig(
        area_size_m=area_size,
        n_ap=int(data.get("n_ap", len(aps) if aps else 20)),
        n_sta_per_ap=int(data.get("n_sta_per_ap", 0)),
        ap_tx_power_dbm=float(data.get("ap_tx_power_dBm", 20.0)),
        pathloss_n=float(data.get("pathloss_n", 2.4)),
        pl0_db=float(data.get("pl0_db", 46.0)),
        shadowing_sigma_db=float(data.get("shadowing_sigma_db", 4.0)),
        duty_cycle=float(data.get("duty_cycle", 0.75)),
        txop_ms=float(data.get("txop_ms", 0.9)),
        gap_jitter_ms=float(data.get("gap_jitter_ms", 0.3)),
        seed=int(data.get("seed", 1)),
        cca_threshold_dbm=float(data.get("cca_threshold_dbm", -82.0)),
        cca_duration_ms=float(data.get("cca_duration_ms", 0.05)),
        backoff_slot_ms=float(data.get("backoff_slot_ms", 0.009)),
        cw_min_slots=int(data.get("cw_min_slots", 15)),
        default_center_hz=float(data.get("default_center_hz", wifi_ch_to_fc_hz(int(default_set[0])))),
        default_bw_hz=float(data.get("default_bw_hz", 160e6)),
        time_quantum_ms=float(data.get("time_quantum_ms", 0.05)),
        default_burst_mean_on_ms=float(data.get("default_burst_mean_on_ms", 2.0)),
        default_burst_mean_off_ms=float(data.get("default_burst_mean_off_ms", 2.0)),
        default_160mhz_ch_set=tuple(int(v) for v in default_set),
        enable_rician_fading=bool(data.get("enable_rician_fading", True)),
        rician_k_factor_db=float(data.get("rician_k_factor_db", 6.0)),
        aci_model=str(data.get("aci_model", "overlap")),
        aci_lut_path=data.get("aci_lut_path", None),
        aps=aps if aps else None,
    )
