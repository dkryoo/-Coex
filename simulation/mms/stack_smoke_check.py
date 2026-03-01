# simulation/mms/stack_smoke_check.py
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------
@dataclass
class CaseMetrics:
    name: str
    nb_control_ok: Optional[bool] = None
    ber: Optional[float] = None
    fer: Optional[float] = None
    rmse_valid_m: Optional[float] = None
    bias_valid_m: Optional[float] = None
    std_valid_m: Optional[float] = None
    rmse_all_m: Optional[float] = None
    bias_all_m: Optional[float] = None
    std_all_m: Optional[float] = None
    ranging_fail: Optional[float] = None
    snr_db: Optional[float] = None

    nb_wifi_overlap: Optional[float] = None
    uwb_wifi_overlap: Optional[float] = None
    uwb_waveform_interference: Optional[str] = None  # "ON"/"OFF"
    range_bias_corr_m: Optional[float] = None

    detect_ok: Optional[bool] = None
    frame_success: Optional[bool] = None

    # RF selectivity stage (victim receiver view)
    p_int_dbm: Optional[float] = None
    p_sig_dbm: Optional[float] = None
    noise_dbm: Optional[float] = None
    rssi_total_dbm: Optional[float] = None

    # PSD artifacts
    psd_csv: Optional[str] = None
    psd_png: Optional[str] = None
    psd_sanity_delta_db: Optional[float] = None

    raw_block: str = ""


@dataclass
class CheckResult:
    name: str
    status: str  # PASS/WARN/FAIL/SKIP
    details: str = ""


# -----------------------------
# Regex patterns
# -----------------------------
RE_CASE = re.compile(r"--- Case:\s*(?P<name>.+?)\s*---")
RE_FLOW = re.compile(r"Flow:\s*(?P<flow>.+)")
RE_OVERLAP = re.compile(
    r"NB/Wi-Fi overlap=(?P<nb>[-0-9.]+),\s*UWB/Wi-Fi overlap=(?P<uwb>[-0-9.]+),\s*UWB waveform interference=(?P<intf>ON|OFF)"
    r"(?:,\s*range_bias_corr=(?P<rb>[-0-9.]+)\s*m)?"
)
RE_NB_SUMMARY_LINE = re.compile(r"NB control_ok=")

RE_DETECT_OK = re.compile(r"\[MMS stage detect\].*detect_ok=(True|False)")
RE_FRAME_GATE = re.compile(r"\[MMS frame gate\].*frame_success=(True|False)")

RE_PSD_SAVED = re.compile(r"PSD saved:\s*\{.*'csv':\s*'([^']+)'.*'png':\s*'([^']+)'.*\}")
RE_PSD_SANITY = re.compile(r"PSD sanity:\s*.*delta=([-0-9.]+)\s*dB")

# Example:
# [MMS stage rf_selectivity] P_sig=... (-60.33 dBm), P_int=... (-53.66 dBm), Noise=... (-83.24 dBm), RSSI_total=... (-52.85 dBm)
RE_RF_SELECTIVITY = re.compile(
    r"\[MMS stage rf_selectivity\].*?"
    r"P_sig=.*?\((?P<p_sig>[-0-9.]+)\s*dBm\).*?"
    r"P_int=.*?\((?P<p_int>[-0-9.]+)\s*dBm\).*?"
    r"Noise=.*?\((?P<noise>[-0-9.]+)\s*dBm\).*?"
    r"RSSI_total=.*?\((?P<rssi>[-0-9.]+)\s*dBm\)"
)


# -----------------------------
# Helpers
# -----------------------------
def _parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if value.upper().startswith("N/A"):
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", value, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_bool(value: str) -> Optional[bool]:
    value = value.strip()
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _split_blocks_by_config_debug(stdout: str) -> List[str]:
    """
    full_stack_mms_demo.py 출력은 케이스마다 [MMS config debug]가 등장.
    이걸 기준으로 블록을 나누면 케이스별 파싱이 쉬움.
    """
    marker = "[MMS config debug]"
    if marker not in stdout:
        # fallback: case 헤더 기준 split
        return [stdout]

    parts = stdout.split(marker)
    blocks: List[str] = []
    # 첫 파트는 preamble
    preamble = parts[0]
    for rest in parts[1:]:
        blocks.append(marker + rest)

    # preamble이 케이스 1의 일부일 수 있어 blocks[0] 앞에 붙여줌
    if blocks:
        blocks[0] = preamble + blocks[0]
    else:
        blocks = [stdout]
    return blocks


def _parse_nb_summary_line(line: str, cm: CaseMetrics) -> None:
    # "NB control_ok=True | BER=... | FER=... | RMSE(valid)=... m | Bias(valid)=... m | ..."
    parts = [p.strip() for p in line.split("|")]
    for p in parts:
        if not p:
            continue
        if p.startswith("NB control_ok="):
            cm.nb_control_ok = _parse_bool(p.split("=", 1)[1])
            continue

        if "=" not in p:
            continue

        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k == "BER":
            cm.ber = _parse_float(v)
        elif k == "FER":
            cm.fer = _parse_float(v)
        elif k == "RMSE(valid)":
            cm.rmse_valid_m = _parse_float(v)
        elif k == "Bias(valid)":
            cm.bias_valid_m = _parse_float(v)
        elif k == "Std(valid)":
            cm.std_valid_m = _parse_float(v)
        elif k == "RMSE(all-phy)":
            cm.rmse_all_m = _parse_float(v)
        elif k == "Bias(all-phy)":
            cm.bias_all_m = _parse_float(v)
        elif k == "Std(all-phy)":
            cm.std_all_m = _parse_float(v)
        elif k == "RangingFail":
            cm.ranging_fail = _parse_float(v)
        elif k == "SNR":
            cm.snr_db = _parse_float(v)


def parse_cases(stdout: str) -> Tuple[Optional[str], Dict[str, CaseMetrics]]:
    flow = None
    m_flow = RE_FLOW.search(stdout)
    if m_flow:
        flow = m_flow.group("flow").strip()

    blocks = _split_blocks_by_config_debug(stdout)
    cases: Dict[str, CaseMetrics] = {}

    for block in blocks:
        m_case = RE_CASE.search(block)
        if not m_case:
            continue

        name = m_case.group("name").strip()
        cm = CaseMetrics(name=name, raw_block=block)

        # overlap / interference
        m_ov = RE_OVERLAP.search(block)
        if m_ov:
            cm.nb_wifi_overlap = _parse_float(m_ov.group("nb"))
            cm.uwb_wifi_overlap = _parse_float(m_ov.group("uwb"))
            cm.uwb_waveform_interference = m_ov.group("intf")
            rb = m_ov.groupdict().get("rb")
            if rb is not None:
                cm.range_bias_corr_m = _parse_float(rb)

        # detect_ok
        m_det = RE_DETECT_OK.search(block)
        if m_det:
            cm.detect_ok = _parse_bool(m_det.group(1))

        # frame gate success
        m_fg = RE_FRAME_GATE.search(block)
        if m_fg:
            cm.frame_success = _parse_bool(m_fg.group(1))

        # PSD
        m_psd = RE_PSD_SAVED.search(block)
        if m_psd:
            cm.psd_csv = m_psd.group(1)
            cm.psd_png = m_psd.group(2)

        m_sanity = RE_PSD_SANITY.search(block)
        if m_sanity:
            cm.psd_sanity_delta_db = _parse_float(m_sanity.group(1))

        # RF selectivity stage powers
        m_rf = RE_RF_SELECTIVITY.search(block)
        if m_rf:
            cm.p_sig_dbm = _parse_float(m_rf.group("p_sig"))
            cm.p_int_dbm = _parse_float(m_rf.group("p_int"))
            cm.noise_dbm = _parse_float(m_rf.group("noise"))
            cm.rssi_total_dbm = _parse_float(m_rf.group("rssi"))

        # NB summary line
        for line in block.splitlines():
            if RE_NB_SUMMARY_LINE.search(line):
                _parse_nb_summary_line(line, cm)
                break

        cases[name] = cm

    return flow, cases


def _exists_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def cca_energy_detect_busy(power_dbm: Optional[float], ed_threshold_dbm: float) -> Optional[bool]:
    """
    아주 단순한 에너지 감지(ED) 기반 CCA.
    - power_dbm이 None이면 판단 불가(None)
    """
    if power_dbm is None:
        return None
    return power_dbm >= ed_threshold_dbm


# -----------------------------
# Checks (mapping to your 5 items)
# -----------------------------
def check_1_protocol(flow: Optional[str], cases: Dict[str, CaseMetrics]) -> CheckResult:
    if not flow:
        return CheckResult(
            name="1) MMS Ranging protocol (initiation 포함)",
            status="FAIL",
            details="출력에서 Flow: 라인을 찾지 못했습니다. (프로토콜 시퀀스 로그가 안 찍히는 상태일 수 있음)",
        )

    # 프로토콜 문자열에 ADV/POLL/RESP/CONF가 들어가는지 확인 (없으면 WARN)
    need = ["ADV", "POLL", "RESP", "CONF"]
    missing = [x for x in need if x not in flow]
    if missing:
        return CheckResult(
            name="1) MMS Ranging protocol (initiation 포함)",
            status="WARN",
            details=f"Flow 라인은 있으나 {missing} 토큰이 없습니다. flow='{flow}'",
        )

    # baseline 케이스에서 NB control_ok=True인지 확인
    base = cases.get("Wi-Fi OFF baseline")
    if not base:
        return CheckResult(
            name="1) MMS Ranging protocol (initiation 포함)",
            status="FAIL",
            details="케이스 'Wi-Fi OFF baseline'을 찾지 못했습니다.",
        )

    if base.nb_control_ok is not True:
        return CheckResult(
            name="1) MMS Ranging protocol (initiation 포함)",
            status="FAIL",
            details=f"baseline에서 NB control_ok가 True가 아닙니다: {base.nb_control_ok}",
        )

    return CheckResult(
        name="1) MMS Ranging protocol (initiation 포함)",
        status="PASS",
        details=f"flow='{flow}', baseline NB control_ok=True 확인",
    )


def check_2_uwb_nb_waveforms(cases: Dict[str, CaseMetrics], repo_root: Path) -> CheckResult:
    """
    2) UWB/NB 송수신 파형 및 동작:
    - detect_ok=True
    - frame_success=True (가능한 케이스에서)
    - PSD 산출물/PSD sanity(delta) 확인
    """
    must_have = ["Wi-Fi OFF baseline", "Wi-Fi ON out-of-band, stop=60dB", "Wi-Fi OFF baseline sanity (no multipath/ppm)"]
    missing = [n for n in must_have if n not in cases]
    if missing:
        return CheckResult(
            name="2) UWB + Narrowband Tx/Rx 파형 및 동작",
            status="FAIL",
            details=f"필수 케이스 누락: {missing}",
        )

    problems: List[str] = []

    for name in must_have:
        cm = cases[name]

        # detect_ok
        if cm.detect_ok is not True:
            problems.append(f"[{name}] detect_ok != True (got {cm.detect_ok})")

        # frame_success (sanity/baseline/oob60은 성공 기대)
        if cm.frame_success is not True:
            problems.append(f"[{name}] frame_success != True (got {cm.frame_success})")

        # PSD artifacts
        if cm.psd_csv:
            csv_path = (repo_root / cm.psd_csv).resolve()
            if not _exists_nonempty(csv_path):
                problems.append(f"[{name}] PSD csv missing/empty: {cm.psd_csv}")
        else:
            problems.append(f"[{name}] PSD csv path not found in logs")

        if cm.psd_png:
            png_path = (repo_root / cm.psd_png).resolve()
            if not _exists_nonempty(png_path):
                problems.append(f"[{name}] PSD png missing/empty: {cm.psd_png}")
        else:
            problems.append(f"[{name}] PSD png path not found in logs")

        # PSD sanity delta
        if cm.psd_sanity_delta_db is None:
            problems.append(f"[{name}] PSD sanity delta not found")
        else:
            if abs(cm.psd_sanity_delta_db) > 0.5:
                problems.append(f"[{name}] PSD sanity delta too large: {cm.psd_sanity_delta_db:.2f} dB")

    if problems:
        return CheckResult(
            name="2) UWB + Narrowband Tx/Rx 파형 및 동작",
            status="FAIL",
            details="; ".join(problems),
        )

    return CheckResult(
        name="2) UWB + Narrowband Tx/Rx 파형 및 동작",
        status="PASS",
        details="baseline/oob60/sanity에서 detect_ok, frame_success, PSD 산출물, PSD sanity(delta) 모두 OK",
    )


def check_3_wifi_waveform(cases: Dict[str, CaseMetrics]) -> CheckResult:
    """
    3) Wi-Fi 송신 파형/동작:
    - in-band / out-of-band 케이스 존재
    - in-band에서 interference=ON, overlap~1
    - out-of-band에서 overlap~0
    """
    inband = cases.get("Wi-Fi ON in-band (Δf~0.4MHz)")
    oob60 = cases.get("Wi-Fi ON out-of-band, stop=60dB")

    if not inband or not oob60:
        return CheckResult(
            name="3) Wi-Fi 송신 파형 및 동작",
            status="FAIL",
            details="in-band 또는 out-of-band(60dB) 케이스가 로그에 없습니다.",
        )

    probs: List[str] = []

    if inband.uwb_waveform_interference != "ON":
        probs.append(f"in-band: UWB waveform interference expected ON, got {inband.uwb_waveform_interference}")
    if inband.nb_wifi_overlap is not None and inband.nb_wifi_overlap < 0.9:
        probs.append(f"in-band: NB/Wi-Fi overlap expected ~1, got {inband.nb_wifi_overlap}")

    if oob60.nb_wifi_overlap is not None and oob60.nb_wifi_overlap > 0.1:
        probs.append(f"oob60: NB/Wi-Fi overlap expected ~0, got {oob60.nb_wifi_overlap}")

    # 간섭 파워가 실제로 올라갔는지(대략) 확인: in-band는 P_int가 매우 큼(-70~-40dBm대가 흔함)
    if inband.p_int_dbm is None:
        probs.append("in-band: rf_selectivity P_int(dBm) 파싱 실패")
    else:
        if inband.p_int_dbm < -120:  # 너무 작으면 Wi-Fi 파형 주입이 안된 느낌
            probs.append(f"in-band: P_int_dbm too small (expected strong), got {inband.p_int_dbm:.2f} dBm")

    if probs:
        return CheckResult(
            name="3) Wi-Fi 송신 파형 및 동작",
            status="WARN",
            details="; ".join(probs),
        )

    return CheckResult(
        name="3) Wi-Fi 송신 파형 및 동작",
        status="PASS",
        details="in-band/out-of-band 케이스 존재 + overlap/interference 상태 정상 + in-band 간섭 파워 확인",
    )


def check_4_wifi_cca(cases: Dict[str, CaseMetrics], ed_threshold_dbm: float) -> CheckResult:
    """
    4) Wi-Fi CCA 동작:
    - 여기서는 '실제 Wi-Fi MAC CCA 구현' 호출이 아니라,
      로그에 찍히는 rf_selectivity 단계의 P_int(dBm)을 이용해
      ED 기반 busy/idle이 기대대로 나오는지를 확인하는 스모크 체크를 제공.
    """
    base = cases.get("Wi-Fi OFF baseline")
    inband = cases.get("Wi-Fi ON in-band (Δf~0.4MHz)")
    oob60 = cases.get("Wi-Fi ON out-of-band, stop=60dB")

    if not base or not inband or not oob60:
        return CheckResult(
            name="4) Wi-Fi CCA 동작(ED 기반 스모크 체크)",
            status="SKIP",
            details="baseline/in-band/oob60 케이스가 모두 있어야 CCA 스모크 체크 가능",
        )

    base_busy = cca_energy_detect_busy(base.p_int_dbm, ed_threshold_dbm)
    inband_busy = cca_energy_detect_busy(inband.p_int_dbm, ed_threshold_dbm)
    oob60_busy = cca_energy_detect_busy(oob60.p_int_dbm, ed_threshold_dbm)

    probs: List[str] = []
    # 기대: baseline idle(False), in-band busy(True), oob60 idle(False)
    if base_busy is None or inband_busy is None or oob60_busy is None:
        probs.append("P_int_dbm 파싱이 일부 실패해서 busy/idle 판단 불가")
    else:
        if base_busy is True:
            probs.append(f"baseline: expected IDLE but busy=True (P_int={base.p_int_dbm:.2f} dBm)")
        if inband_busy is False:
            probs.append(f"in-band: expected BUSY but busy=False (P_int={inband.p_int_dbm:.2f} dBm)")
        if oob60_busy is True:
            probs.append(f"oob60: expected IDLE but busy=True (P_int={oob60.p_int_dbm:.2f} dBm)")

    if probs:
        return CheckResult(
            name="4) Wi-Fi CCA 동작(ED 기반 스모크 체크)",
            status="WARN",
            details=f"ED threshold={ed_threshold_dbm:.1f} dBm, " + "; ".join(probs),
        )

    return CheckResult(
        name="4) Wi-Fi CCA 동작(ED 기반 스모크 체크)",
        status="PASS",
        details=(
            f"ED threshold={ed_threshold_dbm:.1f} dBm에서 "
            f"baseline(IDLE)/in-band(BUSY)/oob60(IDLE) 판정이 기대대로 나옴"
        ),
    )


def check_5_end_to_end_performance(cases: Dict[str, CaseMetrics]) -> CheckResult:
    """
    5) Wi-Fi/UWB 전체 배치 입력 시 MMS 성능:
    - 여기서는 demo가 이미 '배치/채널/간섭' 케이스를 종합 실행하므로,
      주요 케이스에서 성능 지표가 '말이 되는 값' + '예상되는 성공/실패 패턴'인지 확인.
    """
    base = cases.get("Wi-Fi OFF baseline")
    inband = cases.get("Wi-Fi ON in-band (Δf~0.4MHz)")
    oob60 = cases.get("Wi-Fi ON out-of-band, stop=60dB")

    if not base or not inband or not oob60:
        return CheckResult(
            name="5) 배치 입력 시 MMS 성능(End-to-End)",
            status="FAIL",
            details="baseline/in-band/oob60 케이스가 모두 필요합니다.",
        )

    probs: List[str] = []

    # baseline은 성공해야 함
    if base.ranging_fail is None or base.ranging_fail > 0.0:
        probs.append(f"baseline: ranging_fail expected 0.0, got {base.ranging_fail}")
    if base.rmse_valid_m is None:
        probs.append("baseline: RMSE(valid) is None (N/A?)")
    else:
        if base.rmse_valid_m > 1.0:
            probs.append(f"baseline: RMSE(valid) too large: {base.rmse_valid_m:.3f} m")

    # in-band는 (현재 시뮬 출력 기준) 실패가 정상(간섭 주입 확인)
    if inband.ranging_fail is not None and inband.ranging_fail < 0.9:
        probs.append(f"in-band: expected high ranging_fail (~1), got {inband.ranging_fail}")
    if inband.ber is not None and inband.ber < 0.1:
        probs.append(f"in-band: expected BER high, got {inband.ber}")

    # oob60은 다시 성공해야 함
    if oob60.ranging_fail is None or oob60.ranging_fail > 0.0:
        probs.append(f"oob60: ranging_fail expected 0.0, got {oob60.ranging_fail}")

    if probs:
        return CheckResult(
            name="5) 배치 입력 시 MMS 성능(End-to-End)",
            status="WARN",
            details="; ".join(probs),
        )

    return CheckResult(
        name="5) 배치 입력 시 MMS 성능(End-to-End)",
        status="PASS",
        details="baseline/oob60 성공 + in-band 실패 패턴이 기대대로(간섭/필터/시나리오가 정상 동작)",
    )


# -----------------------------
# Runner
# -----------------------------
def run_demo(repo_root: Path, demo_relpath: str, extra_env: Dict[str, str], timeout_s: int) -> str:
    demo_path = (repo_root / demo_relpath).resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo script not found: {demo_path}")

    env = os.environ.copy()
    env.update(extra_env)

    # repo 루트를 PYTHONPATH에 추가 (패키지 import 안정화)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, str(demo_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        # 실패해도 stdout/stderr를 최대한 보여주기 위해 예외에 같이 넣음
        raise RuntimeError(
            f"Demo returned non-zero exit code: {proc.returncode}\n"
            f"--- STDOUT ---\n{stdout[-4000:]}\n"
            f"--- STDERR ---\n{stderr[-4000:]}\n"
        )

    # stderr가 의미있는 로그일 수도 있어서 합쳐줌
    combined = stdout + ("\n" + stderr if stderr.strip() else "")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-stack MMS smoke checks")
    ap.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="레포 루트(기본: 현재 디렉토리). 예: /workspaces/simulation",
    )
    ap.add_argument(
        "--demo",
        type=str,
        default="simulation/mms/full_stack_mms_demo.py",
        help="full stack demo 스크립트 상대경로",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="demo 실행 타임아웃(초)",
    )
    ap.add_argument(
        "--cca-ed-threshold-dbm",
        type=float,
        default=-62.0,
        help="ED 기반 CCA 임계값(dBm). 스모크 체크용(기본 -62 dBm)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="추가 환경변수 KEY=VALUE (여러 번 사용 가능)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()

    extra_env: Dict[str, str] = {}
    for kv in args.env:
        if "=" not in kv:
            raise ValueError(f"--env must be KEY=VALUE, got: {kv}")
        k, v = kv.split("=", 1)
        extra_env[k.strip()] = v.strip()

    print(f"[smoke] repo_root={repo_root}")
    print(f"[smoke] demo={args.demo}")
    if extra_env:
        print(f"[smoke] extra env={extra_env}")

    try:
        out = run_demo(repo_root=repo_root, demo_relpath=args.demo, extra_env=extra_env, timeout_s=args.timeout)
    except Exception as e:
        print(f"[smoke][FAIL] demo run failed:\n{e}")
        return 2

    flow, cases = parse_cases(out)

    # Print quick parsed summary
    print("\n[smoke] Parsed cases:")
    for k in sorted(cases.keys()):
        cm = cases[k]
        print(
            f"  - {k}: control_ok={cm.nb_control_ok}, BER={cm.ber}, FER={cm.fer}, "
            f"RMSE(valid)={cm.rmse_valid_m}, Bias(valid)={cm.bias_valid_m}, "
            f"RangingFail={cm.ranging_fail}, SNR={cm.snr_db}"
        )

    results: List[CheckResult] = []
    results.append(check_1_protocol(flow, cases))
    results.append(check_2_uwb_nb_waveforms(cases, repo_root))
    results.append(check_3_wifi_waveform(cases))
    results.append(check_4_wifi_cca(cases, ed_threshold_dbm=args.cca_ed_threshold_dbm))
    results.append(check_5_end_to_end_performance(cases))

    print("\n[smoke] Results:")
    fail = 0
    warn = 0
    for r in results:
        print(f"  [{r.status}] {r.name}")
        if r.details:
            print(f"         - {r.details}")
        if r.status == "FAIL":
            fail += 1
        elif r.status == "WARN":
            warn += 1

    if fail > 0:
        print(f"\n[smoke] FAIL ({fail} failing checks, {warn} warnings)")
        return 1

    print(f"\n[smoke] PASS (warnings={warn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
