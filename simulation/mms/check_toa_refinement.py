from __future__ import annotations

import sys
from pathlib import Path

try:
    from simulation.mms.performance import simulate_mms_performance
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.mms.performance import simulate_mms_performance


def run_check() -> None:
    d = 20.0
    seed = 1
    print("=== ToA Refinement Check ===")
    print("Case A: sanity (no multipath/ppm), Wi-Fi OFF")
    a = simulate_mms_performance(
        distances_m=(d,),
        n_trials=80,
        seed=seed,
        wifi_interference_on=False,
        baseline_sanity_mode=True,
        save_psd=False,
        debug_first_trial=False,
        toa_refine_method="fft_upsample",
        corr_upsample=8,
        corr_win=64,
        first_path=False,
        toa_calibration_distance_m=d,
    )[0]
    print(
        f"  RMSE(valid)={a['ranging_rmse_m']:.6f} m, "
        f"FER={a['fer']:.3f}, RangingFail={a['ranging_fail_rate']:.3f}"
    )

    print("Case B: multipath ON, Wi-Fi OFF, upsample sweep")
    print("  (Calibration disabled here to expose quantization/refinement trend)")
    ups = [1, 8, 16]
    rmses = []
    for u in ups:
        r = simulate_mms_performance(
            distances_m=(d,),
            n_trials=60,
            seed=seed,
            wifi_interference_on=False,
            baseline_sanity_mode=False,
            save_psd=False,
            debug_first_trial=False,
            toa_refine_method="fft_upsample",
            corr_upsample=u,
            corr_win=64,
            first_path=False,
            toa_calibration_distance_m=d,
            enable_toa_calibration=False,
            auto_calibrate=False,
        )[0]
        rmses.append(float(r["ranging_rmse_m"]))
        print(
            f"  up={u:2d} -> RMSE(valid)={r['ranging_rmse_m']:.6f} m, "
            f"FER={r['fer']:.3f}, RangingFail={r['ranging_fail_rate']:.3f}"
        )

    if rmses[-1] <= rmses[0]:
        print("  PASS: higher corr_upsample did not worsen RMSE.")
    else:
        print("  WARN: RMSE increased with higher corr_upsample; tune first_path/corr_win.")


if __name__ == "__main__":
    run_check()
