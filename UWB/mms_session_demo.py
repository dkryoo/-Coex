"""UWB compatibility wrapper for simulation-level MMS session demo."""

from simulation.mms.session import *  # noqa: F401,F403


if __name__ == "__main__":
    run_demo(distance_m=300.0)
