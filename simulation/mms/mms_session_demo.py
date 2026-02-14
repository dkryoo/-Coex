"""MMS session demo entrypoints for protocol-level simulation."""

from .session import *  # noqa: F401,F403


if __name__ == "__main__":
    run_demo(distance_m=300.0)
