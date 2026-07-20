"""Task 10 controlled experiment suite for identifiability-aware source apportionment.

One shared New Delhi base platform (`nd_platform.py`) feeds ten one-factor
controlled experiments plus an observed-New-Delhi study mode (`experiments.py`).
`run_experiment.py` executes a single configured experiment reproducibly and
saves artifacts; `summarize_results.py` rolls runs up into Task 11 tables.
"""

from experiments.iasa_pol.nd_platform import (
    Platform,
    PlatformConfig,
    build_platform,
    compact_source,
    make_wind,
    sensor_layout,
)

__all__ = [
    "Platform",
    "PlatformConfig",
    "build_platform",
    "compact_source",
    "make_wind",
    "sensor_layout",
]
