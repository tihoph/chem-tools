"""Wide range of external functionality functions and scripts."""

from __future__ import annotations

from chem_tools.external.fcd import calc_fcd_stats
from chem_tools.external.sascorer import calculateScore, processMols
from chem_tools.external.tmap_plot import (
    create_lsh_forest,
    create_tmap,
    plot_faerun,
    plot_matplotlib,
)

# MAP4Calculator is exposed in chem_tools.fingerprints.external

__all__ = [
    "calc_fcd_stats",
    "calculateScore",
    "create_lsh_forest",
    "create_tmap",
    "plot_faerun",
    "plot_matplotlib",
    "processMols",
]
