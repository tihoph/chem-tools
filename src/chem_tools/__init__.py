"""Wrapper functions for RDKit to handle molecules and fingerprints."""

from __future__ import annotations

from chem_tools import external, filters, fingerprints, randomized
from chem_tools._conversion import (
    canonicalize,
    check_mol,
    get_processed_smi,
    get_scaf_smi,
    mol_from_smi,
    mol_to_smi,
)
from chem_tools._parallel import auto_parallel

__all__ = [
    "auto_parallel",
    "canonicalize",
    "check_mol",
    "external",
    "filters",
    "fingerprints",
    "get_processed_smi",
    "get_scaf_smi",
    "mol_from_smi",
    "mol_to_smi",
    "randomized",
]
