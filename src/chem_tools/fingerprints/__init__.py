"""Fingerprint generation."""

from __future__ import annotations

from chem_tools.external.map4 import MAP4Calculator
from chem_tools.fingerprints.api import FPInfo, get_fp_gen
from chem_tools.fingerprints.blockwise import blockwise_array
from chem_tools.fingerprints.rdkit import (
    AtomPairGenerator,
    AtomPairNumpyGenerator,
    MorganGenerator,
    MorganNumpyGenerator,
)
from chem_tools.fingerprints.utils import (
    FingerprintGenerator,
    GenericRDKitGenerator,
    GenericRDKitNumpyGenerator,
    convert_bv_to_np,
    convert_np_to_bv,
)

__all__ = [
    "AtomPairGenerator",
    "AtomPairNumpyGenerator",
    "FPInfo",
    "FingerprintGenerator",
    "GenericRDKitGenerator",
    "GenericRDKitNumpyGenerator",
    "MAP4Calculator",
    "MorganGenerator",
    "MorganNumpyGenerator",
    "blockwise_array",
    "convert_bv_to_np",
    "convert_np_to_bv",
    "get_fp_gen",
]
