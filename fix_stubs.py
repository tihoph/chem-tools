#!/usr/bin/env python
"""Fix RDKit stubs."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def fix_stubs() -> None:
    """Fix RDKit stubs."""
    rdkit_path = Path(sysconfig.get_paths()["purelib"]) / "rdkit-stubs"
    if not rdkit_path.exists():
        raise FileNotFoundError(f"RDKit stubs not found at {rdkit_path}")

    def _replace(file: Path, old: bytes, new: bytes) -> None:
        """Replace old bytes with new bytes in file."""
        file.write_bytes(file.read_bytes().replace(old, new))

    for file in rdkit_path.rglob("*.pyi"):
        _replace(file, b"\x00", b"")

    rd_mol_descriptors = rdkit_path / "Chem" / "rdMolDescriptors.pyi"
    _replace(rd_mol_descriptors, b"(self, self", b"(self_, self")
    rd_rgroup_decomposition = rdkit_path / "Chem" / "rdRGroupDecomposition.pyi"
    _replace(rd_rgroup_decomposition, b"None:", b"None_:")


if __name__ == "__main__":
    fix_stubs()
