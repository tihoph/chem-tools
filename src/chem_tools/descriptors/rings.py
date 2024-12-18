"""Functions which calculate ring properties of molecules."""

# adapted from:
# https://chemistry.stackexchange.com/questions/150736/how-to-find-the-largest-cyclic-substructure-with-rdkit # noqa: E501
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem


def get_ring_systems(mol: Chem.Mol, incl_spiro: bool = False) -> set[frozenset[int]]:
    """Get the ring systems of a molecule.

    Args:
        mol: A molecule.
        incl_spiro: Whether to include spiro systems. Defaults to False.

    Returns:
        A list of sets of atom indices in each ring system.
    """
    ri = mol.GetRingInfo()
    systems: list[set[int]] = []
    for ring in ri.AtomRings():
        ring_ats = set(ring)
        n_systems: list[set[int]] = []
        for system in systems:
            n_in_common = len(ring_ats.intersection(system))
            if n_in_common and (incl_spiro or n_in_common > 1):
                ring_ats = ring_ats.union(system)
            else:
                n_systems.append(system)
        n_systems.append(ring_ats)
        systems = n_systems
    return {frozenset(x) for x in systems}


def get_largest_ring_size(mol: Chem.Mol, incl_spiro: bool = False) -> int:
    """Get the size of the largest ring system.

    Args:
        mol: A molecule.
        incl_spiro: Whether to include spiro systems. Defaults to False.

    Returns:
        The size of the largest ring system.
    """
    return max((len(x) for x in get_ring_systems(mol, incl_spiro)), default=0)


__all__ = ["get_largest_ring_size", "get_ring_systems"]
