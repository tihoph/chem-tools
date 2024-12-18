"""Provide access to fingerprint generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from chem_tools.fingerprints.external import MAP4Generator, MHFPGenerator
from chem_tools.fingerprints.rdkit import AtomPairGenerator, MorganGenerator

if TYPE_CHECKING:
    from chem_tools.fingerprints.utils import FingerprintGenerator


class FPInfo(TypedDict, total=False):
    """Dictionary with Fingerprint information."""

    name: Literal["atom_pair", "morgan", "map", "mhfp"]
    """The name of the fingerprint type."""
    radius: int
    """The radius of the fingerprint. (disabled for atom_pair)"""
    dims: int
    """The dimension of the fingerprint."""


def get_fp_gen(info: FPInfo) -> tuple[FingerprintGenerator, str]:
    """Get the fingerprint generator and formatted name based on the input name.

    Example:
        >>> info = {"name": "morgan", "radius": 3, "dims": 2048}
        >>> get_fp_gen(info)
        (MorganGenerator(radius=3, dims=2048), 'ECFP6$_{2048}$')

    Args:
        info (FPInfo): The fingerprint information.

    Returns:
        tuple: A tuple containing:
            - FingerprintGenerator: The appropriate fingerprint generator object.
            - str: A formatted string representation of the fingerprint name.

    Raises:
        ValueError: If no or unknown fingerprint name is provided.
        ValueError: If radius is defined for atom_pair.
    """
    name = info.get("name", None)
    radius = info.get("radius", None)
    dims = info.get("dims", None)
    if name is None:
        raise ValueError("No fingerprint name provided")
    if name == "atom_pair":
        if radius is not None:
            raise ValueError("Radius is not supported for atom_pair")
        dims = dims if dims is not None else 2048
        return (AtomPairGenerator(dims), f"APFP${{{dims}}}$")
    if name == "morgan":
        radius = radius if radius is not None else 3
        dims = dims if dims is not None else 2048
        return (MorganGenerator(radius, dims), f"ECFP{2*radius}$_{{{dims}}}$")
    if name == "map":
        radius = radius if radius is not None else 2
        dims = dims if dims is not None else 1024
        return (MAP4Generator(radius, dims), f"MAP{2*radius}$_{{{dims}}}$")
    if name == "mhfp":
        radius = radius if radius is not None else 3
        dims = dims if dims is not None else 1024
        return (MHFPGenerator(radius, 0, dims), f"MHFP{2*radius}$_{{{dims}}}$")
    raise ValueError(f"Unknown fingerprint: {name}")


__all__ = ["FPInfo", "get_fp_gen"]
