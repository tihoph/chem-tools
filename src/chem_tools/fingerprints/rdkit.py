"""FingerprintGenerators for RDKit's Morgan and AtomPair fingerprints."""

from __future__ import annotations

from rdkit.Chem import rdFingerprintGenerator as rdFPGen

from chem_tools.fingerprints.utils import (
    GenericRDKitGenerator,
    GenericRDKitNumpyGenerator,
)


class MorganGenerator(GenericRDKitGenerator):
    """RDKit Morgan fingerprint (ECFP) generator."""

    def __init__(
        self, radius: int = 2, n_bits: int = 2048, features: bool = False
    ) -> None:
        """Initialize the MorganGenerator.

        Args:
            radius: Maximum radius for atom pairs. Defaults to 2.
            n_bits: Number of dimensions for the fingerprint. Defaults to 2048.
            features: Whether to use feature fingerprints. Defaults to False.
        """
        name = "radius={}, n_bits={}, features={}"
        self._name = name.format(radius, n_bits, features)
        invar = rdFPGen.GetMorganFeatureAtomInvGen() if features else None
        self._fpgen = rdFPGen.GetMorganGenerator(  # type: ignore[misc]
            radius=radius, fpSize=n_bits, atomInvariantsGenerator=invar
        )
        super().__init__()


class MorganNumpyGenerator(GenericRDKitNumpyGenerator):
    """RDKit Morgan fingerprint (ECFP) generator using numpy arrays."""

    def __init__(
        self, radius: int = 2, n_bits: int = 2048, features: bool = False
    ) -> None:
        """Initialize the MorganNumpyGenerator.

        Args:
            radius: Maximum radius for atom pairs. Defaults to 2.
            n_bits: Number of dimensions for the fingerprint. Defaults to 2048.
            features: Whether to use feature fingerprints. Defaults to False.
        """
        name = "radius={}, n_bits={}, features={}"
        self._name = name.format(radius, n_bits, features)
        self._subgen = MorganGenerator(  # type: ignore[misc]
            radius=radius, n_bits=n_bits, features=features
        )
        super().__init__()


class AtomPairGenerator(GenericRDKitGenerator):
    """RDKit Atom Pair fingerprint generator."""

    def __init__(self, n_bits: int = 2048) -> None:
        """Initialize the AtomPairGenerator.

        Args:
            n_bits: Number of dimensions for the fingerprint. Defaults to 2048.
        """
        self._name = f"n_bits={n_bits}"
        self._fpgen = rdFPGen.GetAtomPairGenerator(fpSize=n_bits)  # type: ignore[misc]
        super().__init__()


class AtomPairNumpyGenerator(GenericRDKitNumpyGenerator):
    """RDKit Atom Pair fingerprint generator using numpy arrays."""

    def __init__(self, n_bits: int = 2048) -> None:
        """Initialize the AtomPairNumpyGenerator.

        Args:
            n_bits: Number of dimensions for the fingerprint. Defaults to 2048.
        """
        self._name = f"n_bits={n_bits}"
        self._subgen = AtomPairGenerator(n_bits=n_bits)  # type: ignore[misc]
        super().__init__()


__all__ = [
    "AtomPairGenerator",
    "AtomPairNumpyGenerator",
    "MorganGenerator",
    "MorganNumpyGenerator",
]
