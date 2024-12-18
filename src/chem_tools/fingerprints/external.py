"""FingerprintGenerators for external MAP4 and MHFP libraries."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from mhfp.encoder import MHFPEncoder
from typing_extensions import override

from chem_tools._parallel import auto_parallel
from chem_tools._typing import Float32Array, Int64Array
from chem_tools.external.map4 import MAP4Calculator
from chem_tools.fingerprints.utils import FingerprintGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rdkit import Chem

    from chem_tools.external import _tmap as tm
else:
    try:
        import tmap as tm
    except ImportError:
        import chem_tools.external._tmap as tm

Minhash = tm.Minhash()


class MAP4Generator(FingerprintGenerator[tm.VectorUint]):
    """MinHashed Atom Pair fingerprint generator.

    Only useable with the `tmap` library.
    """

    def __init__(self, radius: int = 2, dims: int = 1024, folded: bool = False) -> None:
        """Initialize the MAP4Generator.

        Args:
            radius: Maximum radius for atom pairs. Defaults to 2.
            dims: Number of dimensions for the fingerprint. Defaults to 1024.
            folded: Whether the fingerprint is folded. Defaults to False.
        """
        self._name = f"radius={radius}, dims={dims}"
        if folded:
            logging.warning(
                "For folded fingerprints, use the MHFPGenerator for similarity"
                " calculation."
            )
        self._calc = MAP4Calculator(dimensions=dims, radius=radius, is_folded=folded)

    @override
    def get(self, mol: Chem.Mol) -> tm.VectorUint:
        return self._calc.calculate(mol)  # type: ignore[return-value]

    @override
    def get_many(self, mols: Iterable[Chem.Mol], n_jobs: int = 1) -> Sequence:
        if n_jobs != 1:
            raise ValueError("n_jobs must be 1 for MAP4Generator")
        mols = list(mols)
        return self._calc.calculate_many(mols)

    @override
    @classmethod
    def similarity(cls, fp1: tm.VectorUint, fp2: tm.VectorUint) -> float:
        return 1 - Minhash.get_distance(fp1, fp2)

    @override
    @classmethod
    def bulk_similarity(
        cls, fp1: tm.VectorUint, fps: Sequence[tm.VectorUint]
    ) -> list[float]:
        return [1 - Minhash.get_distance(fp1, fp) for fp in fps]

    @override
    @classmethod
    def matrix_similarity(
        cls,
        fps1: Sequence[tm.VectorUint],
        fps2: Sequence[tm.VectorUint],
        max_size: int = 1000,
        n_jobs: int = 1,
    ) -> Float32Array:
        """Calculate the similarity between two sequences of fingerprints.

        For faster calculation, convert :class:`tm.VectorUint` to :class:`np.ndarray`
        and use :meth:`MHFPGenerator.matrix_similarity` with parallel processing.
        """
        if n_jobs != 1:
            raise ValueError("n_jobs must be 1 for MAP4Generator")
        return super().matrix_similarity(fps1, fps2, max_size=max_size, n_jobs=n_jobs)


class MHFPGenerator(FingerprintGenerator[Int64Array]):
    """MinHashed Fingerprint generator."""

    def __init__(self, radius: int = 3, min_radius: int = 0, dim: int = 1024) -> None:
        """Initialize the MHFPGenerator.

        Args:
            radius: Maximum radius for atom pairs. Defaults to 3.
            min_radius: Minimum radius for atom pairs. Defaults to 0.
            dim: Number of dimensions for the fingerprint. Defaults to 1024.
        """
        self._name = f"radius={radius}, min_radius={min_radius}, dim={dim}"
        self._encoder = MHFPEncoder(dim)
        self._get = partial(
            self._encoder.encode_mol, radius=radius, min_radius=min_radius
        )

    @override
    def get(self, mol: Chem.Mol) -> Int64Array:
        return self._get(mol)  # type: ignore[no-any-return]

    @override
    def get_many(self, mols: Iterable[Chem.Mol], n_jobs: int = 1) -> list[Int64Array]:
        return auto_parallel(self._get, mols, n_jobs=n_jobs)

    @override
    @classmethod
    def similarity(cls, fp1: Int64Array, fp2: Int64Array) -> float:
        num = np.count_nonzero(fp1 == fp2, axis=-1)
        return num / fp1.shape[-1]  # type: ignore[no-any-return]

    @override
    @classmethod
    def bulk_similarity(  # type: ignore[override]
        cls, fp1: Int64Array, fps: Int64Array
    ) -> Float32Array:
        return cls.similarity(fp1, fps).astype(np.float32)  # type: ignore[attr-defined, no-any-return]


__all__ = ["MAP4Generator", "MHFPGenerator"]
