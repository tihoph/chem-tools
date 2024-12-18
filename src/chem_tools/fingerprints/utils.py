# %%
"""Fingerprint generators for molecular fingerprints."""

# ruff: noqa: SLF001
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from typing_extensions import override

from chem_tools._typing import UInt8Array
from chem_tools.fingerprints.blockwise import blockwise_array

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rdkit import Chem

    from chem_tools._typing import Float32Array


T = TypeVar("T")


class FingerprintGenerator(Generic[T]):
    """Base class for fingerprint generators.

    Usage:
        Implement a subclass by creating a subclass of this class
        and providing the abstract methods and a `_name` attribute.

    Abstract Methods:
        get: Get the fingerprint of a molecule
        get_many: Get the fingerprints of multiple molecules
        similarity: Calculate the similarity between two fingerprints
        bulk_similarity: Calculate the similarity between
            a single and multiple fingerprints

    Methods:
        matrix_similarity: Calculate the similarity between
            two sequences of fingerprints
    """

    _name: str

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    @abstractmethod
    def get(self, mol: Chem.Mol) -> T:
        """Get the fingerprint of a molecule."""

    @abstractmethod
    def get_many(self, mols: Iterable[Chem.Mol], n_jobs: int = 1) -> Sequence[T]:
        """Get the fingerprints of multiple molecules."""

    @classmethod
    @abstractmethod
    def similarity(cls, fp1: T, fp2: T) -> float:
        """Calculate the similarity between two fingerprints.

        For more sophisticated methods,
        consider using :module:`chem.similarity`.
        """

    @classmethod
    @abstractmethod
    def bulk_similarity(cls, fp1: T, fps: Sequence[T]) -> Sequence[float]:
        """Calculate the similarity between a single and multiple fingerprints."""

    @classmethod
    def matrix_similarity(
        cls, fps1: Sequence[T], fps2: Sequence[T], max_size: int = 1000, n_jobs: int = 1
    ) -> Float32Array:
        """Calculate the similarity between two sequences of fingerprints."""

        def _matrix_similarity(fps1: Sequence[T], fps2: Sequence[T]) -> Float32Array:
            return np.stack(
                [
                    np.array(cls.bulk_similarity(fp1, fps2), dtype=np.float32)
                    for fp1 in fps1
                ]
            )

        return blockwise_array(
            fps1, fps2, _matrix_similarity, max_size=max_size, n_jobs=n_jobs
        ).astype(np.float32)


class GenericRDKitGenerator(FingerprintGenerator[DataStructs.ExplicitBitVect]):
    """Base class for RDKit fingerprint generators.

    Usage:
        Implement a subclass by creating a subclass of this class
        and providing the `_fpgen` attribute for a
        `rdFPGen.FingeprintGenerator64` object.
    """

    @override
    def get(self, mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
        return self._fpgen.GetFingerprint(mol)  # type: ignore[no-any-return]

    @override
    def get_many(
        self, mols: Iterable[Chem.Mol], n_jobs: int = 16
    ) -> tuple[DataStructs.ExplicitBitVect]:
        mols = list(mols)
        return self._fpgen.GetFingerprints(mols, numThreads=n_jobs)

    @override
    @classmethod
    def similarity(
        cls, fp1: DataStructs.ExplicitBitVect, fp2: DataStructs.ExplicitBitVect
    ) -> float:
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @override
    @classmethod
    def bulk_similarity(
        cls,
        fp1: DataStructs.ExplicitBitVect,
        fps: Sequence[DataStructs.ExplicitBitVect],
    ) -> list[float]:
        return DataStructs.BulkTanimotoSimilarity(fp1, fps)

    if TYPE_CHECKING:

        @property
        def _fpgen(self) -> rdFPGen.FingeprintGenerator64: ...


class GenericRDKitNumpyGenerator(FingerprintGenerator[UInt8Array]):
    """Base class for RDKit fingerprint generators.

    Uses `FingeprintGenerator64.GetFingerprintAsNumPy`.

    Usage:
        Implement a subclass by creating first a subclass of
        `GenericRDKitGenerator` and subclass this class
        and hook the subclassed generator to the `_subgen`
        attribute.
    """

    @override
    def get(self, mol: Chem.Mol) -> UInt8Array:
        fp = self._subgen._fpgen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.uint8)  # type: ignore[no-any-return]

    @override
    def get_many(  # type: ignore[override]
        self, mols: Iterable[Chem.Mol], n_jobs: int = 16
    ) -> UInt8Array:
        fps = self._subgen._fpgen.GetFingerprints(mols, numThreads=n_jobs)
        return np.stack([convert_bv_to_np(fp) for fp in fps])

    @override
    @classmethod
    def similarity(cls, fp1: UInt8Array, fp2: UInt8Array) -> float:
        num = np.sum(fp1 & fp2, axis=-1)
        den = np.sum(fp1 | fp2, axis=-1)
        return num / den  # type: ignore[no-any-return]

    @override
    @classmethod
    def bulk_similarity(  # type: ignore[override]
        cls, fp1: UInt8Array, fps: UInt8Array
    ) -> Float32Array:
        return cls.similarity(fp1, fps)  # type: ignore[return-value]

    if TYPE_CHECKING:

        @property
        def _subgen(self) -> GenericRDKitGenerator: ...


def convert_bv_to_np(fp: DataStructs.ExplicitBitVect) -> UInt8Array:
    """Convert a `DataStructs.ExplicitBitVect` to a `np.ndarray`.

    Args:
        fp: Fingerprint to convert

    Returns:
    Bit vector as numpy array
    """
    fp_: bytes = DataStructs.BitVectToBinaryText(fp)
    return np.unpackbits(np.frombuffer(fp_, dtype=np.uint8), bitorder="little")


def convert_np_to_bv(fp: UInt8Array) -> DataStructs.ExplicitBitVect:
    """Convert a `np.ndarray` to a `DataStructs.ExplicitBitVect`.

    Args:
        fp: Fingerprint to convert

    Returns:
        Bit vector as ExplicitBitVect
    """
    new = DataStructs.ExplicitBitVect(len(fp))
    new.SetBitsFromList(fp.nonzero()[0].tolist())
    return new


__all__ = [
    "FingerprintGenerator",
    "GenericRDKitGenerator",
    "GenericRDKitNumpyGenerator",
    "convert_bv_to_np",
    "convert_np_to_bv",
]
