"""This module contains functions to calculate molecular descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit.Chem import QED, rdMolDescriptors

from chem_tools.descriptors.rings import get_largest_ring_size, get_ring_systems
from chem_tools.descriptors.similarity import (
    calc_internal_similarity,
    calc_internal_similarity_matrix,
)
from chem_tools.external import sascorer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem


def calc_qed(mol: Chem.Mol) -> float:
    """Calculate the quantitative estimate of drug-likeness of a molecule.

    Args:
        mol: A molecule.

    Returns:
        The quantitative estimate of drug-likeness of the molecule.
    """
    return QED.qed(mol)  # type:ignore[no-untyped-call,no-any-return]


def calc_sa_score(mol: Chem.Mol) -> float:
    """Calculate the synthetic accessibility score of a molecule.

    Args:
        mol: A molecule.

    Returns:
        The synthetic accessibility score of the molecule.
    """
    return sascorer.calculateScore(mol)


def calc_sa_scores(mols: Sequence[Chem.Mol], n_jobs: int = 16) -> list[float]:
    """Calculate the synthetic accessibility scores of a sequence of molecules.

    Args:
        mols: A sequence of molecules.
        n_jobs: The number of threads to use. Defaults to 16.

    Returns:
        A list of synthetic accessibility scores.
    """
    return [
        score for _, score in sascorer.processMols(mols, numThreads=n_jobs, verbose=0)
    ]


def calc_mol_weight(mol: Chem.Mol) -> float:
    """Calculate the molecular weight of a molecule.

    Args:
        mol: A molecule.

    Returns:
        The molecular weight of the molecule.
    """
    return rdMolDescriptors.CalcExactMolWt(mol)


__all__ = [
    "calc_internal_similarity",
    "calc_internal_similarity_matrix",
    "calc_mol_weight",
    "calc_qed",
    "calc_sa_score",
    "calc_sa_scores",
    "get_largest_ring_size",
    "get_ring_systems",
]
