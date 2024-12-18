"""Randomized SMILES strings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from typing_extensions import Unpack

from chem_tools._conversion import mol_from_smi, mol_to_smi
from chem_tools._parallel import auto_parallel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chem_tools._typing import NDArray, StrArray

logger = logging.getLogger(__package__)


def _has_hydrogens(mol: Chem.Mol) -> bool:
    """Check if a molecule has no hydrogens."""
    return mol.GetNumAtoms() != mol.GetNumHeavyAtoms()


def _randomly_renumber_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Randomly renumber heavy atoms in a molecule."""
    # Randomize the atom order
    atom_order: list[int] = np.random.permutation(mol.GetNumHeavyAtoms()).tolist()

    return Chem.RenumberAtoms(mol, newOrder=atom_order)


def randomize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Randomizes the atom order of a molecule.

    If the molecule has hydrogens, they are removed before randomization.

    Args:
        mol: The molecule

    Returns:
        The randomized molecule
    """
    orig_mol = mol

    if _has_hydrogens(mol):
        mol = Chem.RemoveHs(mol)

        # isomeric hydrogen is not removed by RemoveHs
        # so we need the more forceful RemoveAllHs
        if _has_hydrogens(mol):
            mol = Chem.RemoveAllHs(mol)
            message = "Removed all Hs to randomize molecule: %s"
        else:
            message = "Removed Hs to randomize molecule: %s"

        logger.warning(message, Chem.MolToSmiles(orig_mol, canonical=False))

    return _randomly_renumber_atoms(mol)


def randomize_smi(smi: str) -> str:
    """Randomizes the atom order of a SMILES string.

    If the molecule has hydrogens, they are removed before randomization.

    Args:
        smi: SMILES string

    Returns:
        Randomized SMILES string
    """
    mol = mol_from_smi(smi, assert_valid=True)

    random_mol = randomize_mol(mol)
    # Return the randomized SMILES string
    return mol_to_smi(random_mol, canonical=False, isomeric=True)


def randomize_many(
    smis: Sequence[str] | StrArray, n_jobs: int, verbose: int = 0
) -> list[str]:
    """Randomize SMILES strings in parallel.

    Args:
        smis: The SMILES strings to randomize.
        n_jobs: The number of jobs to use for parallelization.
        verbose: Whether to show progress bars. Defaults to 0.

    Returns:
        The randomized SMILES strings.
    """
    return auto_parallel(
        randomize_smi, smis, n_jobs, desc="Creating new smiles...", verbose=verbose
    )

    # Encode randomized SMILES


def augment_smis(
    smis: str | Sequence[str] | StrArray, n_repr: int, n_jobs: int, verbose: int = 0
) -> list[str]:
    """Augment SMILES strings by randomizing them N times.

    Args:
        smis: The SMILES strings to randomize.
        n_repr: The number of times to randomize each SMILES string.
        n_jobs: The number of jobs to use for parallelization.
        verbose: Whether to show progress bars. Defaults to 0.

    Returns:
        The randomized SMILES strings.
    """
    smis = np.array(smis, dtype=np.str_)
    smis = np.repeat(smis, n_repr)

    return randomize_many(smis, n_jobs, verbose)


def augment_smis_plus(
    smis: Sequence[str] | StrArray,
    *args: Sequence | NDArray,
    n_repr: int,
    n_jobs: int,
    verbose: int = 0,
) -> tuple[list[str], Unpack[tuple[NDArray, ...]]]:
    """Augment SMILES strings and adjust additional arguments.

    - [S0, S1, S2] -> [S0_0, S0_1, ..., S1_0, S1_1, ..., S2_0, S2_1, ...]

    Args:
        smis: The SMILES strings to randomize.
        *args: Additional arguments to augment.
        n_repr: The number of times to randomize each SMILES string.
        n_jobs: The number of jobs to use for parallelization.
        verbose: Whether to show progress bars. Defaults to 0.

    Returns:
        A tuple with the randomized SMILES strings
        and the augmented additional arguments.
    """
    rnd_smis = augment_smis(smis, n_repr, n_jobs, verbose)
    additional: list[NDArray] = [np.repeat(arg, n_repr) for arg in args]

    if not all(len(x) == len(rnd_smis) for x in additional):
        raise ValueError("All arguments must have the same length.")

    return (rnd_smis, *additional)


def mol_to_random_smi(mol: Chem.Mol) -> str:
    """Convert a molecule to a randomized SMILES string.

    Args:
        mol: The molecule to randomize

    Returns:
        A randomized SMILES representation of the molecule
    """
    return mol_to_smi(randomize_mol(mol), canonical=False)


__all__ = [
    "augment_smis",
    "augment_smis_plus",
    "mol_to_random_smi",
    "randomize_many",
    "randomize_mol",
    "randomize_smi",
]
