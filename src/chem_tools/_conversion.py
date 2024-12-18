"""Conversion functions for SMILES strings and fingerprints."""

from __future__ import annotations

import base64
import logging
from typing import Any, Literal, overload

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.rdBase import DisableLog


class RDKitError(ValueError):
    """RDKit processing failed."""

    def __init__(self, mol: Chem.Mol, message: str) -> None:
        """Initialize the error.

        If conversion to SMILES fails, the molecule is represented as a binary.

        Args:
            mol: The molecule that failed to process
            message: The error message
        """
        try:
            representation = Chem.MolToSmiles(mol, canonical=False)
        except ValueError:
            representation = base64.b64encode(mol.ToBinary()).decode("utf-8")
        super().__init__(f"{message} {representation}")


class InvalidSmilesError(ValueError):
    """Conversion of SMILES to RDKit molecule failed."""

    def __init__(self, smi: str) -> None:
        """Initialize the error.

        Args:
            smi: The invalid SMILES string
        """
        # message adapted from float(invalid) copied
        super().__init__(f"Could not convert SMILES to molecule: {smi}")


def canonicalize(smi: str, isomeric: bool = True) -> str:
    """Canonicalize a SMILES string.

    Args:
        smi: SMILES string
        isomeric: toggles isomeric SMILES. Defaults to True.

    Returns:
        Canonicalized SMILES string
    """
    mol = mol_from_smi(smi, assert_valid=True)
    return mol_to_smi(mol, isomeric=isomeric, canonical=True)


def check_mol(mol: Chem.Mol | None) -> bool:
    """Check if a molecule is valid.

    Includes sanitization and checks for atom number.

    Args:
        mol: A potential molecule.

    Returns:
        Whether the molecule is valid.
    """
    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
    except ValueError:  # pragma: no cover # TODO: find smiles that fail sanitization
        logging.warning("Molecule failed sanitization: %s", mol)
        return False

    return mol.GetNumAtoms() > 0


@overload
def get_processed_smi(  # pragma: no cover
    smi: str, assert_valid: Literal[False] = False, remove_stereo: bool = ...
) -> tuple[str, Chem.Mol] | None: ...


@overload
def get_processed_smi(  # pragma: no cover
    smi: str, assert_valid: Literal[True], remove_stereo: bool = ...
) -> tuple[str, Chem.Mol]: ...


@overload
def get_processed_smi(  # pragma: no cover
    smi: str, assert_valid: bool = False, remove_stereo: bool = ...
) -> tuple[str, Chem.Mol] | None: ...


def get_processed_smi(
    smi: str, assert_valid: bool = False, remove_stereo: bool = False
) -> tuple[str, Chem.Mol] | None:
    """Canonicalizes and converts a SMILES string.

    Args:
        smi: SMILES string
        remove_stereo: toggles removing stereo information. Defaults to False.
        assert_valid: toggles raising an error if invalid. Defaults to False.

    Returns:
        Processed SMILES string
    """
    mol = mol_from_smi(smi, assert_valid=assert_valid)

    if mol is None:
        return None

    if remove_stereo:
        Chem.RemoveStereochemistry(mol)

    return mol_to_smi(mol), mol


def get_scaf_smi(mol: Chem.Mol) -> str:
    """Return the Murcko scaffold of a molecule.

    Args:
        mol: The molecule

    Returns:
        The SMILES string of the Murcko scaffold
    """
    mol = GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
    return mol_to_smi(mol)


@overload
def mol_from_smi(  # pragma: no cover
    smi: str,
    assert_valid: Literal[False] = False,
    sanitize: bool = ...,
    replacements: dict[str, str] | None = ...,
) -> Chem.Mol | None: ...


@overload
def mol_from_smi(  # pragma: no cover
    smi: str,
    assert_valid: Literal[True],
    sanitize: bool = ...,
    replacements: dict[str, str] | None = ...,
) -> Chem.Mol: ...


@overload
def mol_from_smi(  # pragma: no cover
    smi: str,
    assert_valid: bool = False,
    sanitize: bool = ...,
    replacements: dict[str, str] | None = ...,
) -> Chem.Mol | None: ...


def mol_from_smi(
    smi: str,
    assert_valid: bool = False,
    sanitize: bool = True,
    replacements: dict[str, str] | None = None,
) -> Chem.Mol | None:
    """Wraps `Chem.MolFromSmiles`.

    Args:
        smi: SMILES string
        assert_valid: toggles raising an error if invalid. Defaults to False.
        sanitize: toggles sanitization of the molecule. Defaults to True.
        replacements: dictionary of replacement strings. Defaults to {}.


    Returns:
        Molecule or None if invalid and assert_valid is False.

    Raises:
        InvalidSmilesError: If the SMILES string is invalid and allow_invalid is False.
    """
    if not smi:
        if assert_valid:
            raise InvalidSmilesError(smi)
        return None

    if replacements is None:
        replacements = {}

    DisableLog("rdApp.error")
    mol: Chem.Mol | None = Chem.MolFromSmiles(smi, sanitize, replacements)

    if assert_valid and not check_mol(mol):
        raise InvalidSmilesError(smi)

    return mol


def mol_to_smi(
    mol: Chem.Mol, isomeric: bool = True, canonical: bool = True, **kwargs: Any
) -> str:
    """Wraps `Chem.MolToSmiles`.

    Args:
        mol: The molecule
        isomeric: toggles isomeric SMILES. Defaults to True.
        canonical: toggles canonical SMILES. Defaults to True.
        kwargs: Additional keyword arguments
            (kekuleSmiles, rootedAtAtom, allBondsExplicit, allHsExplicit, doRandom)

    Returns:
        SMILES string

    Raises:
        ValueError: If isomericSmiles is in kwargs.
        RDKitError: If the conversion to SMILES fails.
    """
    if "isomericSmiles" in kwargs:
        raise ValueError("isomericSmiles is unsupported. Use isomeric instead.")

    smi = Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=canonical, **kwargs)

    if not smi:
        raise RDKitError(mol, "Failed to convert Mol to SMILES:")

    return smi


__all__ = [
    "canonicalize",
    "check_mol",
    "get_processed_smi",
    "get_scaf_smi",
    "mol_from_smi",
    "mol_to_smi",
]
