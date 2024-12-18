"""Test conversion functions."""

from __future__ import annotations

import pytest
from rdkit import Chem

from chem_tools._conversion import (
    InvalidSmilesError,
    RDKitError,
    canonicalize,
    check_mol,
    get_processed_smi,
    get_scaf_smi,
    mol_from_smi,
    mol_to_smi,
)


def assert_mols_equal(mol1: Chem.Mol, mol2: Chem.Mol) -> None:
    assert Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)


def test_mol_from_smi() -> None:
    assert_mols_equal(mol_from_smi("CCO", assert_valid=True), Chem.MolFromSmiles("CCO"))
    assert_mols_equal(
        mol_from_smi("MeCO", assert_valid=True, replacements={"Me": "C"}),
        Chem.MolFromSmiles("CCO"),
    )


@pytest.mark.parametrize("smi", ["X", ""])
def test_mol_from_smi_fails(smi: str) -> None:
    with pytest.raises(
        InvalidSmilesError, match=r"Could not convert SMILES to molecule: " + smi
    ):
        mol_from_smi(smi, assert_valid=True)

    assert mol_from_smi(smi) is None
    assert mol_from_smi(smi, assert_valid=False) is None


def test_mol_to_smi() -> None:
    assert mol_to_smi(Chem.MolFromSmiles("CCO")) == "CCO"
    assert mol_to_smi(Chem.MolFromSmiles("N[C@H](C)C(=O)O")) == "C[C@@H](N)C(=O)O"
    assert (
        mol_to_smi(Chem.MolFromSmiles("N[C@H](C)C(=O)O"), isomeric=False)
        == "CC(N)C(=O)O"
    )


def test_check_mol() -> None:
    assert check_mol(None) is False
    assert check_mol(Chem.Mol()) is False
    assert check_mol(Chem.MolFromSmiles("CCO")) is True


def test_canonicalize() -> None:
    assert canonicalize("OCC") == "CCO"
    assert canonicalize("N[C@H](C)C(=O)O") == "C[C@@H](N)C(=O)O"
    assert canonicalize("N[C@H](C)C(=O)O", isomeric=False) == "CC(N)C(=O)O"


def test_get_scaf_smi() -> None:
    with pytest.raises(RDKitError, match=r"Failed to convert Mol to SMILES: "):
        get_scaf_smi(Chem.MolFromSmiles("CCO"))
    assert get_scaf_smi(Chem.MolFromSmiles("C1CC1CCC")) == "C1CC1"


@pytest.mark.parametrize(
    ("smi", "expected", "remove_stereo"),
    [
        ("CCO", "CCO", False),
        ("C[C@@H](N)C(=O)O", "C[C@@H](N)C(=O)O", False),
        ("C[C@@H](N)C(=O)O", "CC(N)C(=O)O", True),
    ],
)
def test_get_processed_smi(smi: str, expected: str, remove_stereo: bool) -> None:
    result = get_processed_smi(smi, remove_stereo=remove_stereo)
    assert isinstance(result, tuple)
    assert len(result) == 2
    can, mol = result
    assert can == expected
    assert_mols_equal(mol, Chem.MolFromSmiles(expected))


@pytest.mark.parametrize("smi", ["X", ""])
def test_get_processed_smi_fails(smi: str) -> None:
    with pytest.raises(
        InvalidSmilesError, match=r"Could not convert SMILES to molecule: " + smi
    ):
        get_processed_smi(smi, assert_valid=True)

    assert get_processed_smi(smi) is None
    assert get_processed_smi(smi, assert_valid=False) is None


# %%
