"""Test the descriptors module."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.RDConfig import RDContribDir

from chem_tools.descriptors import (
    calc_mol_weight,
    calc_qed,
    calc_sa_score,
    calc_sa_scores,
)

sys.path.append(RDContribDir + "/SA_Score")

import sascorer  # type: ignore[import-not-found] # pylint: disable=import-error,wrong-import-position,wrong-import-order


@pytest.fixture
def mol() -> Chem.Mol:
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def mols() -> list[Chem.Mol]:
    return [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]


def test_calc_qed(mol: Chem.Mol) -> None:
    np.testing.assert_almost_equal(calc_qed(mol), 0.40680796565539457)


def test_calc_sa_score(mol: Chem.Mol) -> None:
    expected = sascorer.calculateScore(mol)
    np.testing.assert_almost_equal(calc_sa_score(mol), expected)


def test_calc_sa_scores(mols: list[Chem.Mol]) -> None:
    expected = [sascorer.calculateScore(mol) for mol in mols]
    np.testing.assert_almost_equal(calc_sa_scores(mols), expected)


def test_calc_mol_weight(mol: Chem.Mol) -> None:
    np.testing.assert_almost_equal(calc_mol_weight(mol), 46.041864812)
