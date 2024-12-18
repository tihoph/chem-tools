"""Test the randomized module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rdkit import Chem

from chem_tools.randomized import (
    augment_smis,
    augment_smis_plus,
    mol_to_random_smi,
    mol_to_smi,
    randomize_many,
    randomize_mol,
    randomize_smi,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def patch_permutation(orders: Iterable[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    gen = (np.array(order) for order in orders)
    monkeypatch.setattr(np.random, "permutation", lambda _: next(gen))


params = [
    ("CCO", [0, 1, 2], "CCO"),
    ("CCO", [0, 2, 1], "CCO"),
    ("CCO", [1, 0, 2], "C(C)O"),
    ("CCO", [1, 2, 0], "C(O)C"),
    ("CCO", [2, 0, 1], "OCC"),
    ("CCO", [2, 1, 0], "OCC"),
]


@pytest.mark.parametrize(("smi", "order", "expected"), params)
def test_randomize_mol(
    smi: str, order: list[int], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mol = Chem.MolFromSmiles(smi)
    patch_permutation([order], monkeypatch)
    rnd = randomize_mol(mol)
    new_smi = mol_to_smi(rnd, canonical=False)
    assert new_smi == expected

    patch_permutation([order], monkeypatch)
    new_smi = mol_to_random_smi(mol)
    assert new_smi == expected


def test_randomize_h_mol(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mol = Chem.MolFromSmiles("CCO")
    hs_mol = Chem.AddHs(mol)
    patch_permutation([[2, 0, 1]], monkeypatch)
    with caplog.at_level(logging.WARNING):
        rnd = randomize_mol(hs_mol)
        assert "Removed Hs to randomize molecule:" in caplog.text
    new_smi = mol_to_smi(rnd, canonical=False)
    assert new_smi == "OCC"


def test_randomize_isomeric_h_mol(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mol = Chem.MolFromSmiles("[4H]OC")
    patch_permutation([[1, 0]], monkeypatch)
    with caplog.at_level(logging.WARNING):
        rnd = randomize_mol(mol)
        assert "Removed all Hs to randomize molecule:" in caplog.text

    new_smi = mol_to_smi(rnd, canonical=False)
    assert new_smi == "CO"


def test_randomize_complex_mol(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mol = Chem.MolFromSmiles("[4H]CN(C)C(=O)c1cnc2sc(-c3ccc(Br)cc3)cc2c1CCCNC")
    reversed_order = list(range(mol.GetNumHeavyAtoms()))[::-1]
    patch_permutation(
        [reversed_order, reversed_order], monkeypatch
    )  # twice for recursive call
    with caplog.at_level(logging.WARNING):
        rnd = randomize_mol(mol)
        assert "Removed all Hs to randomize molecule:" in caplog.text
    new_smi = mol_to_smi(rnd, canonical=False)
    assert new_smi == "CNCCCc1c2cc(-c3ccc(Br)cc3)sc2ncc1C(=O)N(C)C"


@pytest.mark.parametrize(("smi", "order", "expected"), params)
def test_randomize_smi(
    smi: str, order: list[int], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_permutation([order], monkeypatch)
    rnd = randomize_smi(smi)
    assert rnd == expected


def test_randomize_many(monkeypatch: pytest.MonkeyPatch) -> None:
    smis, orders, new_smis = zip(*params, strict=False)
    patch_permutation(orders, monkeypatch)
    rnd = randomize_many(smis, n_jobs=1)
    assert rnd == list(new_smis)


def test_augment_smis(monkeypatch: pytest.MonkeyPatch) -> None:
    _, orders, new_smis = zip(*params, strict=False)
    patch_permutation(orders, monkeypatch)
    rnd = augment_smis("CCO", n_repr=6, n_jobs=1)
    assert rnd == list(new_smis)


def test_augment_smis_plus(monkeypatch: pytest.MonkeyPatch) -> None:
    _, orders, new_smis = zip(*params, strict=False)
    patch_permutation(orders, monkeypatch)
    rnd = augment_smis_plus("CCO", "VAL", n_repr=6, n_jobs=1)
    rnd_smis, *data = rnd
    assert len(data) == 1
    assert rnd_smis == list(new_smis)
    np.testing.assert_equal(data[0], ["VAL"] * 6)

    patch_permutation(orders, monkeypatch)
    with pytest.raises(ValueError, match=r"All arguments must have the same length."):
        augment_smis_plus("CCO", ["VAL", "VAL"], n_repr=6, n_jobs=1)
