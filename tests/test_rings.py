"""Test ring size calculation."""

from __future__ import annotations

from typing import TypeAlias

import pytest
from rdkit import Chem

from chem_tools.descriptors.rings import get_largest_ring_size, get_ring_systems

RingData: TypeAlias = tuple[
    Chem.Mol, int, int, set[frozenset[int]], set[frozenset[int]]
]


@pytest.fixture(params=["cyclohexane", "spiro", "fused", "multiple"])
def rings(request: pytest.FixtureRequest) -> RingData:
    if request.param == "cyclohexane":
        mol = Chem.MolFromSmiles("C1CCCCC1")
        atoms = 6
        rings = {frozenset(range(atoms))}
        return mol, atoms, atoms, rings, rings

    if request.param == "spiro":
        mol = Chem.MolFromSmiles("C1CCCC12CC2")
        atoms = 5
        rings = {frozenset(range(atoms)), frozenset((4, 5, 6))}
        spiro_rings = {frozenset(range(atoms + 2))}
        return mol, atoms, atoms + 2, rings, spiro_rings

    if request.param == "fused":
        mol = Chem.MolFromSmiles("o2c1ccccc1cc2")
        atoms = 9
        rings = {frozenset(range(atoms))}
        return mol, atoms, atoms, rings, rings

    if request.param == "multiple":
        mol = Chem.MolFromSmiles("C1CCCCC1CC1CCCCC1")
        atoms = 6
        rings = {frozenset(range(atoms)), frozenset(range(atoms + 1, 2 * atoms + 1))}
        return mol, atoms, atoms, rings, rings

    raise ValueError(f"Unknown parameter: {request.param}")


@pytest.mark.parametrize("incl_spiro", [False, True])
@pytest.mark.parametrize(
    "rings",
    [
        pytest.param("cyclohexane"),
        pytest.param("spiro"),
        pytest.param("fused"),
        pytest.param("multiple"),
    ],
    indirect=True,
)
def test_ring_systems(rings: RingData, incl_spiro: bool) -> None:
    mol = rings[0]
    expected = rings[-1] if incl_spiro else rings[-2]
    assert get_ring_systems(mol, incl_spiro) == expected


@pytest.mark.parametrize("incl_spiro", [False, True])
@pytest.mark.parametrize(
    "rings",
    [
        pytest.param("cyclohexane"),
        pytest.param("spiro"),
        pytest.param("fused"),
        pytest.param("multiple"),
    ],
    indirect=True,
)
def test_largest_ring_size(rings: RingData, incl_spiro: bool) -> None:
    mol = rings[0]
    expected = rings[2] if incl_spiro else rings[1]
    assert get_largest_ring_size(mol, incl_spiro) == expected
