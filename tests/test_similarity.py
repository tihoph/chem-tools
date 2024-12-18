"""Test internal similarity calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

from chem_tools.descriptors.similarity import (
    _calc_internal_similarity,
    _lower_to_full_arr,
    calc_internal_similarity,
    calc_internal_similarity_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FixedData = tuple[list[DataStructs.ExplicitBitVect], list[list[float]]]


def build_data(smis: Sequence[str]) -> FixedData:
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    gen: rdFPGen.FingeprintGenerator64 = rdFPGen.GetMorganGenerator(
        radius=2, fpSize=1024
    )
    fps: list[DataStructs.ExplicitBitVect] = [gen.GetFingerprint(mol) for mol in mols]
    sims: list[list[float]] = [
        DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps
    ]
    return fps, sims


def test_lower_to_full_matrix() -> None:
    inputs = [[1], [2, 3], [4, 5, 6]]
    expected = [[0, 1, 2, 4], [1, 0, 3, 5], [2, 3, 0, 6], [4, 5, 6, 0]]
    np.testing.assert_equal(_lower_to_full_arr(inputs, fill=0), expected)


@pytest.mark.parametrize(
    "smis", [("c1ccccc1", "C1CCCCC1", "CC(=O)O"), ("CC(=O)N", "CCC", "CNC")]
)
def test_calc_internal_similarity(smis: tuple[str, str, str]) -> None:
    fps, sims = build_data(smis)

    calcd_sims = _calc_internal_similarity(fps)

    expected: list[list[float]] = [
        DataStructs.BulkTanimotoSimilarity(fps[1], fps[:1]),
        DataStructs.BulkTanimotoSimilarity(fps[2], fps[:2]),
    ]
    np.testing.assert_equal(calcd_sims, expected)

    expanded = _lower_to_full_arr(calcd_sims, fill=1.0)
    sub_matrix = [x[:3] for x in sims[:3]]
    np.testing.assert_allclose(expanded, sub_matrix)

    expected_flat = [x for d in expected for x in d]

    matrix = calc_internal_similarity_matrix(fps, fill=1.0)
    np.testing.assert_allclose(matrix, sims)

    flat = calc_internal_similarity(fps)
    np.testing.assert_allclose(flat, expected_flat)
