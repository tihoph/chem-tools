"""Test the fingerprints module."""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import pytest
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.Chem.rdMolDescriptors import (
    GetHashedAtomPairFingerprintAsBitVect,
    GetMorganFingerprintAsBitVect,
)
from rdkit.DataStructs import ExplicitBitVect, TanimotoSimilarity

from chem_tools.external.map4 import MAP4Calculator
from chem_tools.fingerprints import (
    AtomPairGenerator,
    AtomPairNumpyGenerator,
    FingerprintGenerator,
    MorganGenerator,
    MorganNumpyGenerator,
    convert_bv_to_np,
    convert_np_to_bv,
)
from chem_tools.fingerprints.external import MAP4Generator, MHFPGenerator

tmap_unavailable = importlib.util.find_spec("tmap") is None

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rdkit.Chem.rdFingerprintGenerator import FingeprintGenerator64

    from chem_tools._typing import Int64Array, UInt8Array
    from chem_tools.external import _tmap as tm
else:
    try:
        import tmap as tm
    except ImportError:
        from chem_tools.external import _tmap as tm


T = TypeVar("T")


DIMS = 1024
RADIUS = 2


@pytest.fixture
def mols() -> list[Chem.Mol]:
    return [
        Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O"),  # Aspirin
        Chem.MolFromSmiles("CN1CCC[C@H]1C2=CN=CC=C2"),  # Nicotine
        Chem.MolFromSmiles("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"),  # Testosterone
    ]


def build_rdkit_gen(
    fp_type: Literal["morgan", "atom_pair"], features: bool = False
) -> FingeprintGenerator64:
    if fp_type == "morgan":
        gen = rdFPGen.GetMorganGenerator(RADIUS, fpSize=DIMS)
    elif fp_type == "atom_pair":
        if features:
            raise ValueError("Atom pair fingerprints do not support features")
        gen = rdFPGen.GetAtomPairGenerator(fpSize=DIMS)
    else:
        raise ValueError(f"Invalid fingerprint type: {fp_type}")
    return gen


def get_numpy_fps_func(
    fp_type: Literal["morgan", "atom_pair"], features: bool = False
) -> Callable[[Sequence[Chem.Mol]], list[UInt8Array]]:
    gen = build_rdkit_gen(fp_type, features)
    return lambda mols: [gen.GetFingerprintAsNumPy(mol) for mol in mols]


def get_bv_fps_func(
    fp_type: Literal[
        "morgan", "atom_pair", "old_atom_pair", "old_morgan", "old_features"
    ],
    features: bool = False,
) -> Callable[[Sequence[Chem.Mol]], list[ExplicitBitVect]]:
    if fp_type == "old_atom_pair":
        return lambda mols: [
            GetHashedAtomPairFingerprintAsBitVect(mol, nBits=DIMS) for mol in mols
        ]
    if fp_type == "old_morgan":
        return lambda mols: [
            GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=DIMS) for mol in mols
        ]
    if fp_type == "old_features":
        return lambda mols: [
            GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=DIMS, useFeatures=True)
            for mol in mols
        ]

    gen = build_rdkit_gen(fp_type, features)
    return lambda mols: [gen.GetFingerprint(mol) for mol in mols]


def get_map_func(folded: bool) -> Callable[[Sequence[Chem.Mol]], list[tm.VectorUint]]:
    map_calc = MAP4Calculator(dimensions=DIMS, radius=RADIUS, is_folded=folded)
    return lambda mols: [map_calc.calculate(mol) for mol in mols]  # type: ignore[misc]


def get_orig_mhfp(mols: Sequence[Chem.Mol]) -> list[Int64Array]:
    mhfp_gen = MHFPEncoder(DIMS)
    return [mhfp_gen.encode_mol(mol, radius=RADIUS, min_radius=0) for mol in mols]


def assert_conversion(
    bv_fps: Sequence[ExplicitBitVect], np_fps: Sequence[UInt8Array]
) -> None:
    for bv_fp, np_fp in zip(bv_fps, np_fps, strict=True):
        new_bv = convert_np_to_bv(np_fp)
        new_np = convert_bv_to_np(new_bv)
        assert isinstance(new_bv, ExplicitBitVect)
        assert isinstance(new_np, np.ndarray)
        assert new_np.dtype == np.uint8
        np.testing.assert_equal(new_np, np_fp)
        assert new_bv == bv_fp


def assert_equal_fps(
    gen: FingerprintGenerator[T],
    mols: Sequence[Chem.Mol],
    func: Callable[[Sequence[Chem.Mol]], list[T]],
) -> list[T]:
    fp = gen.get(mols[0])
    fps = gen.get_many(mols)
    if isinstance(fps, tuple):
        fps = list(fps)

    expected = func(mols)

    if isinstance(fp, np.ndarray):
        np.testing.assert_equal(fp, expected[0])
        np.testing.assert_equal(fps, expected)
    else:
        assert fp == expected[0]
        assert fps == expected

    return list(fps)


def test_morgan_generator(mols: Sequence[Chem.Mol]) -> None:
    gen = MorganGenerator(radius=RADIUS, n_bits=DIMS)
    bv_fps = assert_equal_fps(gen, mols, get_bv_fps_func("morgan"))
    assert_equal_fps(gen, mols, get_bv_fps_func("old_morgan"))

    gen = MorganGenerator(radius=RADIUS, n_bits=DIMS, features=True)
    assert_equal_fps(gen, mols, get_bv_fps_func("old_features"))

    np_gen = MorganNumpyGenerator(radius=RADIUS, n_bits=DIMS)
    np_fps = assert_equal_fps(np_gen, mols, get_numpy_fps_func("morgan"))
    assert_conversion(bv_fps, np_fps)


def test_atom_pair_generator(mols: list[Chem.Mol]) -> None:
    gen = AtomPairGenerator(n_bits=DIMS)
    bv_fps = assert_equal_fps(gen, mols, get_bv_fps_func("atom_pair"))
    assert_equal_fps(gen, mols, get_bv_fps_func("old_atom_pair"))

    np_gen = AtomPairNumpyGenerator(n_bits=DIMS)
    np_fps = assert_equal_fps(np_gen, mols, get_numpy_fps_func("atom_pair"))
    assert_conversion(bv_fps, np_fps)


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_map4_generator(mols: list[Chem.Mol], caplog: pytest.LogCaptureFixture) -> None:
    gen = MAP4Generator(dims=DIMS, radius=RADIUS, folded=False)
    assert_equal_fps(gen, mols, get_map_func(False))
    with caplog.at_level(logging.WARNING):
        gen = MAP4Generator(dims=DIMS, radius=RADIUS, folded=True)
        assert (
            "For folded fingerprints, use the MHFPGenerator for similarity calculation."
            in caplog.text
        )
    assert_equal_fps(gen, mols, get_map_func(True))


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_mhfp_generator(mols: list[Chem.Mol]) -> None:
    gen = MHFPGenerator(radius=RADIUS, dim=DIMS)
    assert_equal_fps(gen, mols, get_orig_mhfp)


def assert_similarity(
    gen: FingerprintGenerator[T],
    mols: Sequence[Chem.Mol],
    sim_func: Callable[[T, T], float],
    fps: Sequence[T] | None = None,
) -> None:
    fps = gen.get_many(mols) if fps is None else fps
    assert fps is not None
    sim = gen.similarity(fps[0], fps[1])
    bulk = gen.bulk_similarity(fps[0], fps)
    matrix = gen.matrix_similarity(fps, fps)

    assert sim == sim_func(fps[0], fps[1])

    expected_bulk = [sim_func(fps[0], fp) for fp in fps]
    if isinstance(bulk, np.ndarray):  # type: ignore[unreachable]
        np.testing.assert_allclose(bulk, expected_bulk)  # type: ignore[unreachable]
    else:
        assert bulk == expected_bulk

    expected_matrix = [[sim_func(fp1, fp2) for fp1 in fps] for fp2 in fps]
    if isinstance(matrix, np.ndarray):
        np.testing.assert_allclose(matrix, expected_matrix)
    else:
        assert matrix == expected_matrix  # type: ignore[unreachable]


def np_tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    return TanimotoSimilarity(convert_np_to_bv(fp1), convert_np_to_bv(fp2))


def test_morgan_similarity(mols: list[Chem.Mol]) -> None:
    gen = MorganGenerator(radius=RADIUS, n_bits=DIMS)
    assert_similarity(gen, mols, TanimotoSimilarity)

    np_gen = MorganNumpyGenerator(radius=RADIUS, n_bits=DIMS)
    assert_similarity(np_gen, mols, np_tanimoto)


def test_atom_pair_similarity(mols: list[Chem.Mol]) -> None:
    gen = AtomPairGenerator(n_bits=DIMS)
    assert_similarity(gen, mols, TanimotoSimilarity)

    np_gen = AtomPairNumpyGenerator(n_bits=DIMS)
    assert_similarity(np_gen, mols, np_tanimoto)


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_map4_similarity(mols: list[Chem.Mol]) -> None:
    gen = MAP4Generator(dims=DIMS, radius=RADIUS, folded=False)
    assert_similarity(
        gen, mols, lambda fp1, fp2: 1 - tm.Minhash().get_distance(fp1, fp2)
    )
    gen = MAP4Generator(dims=DIMS, radius=RADIUS, folded=True)
    fps = gen.get_many(mols)
    mhfp_gen = MHFPGenerator(radius=RADIUS, dim=DIMS)
    assert_similarity(
        mhfp_gen, mols, lambda fp1, fp2: 1 - MHFPEncoder.distance(fp1, fp2), fps=fps
    )
    assert_similarity(
        mhfp_gen, mols, lambda fp1, fp2: 1 - np_minhash_distance(fp1, fp2), fps=fps
    )


def np_minhash_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    if not TYPE_CHECKING:
        import tmap as tm

    min_hash = tm.Minhash()  # pylint: disable=possibly-used-before-assignment

    return min_hash.get_distance(tm.VectorUint(fp1), tm.VectorUint(fp2))


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_mhfp_similarity(mols: list[Chem.Mol]) -> None:
    gen = MHFPGenerator(radius=RADIUS, dim=DIMS)
    assert_similarity(gen, mols, lambda fp1, fp2: 1 - MHFPEncoder.distance(fp1, fp2))
    assert_similarity(gen, mols, lambda fp1, fp2: 1 - np_minhash_distance(fp1, fp2))


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_map_multiple_jobs(mols: list[Chem.Mol]) -> None:
    gen = MAP4Generator(dims=DIMS, radius=RADIUS)
    with pytest.raises(ValueError, match="n_jobs must be 1 for MAP4Generator"):
        gen.get_many(mols, n_jobs=2)

    fps = gen.get_many(mols)
    with pytest.raises(ValueError, match="n_jobs must be 1 for MAP4Generator"):
        gen.matrix_similarity(fps, fps, n_jobs=2)


def test_repr() -> None:
    morgan_gen = MorganGenerator(radius=2, n_bits=2048, features=False)
    expected = "MorganGenerator(radius=2, n_bits=2048, features=False)"
    assert repr(morgan_gen) == expected

    morgan_np_gen = MorganNumpyGenerator(radius=2, n_bits=2048, features=False)
    expected = "MorganNumpyGenerator(radius=2, n_bits=2048, features=False)"
    assert repr(morgan_np_gen) == expected


@pytest.mark.skipif(tmap_unavailable, reason="tmap not installed")
def test_map4_repr() -> None:
    map_gen = MAP4Generator(dims=DIMS, radius=RADIUS)
    expected = "MAP4Generator(radius=2, dims=1024)"
    assert repr(map_gen) == expected
