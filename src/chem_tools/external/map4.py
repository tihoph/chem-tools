#!/usr/bin/env python
# adjusted from https://github.com/reymond-group/map4
"""The MinHashed Atom Pair fingerprint of radius 2"""

# ruff: noqa: D100, D101, D103, D415, D417
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING

from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    import chem_tools.external._tmap as tm
    from chem_tools._typing import UInt8Array

else:
    try:
        import tmap as tm
    except ImportError:
        import chem_tools.external._tmap as tm


class MAP4Calculator:
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        is_counted: bool = False,
        is_folded: bool = False,
        return_strings: bool = False,
    ) -> None:
        """MAP4 calculator class"""
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.is_folded = is_folded
        self.return_strings = return_strings

        if self.is_folded:
            self.encoder = MHFPEncoder(dimensions)
        else:
            self.encoder = tm.Minhash(dimensions)

    def calculate(self, mol: Chem.Mol) -> tm.VectorUint | UInt8Array | list[bytes]:
        """Calculates the atom pair minhashed fingerprint

        Args:
            mol: rdkit mol object

        Returns:
            minhashed fingerprint as tmap VectorUint
        """
        atom_env_pairs = self._calculate(mol)
        if self.is_folded:
            return self._fold(atom_env_pairs)

        if self.return_strings:
            return atom_env_pairs

        return self.encoder.from_string_array(atom_env_pairs)  # type: ignore[no-any-return]

    def calculate_many(
        self, mols: Sequence[Chem.Mol]
    ) -> list[tm.VectorUint] | list[UInt8Array] | list[list[bytes]]:
        """Calculates the atom pair minhashed fingerprint

        Args:
            mols: list of mols

        Returns:
            list minhashed fingerprints as tmap VectorUints
        """
        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        if self.is_folded:
            return [self._fold(pairs) for pairs in atom_env_pairs_list]

        if self.return_strings:
            return atom_env_pairs_list

        return self.encoder.batch_from_string_array(  # type: ignore[no-any-return]
            atom_env_pairs_list
        )

    def _calculate(self, mol: Chem.Mol) -> list[bytes]:
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs: list[bytes]) -> UInt8Array:
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)  # type: ignore[no-any-return]

    def _get_atom_envs(self, mol: Chem.Mol) -> dict[int, list[str]]:
        atoms_env: dict[int, list[str]] = {}
        atom: Chem.Atom
        for atom in mol.GetAtoms():  # type: ignore[call-arg, no-untyped-call]
            idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []
                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))
        return atoms_env

    @classmethod
    def _find_env(cls, mol: Chem.Mol, idx: int, radius: int) -> str:
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map: dict[int, int] = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if idx in atom_map:
            return Chem.MolToSmiles(
                submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False
            )
        return ""

    def _all_pairs(self, mol: Chem.Mol, atoms_env: dict[int, list[str]]) -> list[bytes]:
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict: defaultdict[str, int] = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = f"{ordered[0]}|{dist}|{ordered[1]}"

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += "|" + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode("utf-8"))
        return list(set(atom_pairs))


__all__ = ["MAP4Calculator"]

# the command line implementation is removed -> see original repo for details
