# ruff: noqa: C901, N802, N803, N806, E501, PLR2004
"""typed version of https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py.

calculation of synthetic accessibility score as described in:

Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
Peter Ertl and Ansgar Schuffenhauer
Journal of Cheminformatics 1:8 (2009)
http://www.jcheminf.com/content/1/1/8

several small modifications to the original paper are included
particularly slightly different formula for marocyclic penalty
and taking into account also molecule symmetry (fingerprint density)

for a set of 10k diverse molecules the agreement between the original method
as implemented in PipelinePilot and this implementation is r2 = 0.97

peter ertl & greg landrum, september 2013
"""

from __future__ import annotations

import gzip
import logging
import math
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem, RDConfig
from rdkit.Chem import DataStructs, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

chem_tools_logger = logging.getLogger("chem_tools")


@lru_cache
def readFragmentScores(  # pragma: no cover
    name: Path | PathLike[str] | str = "fpscores",
) -> dict[int, float]:
    """Read the fragment scores from a file.

    Args:
        name: Path to the file containing the fragment scores.
            If "fpscores", the file is read from the RDKit contrib directory.

    Returns:
        A dictionary mapping fragment indices to their scores.
    """
    # generate the full path filename:
    if name == "fpscores":
        contrib_dir = Path(RDConfig.RDContribDir)
        path = contrib_dir / "SA_Score" / "fpscores.pkl.gz"
    else:
        path = Path(name)

    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    fscores: dict[int, float] = {}
    for i in data:
        for j in range(1, len(i)):
            fscores[i[j]] = float(i[0])

    return fscores


def numBridgeheadsAndSpiro(  # pragma: no cover
    mol: Chem.Mol, ri: Chem.RingInfo | None = None
) -> tuple[int, int]:
    """Calculates the number of bridgeheads and spiro atoms in a molecule.

    Args:
        mol: The molecule to calculate the number of bridgeheads and spiro atoms for.
        ri: The ring information for the molecule. Unused.

    Returns:
        A tuple containing the number of bridgeheads and spiro atoms.
    """
    del ri  # not used

    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(  # pragma: no cover
    m: Chem.Mol,
    fscores: dict[int, float] | None = None,
    fp: DataStructs.UIntSparseIntVect | None = None,
) -> float:
    """Calculate the synthetic accessibility score of a molecule.

    Args:
        m: The molecule to calculate the synthetic accessibility score for.
        fscores: A dictionary mapping fragment indices to their scores.
            If None, the fragment scores are read from the RDKit contrib directory.
        fp: The fingerprint of the molecule. If None, the fingerprint is calculated.

    Returns:
        The synthetic accessibility score of the molecule.
    """
    if fscores is None:
        fscores = readFragmentScores()

    if fp is None:
        # fragment score
        fp_gen = rdFPGen.GetMorganGenerator(
            radius=2
        )  # <- 2 is the *radius* of the circular fingerprint
        fp = fp_gen.GetSparseCountFingerprint(m)

    fps: dict[int, int] = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    chiralCenters: list = Chem.FindMolChiralCenters(m, includeUnassigned=True)  # type: ignore[no-untyped-call]
    nChiralCenters = len(chiralCenters)
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1) # noqa: ERA001
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0  # noqa: A001
    max = 2.5  # noqa: A001
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore  # type: ignore[no-any-return]


def processMols(  # pragma: no cover
    mols: Iterable[Chem.Mol | None],
    numThreads: int = 16,
    verbose: int = 0,
    fscores: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Process a list of molecules and print their synthetic accessibility scores.

    Args:
        mols: An iterable of molecules to process.
        numThreads: The number of threads to use. Defaults to 16.
        verbose: The verbosity level. Defaults to 0.
        fscores: A dictionary mapping fragment indices to their scores.
            If None, the fragment scores are read from the RDKit contrib directory.

    Returns:
        A list of tuples containing the index and synthetic accessibility score of each molecule.
    """
    if verbose > 0:
        print("smiles\tName\tsa_score")  # noqa: T201

    # only process valid molecules
    vmols = [x for x in mols if x is not None]
    fp_gen = rdFPGen.GetMorganGenerator(
        radius=2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp_gen.GetSparseCountFingerprints(vmols, numThreads=numThreads)

    scores: list[tuple[int, float]] = []

    for ix, (mol, fp) in enumerate(zip(vmols, fps, strict=True)):
        score = calculateScore(mol, fscores, fp)

        if verbose > 0:
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            print(f"{smiles}\t{name}\t{score:.3f}")  # noqa: T201

        scores.append((ix, score))

    return scores


# %%
if __name__ == "__main__":  # pragma: no cover
    t1 = time.time()
    readFragmentScores("fpscores")
    t2 = time.time()

    suppl = Chem.SmilesMolSupplier(sys.argv[1])
    t3 = time.time()
    processMols(suppl, verbose=0)
    t4 = time.time()

    chem_tools_logger.info(
        "Reading took %.2f seconds. Calculating took %.2f seconds", t2 - t1, t4 - t3
    )

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

__all__ = ["calculateScore", "processMols"]
