# %%
"""Calculate the internal similarity of a sequence of fingerprints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rdkit import DataStructs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chem_tools._typing import Float32Array


def _calc_internal_similarity(
    fps: Sequence[DataStructs.ExplicitBitVect],
) -> list[list[float]]:
    """Calculate the internal similarity of a sequence of fingerprints."""

    def _calc_bulk_similarity(i: int) -> list[float]:
        """Calculate the similarity between the ith fingerprint and the previous."""
        bulk_sim = DataStructs.BulkTanimotoSimilarity
        return bulk_sim(fps[i], fps[:i])

    return [_calc_bulk_similarity(i) for i in range(1, len(fps))]


def _lower_to_full_arr(
    lower: Sequence[Sequence[float]], fill: float = 0.0
) -> Float32Array:
    """Convert a lower triangular matrix to a full matrix.

    Example:
        ```
            [1,]                [0, 1, 2, 4]
            [2, 3]         ->   [1, 0, 3, 5]
            [4, 5, 6]           [2, 3, 0, 6]
                                [4, 5, 6, 0]
        ```

    Args:
        lower: A lower triangular matrix.
        fill: The value to fill the diagonal with. Defaults to 0.0.

    Returns:
        A full matrix.
    """
    n = len(lower) + 1
    full = np.empty((n, n), dtype=np.float32)
    full.fill(fill)
    for i in range(1, n):
        full[i, :i] = lower[i - 1]
        full[:i, i] = lower[i - 1]
    return full


def calc_internal_similarity(fps: Sequence[DataStructs.ExplicitBitVect]) -> list[float]:
    """Calculate the internal similarity of a sequence of fingerprints as matrix.

    Args:
        fps: A sequence of fingerprints.

    Returns:
        Flattened internal similarity of the fingerprints.
    """
    data = _calc_internal_similarity(fps)

    return [s for d in data for s in d]


def calc_internal_similarity_matrix(
    fps: Sequence[DataStructs.ExplicitBitVect], fill: float = 0.0
) -> list[float] | Float32Array:
    """Calculate the internal similarity of a sequence of fingerprints as matrix.

    Args:
        fps: A sequence of fingerprints.
        fill: The value to fill the diagonal with. Defaults to 0.0.

    Returns:
        A matrix of the similarity values of all provided fingerprints to each other
    """
    data = _calc_internal_similarity(fps)
    return _lower_to_full_arr(data, fill)


__all__ = ["calc_internal_similarity", "calc_internal_similarity_matrix"]
