"""Frechet ChemNet Distance (FCD)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn

    from chem_tools._typing import Float32Array


def calc_fcd_stats(
    smiles: Sequence[str], model: nn.Sequential | None = None
) -> tuple[Float32Array, Float32Array]:
    """Calculate the mean and covariance of the FCD of a sequence of SMILES strings.

    Args:
        smiles: A sequence of SMILES strings.
        model: A model to use for calculating the FCD.
            If None, the default model is used. Defaults to None.

    Returns:
        A tuple with the mean and covariance of the FCD.
    """
    import fcd

    if model is None:
        model = fcd.load_ref_model()

    act = fcd.get_predictions(model, smiles)

    mu: Float32Array = np.mean(act, axis=0)
    sigma: Float32Array = np.cov(act.T)

    return mu, sigma


__all__ = ["calc_fcd_stats"]
