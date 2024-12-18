"""Test the external FCD module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fcd
import pytest
import torch

from chem_tools.external import fcd as fcd_mod

if TYPE_CHECKING:
    from collections.abc import Sequence


# code contains a resource leak as the temporary directory is not closed
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_load_ref_model() -> None:
    model = fcd.load_ref_model()
    assert isinstance(model, torch.nn.modules.container.Sequential)


@pytest.mark.parametrize("smis", [["CCO", "CCN"], ["CCO", "CCN", "CCC"]])
def test_calc_fcd_stats(smis: Sequence[str]) -> None:
    mean, cov = fcd_mod.calc_fcd_stats(smis)
    assert mean.shape == (512,)
    assert cov.shape == (512, 512)
