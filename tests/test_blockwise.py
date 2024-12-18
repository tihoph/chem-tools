# %%
"""Tests for the np_utils module."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pytest

from chem_tools.fingerprints import blockwise_array

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chem_tools._typing import Int32Array, NDArray

T = TypeVar("T")


def _test_blockwise_array(
    a: Sequence[T],
    b: Sequence[T],
    sub_func: Callable[[Sequence[T], Sequence[T]], NDArray],
    expected: NDArray,
    n_jobs: int,
) -> None:
    blocks = blockwise_array(a, b, sub_func, max_size=2, n_jobs=n_jobs)

    assert blocks.shape == expected.shape

    assert np.allclose(blocks, expected)


def _seq_int_func() -> (
    tuple[
        Sequence[T], Sequence[T], Callable[[Sequence[T], Sequence[T]], NDArray], NDArray
    ]
):
    a = list(range(4))
    b = list(range(8))

    def only_ints(a: Sequence[int], b: Sequence[int]) -> Int32Array:
        return np.array(
            [[10 * a_elem + b_elem for b_elem in b] for a_elem in a], dtype=np.int32
        )

    expected = only_ints(a, b)
    assert expected.shape == (4, 8)

    return a, b, only_ints, expected  # type: ignore[return-value]


def _arr_func() -> (
    tuple[
        Sequence[T], Sequence[T], Callable[[Sequence[T], Sequence[T]], NDArray], NDArray
    ]
):
    a = np.arange(4 * 3).reshape((4, 3)) + 10
    b = np.arange(8 * 3).reshape((8, 3))

    def convert_arrs(a: Sequence[Int32Array], b: Sequence[Int32Array]) -> Int32Array:
        return a[:, np.newaxis, :] + b[np.newaxis, :, :]  # type: ignore[no-any-return, call-overload]

    expected = convert_arrs(a, b)  # type: ignore[arg-type]
    assert expected.shape == (4, 8, 3)

    return a, b, convert_arrs, expected  # type: ignore[return-value]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("build_func", [_seq_int_func, _arr_func])
def test_blockwise_array_int(
    build_func: Callable[
        [],
        tuple[
            Sequence[T],
            Sequence[T],
            Callable[[Sequence[T], Sequence[T]], NDArray],
            NDArray,
        ],
    ],
    n_jobs: int,
) -> None:
    a, b, sub_func, expected = build_func()
    _test_blockwise_array(a, b, sub_func, expected, n_jobs)
