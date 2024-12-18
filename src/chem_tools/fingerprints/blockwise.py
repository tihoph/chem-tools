"""Utility functions for numpy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from chem_tools._parallel import auto_parallel
from chem_tools._typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", Sequence, NDArray)
R = TypeVar("R", bound=NDArray)


def blockwise_array(
    arr_a: T,
    arr_b: T,
    sub_func: Callable[[T, T], R],
    max_size: int = 1000,
    n_jobs: int = 1,
) -> R:
    """Partwise applies a function to two arrays for a full matrix.

    Examples:
        >>> import numpy as np
        >>> from clm_generator.python import blockwise_array
        >>> def sub_func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ...     return a + b
        >>> a = np.ones((200, 100))
        >>> b = np.arange(300*100).reshape(300, 100)
        >>> r = blockwise_array(a, b, sub_func, max_size=10)
        >>> r.shape
        (200, 300, 100)


    Args:
        arr_a: The first array or sequence.
        arr_b: The second array or sequence.
        sub_func: The function to apply to the subarrays.
        max_size: The window size for the subarrays. Defaults to 1000.
        n_jobs: The number of parallel jobs. Defaults to 1.

    Returns:
        The resulting array.
    """
    x_steps = [*list(range(0, len(arr_a), max_size)), len(arr_a)]
    y_steps = [*list(range(0, len(arr_b), max_size)), len(arr_b)]

    x_start_end = [(x_steps[ix], x_steps[ix + 1]) for ix in range(len(x_steps) - 1)]
    y_start_end = [(y_steps[ix], y_steps[ix + 1]) for ix in range(len(y_steps) - 1)]

    blocks: list[list[R]] = [[] for _ in range(len(x_start_end))]

    data: list[tuple[T, T]] = []

    for x_start, x_end in x_start_end:
        for y_start, y_end in y_start_end:
            sub_arr_a = arr_a[x_start:x_end]
            sub_arr_b = arr_b[y_start:y_end]

            data.append((sub_arr_a, sub_arr_b))

    if n_jobs > 1:
        result = auto_parallel(sub_func, data, n_jobs=n_jobs)
    else:
        result = [sub_func(sub_arr_a, sub_arr_b) for sub_arr_a, sub_arr_b in data]

    for cell_ix, raw_sub_result in enumerate(result):
        x_level = cell_ix // len(y_start_end)
        if raw_sub_result.ndim == 3:  # noqa: PLR2004
            sub_result = raw_sub_result.transpose((2, 0, 1))
        else:
            sub_result = raw_sub_result
        blocks[x_level].append(sub_result)

    combined: R = np.block(blocks)  # type: ignore[assignment]
    if combined.ndim == 3:  # noqa: PLR2004
        combined = combined.transpose(1, 2, 0)
    return combined


__all__ = ["blockwise_array"]
