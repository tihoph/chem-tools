"""Run a function in parallel with possible progress bar.

Uses joblib as the backend for parallel processing and tqdm for progress bars.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Any, TypeVar, overload

import joblib
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from typing import Literal

T = TypeVar("T")
R = TypeVar("R")


@overload
def auto_parallel(  # pragma: no cover
    fn: Callable[..., R],
    iterable: Iterable[Any | tuple | dict],
    n_jobs: int,
    as_generator: Literal[False] = False,
    desc: str | None = ...,
    verbose: int = ...,
    keep_arg: bool = ...,
    total: int | None = ...,
) -> list[R]: ...


@overload
def auto_parallel(  # pragma: no cover
    fn: Callable[..., R],
    iterable: Iterable[Any | tuple | dict],
    n_jobs: int,
    as_generator: Literal[True],
    desc: str | None = ...,
    verbose: int = ...,
    keep_arg: bool = ...,
    total: int | None = ...,
) -> Generator[R]: ...


@overload
def auto_parallel(  # pragma: no cover
    fn: Callable[..., R],
    iterable: Iterable[Any | tuple | dict],
    n_jobs: int,
    as_generator: bool = False,
    desc: str | None = ...,
    verbose: int = ...,
    keep_arg: bool = ...,
    total: int | None = ...,
) -> list[R] | Generator[R]: ...


def auto_parallel(
    fn: Callable[..., R],
    iterable: Iterable[T | tuple[T] | dict[str, T]],
    n_jobs: int,
    as_generator: bool = False,
    desc: str | None = None,
    verbose: int = 0,
    keep_arg: bool = False,
    total: int | None = None,
) -> list[R] | Generator[R]:
    """Execute a function in parallel with an optional tqdm progress bar.

    Examples:
        >>> from chem_tools import auto_parallel
        >>> from rdkit import Chem
        >>> def convert(smi: str) -> Chem.Mol: return Chem.MolFromSmiles(smi)
        >>> smis = ["CCO", "CCN", "CCF"] * 1000
        >>> auto_parallel(convert, smis, n_jobs=2)
        ... [Chem.Mol, Chem.Mol, Chem.Mol, ...]
        >>> def tracker(ix_elem: tuple[int, str]) -> tuple[int, str]: return ix_elem
        >>> auto_parallel(tracker, enumerate(smis), n_jobs=2, keep_arg=True)
        ... [(0, "CCO"), (1, "CCN"), (2, "CCF"), ...]

    Args:
        fn: The function to be executed in parallel.
        iterable: The collection to iterate over.
            If an element is a single tuple or dictionary, it will be unpacked.
            To prevent this, set `keep_arg` to True.
        n_jobs: The number of parallel jobs to run.
            If set to 0 or 1, the function runs serially.
            If negative, all CPUs except for abs(`n_jobs`)-1 will be utilized.
        as_generator: Indicates whether to return a generator. Defaults to False.
        desc: A description for the progress bar. Defaults to None.
        verbose: The verbosity level. Defaults to 0.
        keep_arg: Determines whether to retain a single argument as a tuple/dictionary.
            Note that a dictionary will unpack over its keys.
            Disable this if this behavior is not desired.
            Defaults to False.
        total: The total number of items to process.
            If not specified, the length of the iterable will be used (if available).

    Returns:
        The results of the function executed in parallel.
    """
    if n_jobs < 0:
        raise ValueError("n_jobs must be a positive integer")

    iterable = list(iterable)

    def _encapsulate(x: T | tuple[T] | dict[str, T]) -> tuple[T] | dict[str, T]:
        if isinstance(x, dict | tuple) and not keep_arg:
            if isinstance(x, dict) and not all(isinstance(y, str) for y in x):
                raise TypeError("Expanding dictionary requires string keys.")
            return x
        return (x,)  # type: ignore[return-value]

    # important to use a generator to not consume the iterable
    arg_lst = (_encapsulate(x) for x in iterable)

    if total is None and isinstance(iterable, Sized):
        total = len(iterable)

    if n_jobs not in {0, 1}:
        parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator")
        jobs = (
            joblib.delayed(fn)(**x) if isinstance(x, dict) else joblib.delayed(fn)(*x)
            for x in arg_lst
        )
        gen: Generator[R] = parallel(jobs)
    else:
        gen = (fn(**x) if isinstance(x, dict) else fn(*x) for x in arg_lst)

    disable_pbar = verbose < 1

    gen = (x for x in tqdm(gen, desc=desc, total=total, disable=disable_pbar))

    if as_generator:
        return gen

    # if not return `as_generator`, consume the generator
    return list(gen)


__all__ = ["auto_parallel"]
