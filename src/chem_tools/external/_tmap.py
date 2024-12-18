"""Stub implementation of TMAP for MacOS.

As TMAP is not available on MacOS,
but we don't want to disable the whole package
we will raise Errors only if specific classes
or methods are used.
"""

# ruff: noqa: ARG002, ARG004
from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover

if TYPE_CHECKING:
    from collections.abc import Iterable


class VectorUint:  # pragma: no cover
    def __init__(self, arg0: Iterable | VectorUint | None = None) -> None:
        del arg0
        raise ImportError("Tmap is not available on macOS.")


class VectorFloat:  # pragma: no cover
    def __init__(self, arg0: Iterable | VectorFloat | None = None) -> None:
        del arg0
        raise ImportError("Tmap is not available on macOS.")


class Minhash:  # pragma: no cover
    @staticmethod
    def get_distance(fp1: VectorUint, fp2: VectorUint) -> float:
        del fp1, fp2
        raise ImportError("Tmap is not available on macOS.")


class LSHForest:
    def __init__(
        self,
        d: int = ...,
        l: int = ...,  # noqa: E741
        store: bool = ...,
        file_backed: bool = ...,
        weighted: bool = ...,
    ) -> None:
        del d, l, store, file_backed, weighted
        raise ImportError("Tmap is not available on macOS.")


class LayoutConfiguration:
    def __init__(self) -> None:
        raise ImportError("Tmap is not available on macOS.")
