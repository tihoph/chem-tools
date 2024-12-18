"""Types for the chem_tools package."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

StrPath = Path | str | PathLike[str]
UInt8Array = NDArray[np.uint8]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
Float32Array = NDArray[np.float32]
StrArray = NDArray[np.str_]
