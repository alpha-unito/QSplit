"""Public NumPy-compatible interface to the compiled QUBO class."""

import numpy as np

from qsplit._core import QUBO as QUBO

int_arr = np.ndarray[tuple[int], np.dtype[int]]
flt_mat = np.ndarray[tuple[int, int], np.dtype[np.float64]]
