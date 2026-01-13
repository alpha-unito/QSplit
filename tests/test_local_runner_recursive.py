# Copyright (C) 2025  The QSplit Contributors.
# See the 'CONTRIBUTORS' file at the top-level directory of this distribution.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from qsplit.local_runner import qsplit_sampler_recursive


class TestLocalRunnerRecursive(unittest.TestCase):
    def test_qsplit_sampler_recursive_from_csv(self):
        csv_path = Path(__file__).resolve().parents[1] / "streamflow" / "cwl" / "data" / "32x32_b.csv"
        self.assertTrue(csv_path.exists())

        previous_cut_dim = os.environ.get("CUT_DIM")
        os.environ["CUT_DIM"] = "8"
        try:
            mat = np.loadtxt(csv_path, delimiter=",")
            n = mat.shape[0]
            qubo = QUBO(mat, rows_idx=np.arange(n), cols_idx=np.arange(n))

            result = qsplit_sampler_recursive(qubo)

            self.assertIsNotNone(result)
            self.assertIsNotNone(result.solutions)
            self.assertIsInstance(result.solutions, pd.DataFrame)
            self.assertFalse(result.solutions.empty)
            self.assertIn("energy", result.solutions.columns)
            self.assertIn(0, result.solutions.columns)
            self.assertIn(n - 1, result.solutions.columns)

            print(result.solutions)
        finally:
            if previous_cut_dim is None:
                os.environ.pop("CUT_DIM", None)
            else:
                os.environ["CUT_DIM"] = previous_cut_dim


if __name__ == "__main__":
    unittest.main()
