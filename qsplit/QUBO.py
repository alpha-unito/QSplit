from typing import Mapping, Hashable, List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import lu
from numpy import floating, integer


class QUBO:
    def __init__(self, qubo_dict: Mapping[tuple[Hashable, Hashable], float | floating | integer],
                 cols_idx: List[int], rows_idx: List[int]):
        self.qubo_dict: Mapping[tuple[Hashable, Hashable], float | floating | integer] = qubo_dict
        self.cols_idx: List[int] = cols_idx
        self.rows_idx: List[int] = rows_idx
        self.solutions: pd.DataFrame | None = None
        self.qubo_matrix: np.ndarray | None = None
        self.__from_dict_to_matrix((len(rows_idx), len(cols_idx)))

    def __is_upper_triangular(self) -> bool:
        rows, cols = self.qubo_matrix.shape
        if rows != cols:
            return False
        for i in range(rows):
            for j in range(i):
                if self.qubo_matrix[i, j] != 0:
                    return False
        return True

    def __from_dict_to_matrix(self, dims: Tuple[int, int]) -> None:
        self.qubo_matrix = np.zeros(dims)
        for k, v in self.qubo_dict.items():
            self.qubo_matrix[(k[0] - 1) % dims[0], (k[1] - 1) % dims[1]] = v
        if not self.__is_upper_triangular():
            self.qubo_matrix = lu(self.qubo_matrix, permute_l=True)[1]
        self.__from_matrix_to_dict()

    def __from_matrix_to_dict(self) -> None:
        self.qubo_dict = {}
        for i in range(self.qubo_matrix.shape[0]):
            for j in range(self.qubo_matrix.shape[1]):
                if self.qubo_matrix[i, j] != 0:
                    self.qubo_dict[(i + min(self.rows_idx), j + min(self.cols_idx))] = float(self.qubo_matrix[i, j])
