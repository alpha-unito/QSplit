import copy
import importlib
import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from qsplit import _core, configuration
from qsplit.adapters.training import training_subproblem
from qsplit.cwl.cli.scatter import _same_subproblem
from qsplit.cwl.cli.utils import build_qubo_from_matrix
from qsplit.matrix import as_dense
from qsplit.qubo import QUBO
from qsplit.refinement.runner import refine_result

MATRIX = np.array([[2, -4, 0, 3, 0], [0, 1, -2, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 2, -3], [0, 0, 0, 0, -2.0]])
IDS = np.array([20, 10, 70, 30, 50])


def qubo(matrix=MATRIX, factory=np.asarray, cols=IDS):
    return QUBO(factory(matrix), IDS.copy(), cols.copy(), offset=7)


def assert_same(left, right):
    np.testing.assert_array_equal(left.rows_idx, right.rows_idx)
    np.testing.assert_array_equal(left.cols_idx, right.cols_idx)
    np.testing.assert_allclose(as_dense(left.mat), as_dense(right.mat), equal_nan=True)
    assert left.offset == pytest.approx(right.offset)


@pytest.mark.parametrize(
    "factory", [sparse.csr_array, sparse.csr_matrix, sparse.coo_array, sparse.csc_matrix, sparse.lil_matrix]
)
def test_sparse_inputs_preserve_all_energies_and_do_not_alias(factory):
    raw = MATRIX + MATRIX.T * 0.5
    source = factory(raw)
    dense, compact = qubo(raw), qubo(raw, factory=lambda _: source)
    assert isinstance(compact.mat, sparse.csr_array)
    assert_same(dense, compact)
    for bits in product((0, 1), repeat=len(IDS)):
        x = np.asarray(bits, dtype=float)
        assert _core.calculate_energy(x, compact.mat) == pytest.approx(x @ raw @ x)
    compact.mat.data[:] = 0
    np.testing.assert_array_equal(source.toarray(), raw)


@pytest.mark.parametrize("value", [0, 1e-9, 1e-8, 1.1e-8, np.nan, np.inf, -np.inf])
def test_sparse_sanitization_matches_dense_tolerance_and_nonfinite_behavior(value):
    matrix = np.array([[np.inf, 2.0], [value, -1.0]])
    with np.errstate(invalid="ignore"):
        expected, _, _ = QUBO.sanitize(matrix, np.arange(2), np.arange(2))
        actual, _, _ = QUBO.sanitize(sparse.coo_array(matrix), np.arange(2), np.arange(2))
    np.testing.assert_allclose(actual.toarray(), expected, equal_nan=True)


def test_sparse_duplicates_unsorted_indices_and_rounding():
    matrix = sparse.coo_array(([1.0, 2.0, -3.0, 1.1234567894], ([0, 0, 0, 1], [1, 1, 1, 1])), shape=(2, 2))
    q = QUBO(matrix, np.arange(2), np.arange(2))
    assert q.mat.nnz == 1
    assert q.mat[1, 1] == 1.123456789
    # Noncanonical public replacement must work at native and Python boundaries.
    q.mat = sparse.csr_array(([3.0, -1.0, 2.0], [1, 0, 1], [0, 3, 3]), shape=(2, 2))
    assert q.mat.has_canonical_format
    assert _core.calculate_energy(np.ones(2), q.mat) == 4


@pytest.mark.parametrize("method", ["linear", "recursive", "k_interactions", "recursive_graph", "quadtree"])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_splitters_match_dense_including_padding_and_macros(method, stride):
    split = importlib.import_module(f"qsplit.splitting.split_{method}").split_problem
    dense, compact = qubo(), qubo(factory=sparse.csr_array)
    options = {"CUT_DIM": 2, "STRIDE": stride, "EXACT_RATIO": 0.5}
    left, right = split(dense, config=options), split(compact, config=options)
    assert len(left) == len(right)
    assert_same(dense, compact)
    for a, b in zip(left, right):
        assert_same(a, b)
        assert getattr(a, "macro_members", {}) == getattr(b, "macro_members", {})
        if method != "quadtree":
            assert sparse.issparse(b.mat)


def test_linear_occupied_blocks_match_with_wide_overlapping_windows():
    rng = np.random.default_rng(2)
    for size in range(1, 11):
        matrix = np.triu(rng.integers(-1, 2, (size, size)))
        for cut, stride in product((1, 3, 5), (1, 2, 4, 6)):
            ids = np.arange(size)
            dense = QUBO(matrix, ids, ids)
            compact = QUBO(sparse.csr_array(matrix), ids, ids)
            left, right = _core.split_linear(dense, cut, stride), _core.split_linear(compact, cut, stride)
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_same(a, b)


@pytest.mark.parametrize("cols", [IDS, IDS[::-1]])
def test_conditioning_fractional_beliefs_and_distinct_ids(cols):
    dense, compact = qubo(cols=cols), qubo(factory=sparse.csr_array, cols=cols)
    beliefs = dict(zip(IDS, [0.25, 0.75, 0.5, 1, 0]))
    for window in ([70], [50, 20], list(IDS), []):
        assert_same(
            _core.condition_subproblem(dense, window, beliefs), _core.condition_subproblem(compact, window, beliefs)
        )


@pytest.mark.parametrize("strategy", ["mean_field", "soft_consensus"])
def test_full_refinement_loop_matches_dense(strategy, exact_solver):
    dense, compact = qubo(), qubo(factory=sparse.csr_array)
    for q in (dense, compact):
        q.solutions = pd.DataFrame({**{int(i): [0] for i in IDS}, "energy": [q.offset]})
        refine_result(
            q,
            solve=exact_solver,
            block_size=2,
            config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 2, "REFINEMENT_METHOD": strategy},
        )
    pd.testing.assert_frame_equal(dense.solutions, compact.solutions)
    assert dense.refinement_history == compact.refinement_history


def test_sparse_conflicts_logical_expansion_and_energy_match_dense(exact_solver):
    from qsplit.conflict_resolution.local_search import nan_subqubo

    dense, compact = qubo(), qubo(factory=sparse.csr_array)
    frame = pd.DataFrame({**{int(i): [np.nan] for i in IDS}, "energy": [np.nan]})
    pd.testing.assert_frame_equal(nan_subqubo(frame, dense), nan_subqubo(frame, compact))
    left, right = _core.split_recursive(dense), _core.split_recursive(compact)
    for subs in (left, right):
        for sub in subs:
            sub.solutions = exact_solver(sub)
        _core.logical_expansion(subs)
    for a, b in zip(left, right):
        assert_same(a, b)


def test_sparse_training_and_cache_match_dense():
    dense, compact = qubo(), qubo(factory=sparse.csr_array)
    with configuration.use({"LOCAL_TRAINING_QUBITS": 3}):
        assert_same(training_subproblem(dense), training_subproblem(compact))
    assert _same_subproblem(dense, compact)
    for restored in (copy.deepcopy(compact), pickle.loads(pickle.dumps(compact))):
        assert sparse.issparse(restored.mat)
        assert_same(compact, restored)
        restored.mat.data[:] = 0
        assert compact.mat.nnz > 0 and compact.mat.data.any()


def test_sparse_npz_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sparse.save_npz("matrix.npz", sparse.csr_matrix(MATRIX))
    restored = build_qubo_from_matrix("matrix.npz")
    assert sparse.issparse(restored.mat)
    np.testing.assert_array_equal(restored.mat.toarray(), MATRIX)
    assert (tmp_path / "initial_qubo.pkl").exists()


@pytest.mark.parametrize("module", ["dwave.dwave_sa", "ibm.ibm_qaoa_cpu_noiseless", "ibm.ibm_pce_cpu_noiseless"])
def test_sparse_solver_returns_original_energy(module, assert_solution):
    solve = importlib.import_module(f"qsplit.adapters.{module}").solve
    q = QUBO(sparse.csr_array([[-1, 2], [0, -2.0]]), np.array([7, 11]), np.array([7, 11]), offset=3)
    assert_solution(q, solve(q))


def test_large_sparse_paths_never_materialize_global_matrix(monkeypatch):
    size = 20_001
    matrix = sparse.eye_array(size, format="csr")
    q = QUBO(matrix, np.arange(size), np.arange(size), offset=2)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Global sparse matrix was densified")

    monkeypatch.setattr(sparse.csr_array, "toarray", forbid_dense)
    assert not _core.is_empty(q)
    assert _core.vars_count(q) == size
    assert _core.calculate_energy(np.ones(size), q.mat) == size
    assert _core.condition_subproblem(q, [0, 1], dict.fromkeys(range(size), 0.5)).problem_size == 2
    subs = _core.split_linear(q, 128, 128)
    assert len(subs) == (size + 127) // 128
    assert all(sparse.issparse(sub.mat) for sub in subs)
    restored = pickle.loads(pickle.dumps(q))
    assert restored.mat.nnz == size
    # Recursive splitting must also avoid a dense padded copy of the global matrix.
    recursive = _core.split_recursive(q)
    assert all(sparse.issparse(sub.mat) for sub in recursive)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2,)])
def test_invalid_sparse_dimensions_are_rejected(shape):
    with pytest.raises(AssertionError, match="square"):
        QUBO(sparse.csr_array(shape), np.arange(shape[0]), np.arange(shape[-1]))


def test_corrupted_sparse_indices_raise_before_native_access():
    q = qubo(factory=sparse.csr_array)
    q.mat.indices[0] = q.problem_size
    with pytest.raises(ValueError, match="indices"):
        _core.is_empty(q)


def test_sparse_empty_padding_and_nonconsecutive_column_ids():
    q = qubo(factory=sparse.csr_array, cols=IDS[::-1])
    dense = qubo(cols=IDS[::-1])
    q.rows_idx[0] = dense.rows_idx[0] = -1
    q.cols_idx[1] = dense.cols_idx[1] = -2
    assert _core.vars_count(q) == _core.vars_count(dense)
    assignment = pd.DataFrame({**{int(i): [1] for i in IDS}, "energy": [0.0]})
    pd.testing.assert_frame_equal(
        _core.validate_solutions(assignment.copy(), q), _core.validate_solutions(assignment.copy(), dense)
    )
    q.mat.data[:] = 0
    assert _core.is_empty(q)
    assert _core.vars_count(q) == 0
