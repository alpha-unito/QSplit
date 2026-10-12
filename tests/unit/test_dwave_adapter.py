import unittest
from unittest.mock import Mock

import dimod
import numpy as np

from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO


def test_qpu_does_not_open_pickle_cache(monkeypatch, make_qubo):
    import diskcache
    from dwave.cloud import Client

    from qsplit.adapters.dwave import dwave_qa

    def forbid_cache(*args, **kwargs):
        raise AssertionError("QPU client must not open a pickle-backed disk cache")

    def sampler(**options):
        # Even a user configuration enabling the cache must not override the mitigation.
        with Client.from_config(
            config_file=False, endpoint="https://example.invalid/sapi", cache_enabled=True, **options
        ) as client:
            assert not client.solvers_session.session._cache_enabled
        return object()

    qubo = make_qubo([[-1, 0], [0, -2]])
    composite = Mock()
    composite.sample.return_value = dimod.SampleSet.from_samples_bqm({0: 1, 1: 1}, from_qubo_matrix_to_bqm(qubo))
    monkeypatch.setattr(diskcache, "Cache", forbid_cache)
    monkeypatch.setattr(dwave_qa, "DWaveSampler", sampler)
    monkeypatch.setattr(dwave_qa, "EmbeddingComposite", Mock(return_value=composite))
    result = dwave_qa.solve(qubo, config={"DWAVE_API_TOKEN": "test-token"})
    assert result.iloc[0]["energy"] == -3


class TestDWaveAdapter(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[1.0, -2.0], [0.0, 1.0]], dtype=np.float64)
        self.rows_idx = np.array([0, 1], dtype=int)
        self.cols_idx = np.array([0, 1], dtype=int)
        self.offset = 5.0
        self.qubo = QUBO(self.mat, self.rows_idx, self.cols_idx, offset=self.offset)

    def test_conversion_matrix_to_bqm(self):
        bqm = from_qubo_matrix_to_bqm(self.qubo)
        self.assertIsInstance(bqm, dimod.BinaryQuadraticModel)
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.linear[0], 1.0)
        self.assertEqual(bqm.linear[1], 1.0)
        self.assertEqual(bqm.quadratic[(0, 1)], -2.0)
        self.assertEqual(bqm.offset, 5.0)

    def test_to_dataframe(self):
        samples = np.array([[0, 0], [1, 1], [0, 1], [0, 0]])
        energies = np.array([1.5, 5.0, 1.5, 1.5])
        sampleset = dimod.SampleSet.from_samples(samples, vartype=dimod.BINARY, energy=energies)
        df = to_dataframe(sampleset, self.qubo)

        self.assertNotIn("num_occurrences", df.columns)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["energy"] == 1.5).all())


if __name__ == "__main__":
    unittest.main()
