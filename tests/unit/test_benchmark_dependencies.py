import subprocess
import sys
from pathlib import Path

import pytest


def test_benchmark_bridge_runs_without_quantum_sdks(tmp_path):
    pytest.importorskip("qoblib")
    pytest.importorskip("pyscipopt")
    script = r"""
import csv
import importlib.abc
import json
import sys
from pathlib import Path

class BlockQuantumImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dimod', 'dwave', 'qiskit', 'pennylane', 'pytket', 'cudaq', 'iqm', 'qnexus'}:
            raise ModuleNotFoundError(f'Quantum dependency forbidden in benchmark bridge: {fullname}')

sys.meta_path.insert(0, BlockQuantumImports())

from qsplit.benchmark import decode, encode_model, to_qubo
from qsplit.benchmark.bridge import prepare, finalize
from qsplit.benchmark.conversion import read_lp

root = Path(sys.argv[1])
lp = root / 'model.lp'
lp.write_text('Minimize\n obj: x\nSubject To\n c: x <= 1\nBounds\n 0 <= x <= 1\nGeneral\n x\nEnd\n')
record = encode_model(read_lp(lp))
assert decode(record, '0' * record['dim'])['x'] == 0
assert to_qubo(record).offset == record['offset']

source = root / 'source'
source.mkdir()
request = root / 'request.yaml'
request.write_text('runs: 1\ncheck: "off"\ninstances:\n  - problem: labs\n    instance: "4"\n')
bundle = root / 'bundle'
dataset = prepare(request, bundle, repository=source)
record = json.loads(dataset.read_text())
store = root / 'solutions'
store.mkdir()
with (store / f"solutions_{record['id']}.csv").open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['node_id', 'backend', 'bitstring', 'energy'])
    writer.writerow(['root', 'aggregate', '0' * record['dim'], record['offset']])
report = json.loads(finalize(bundle, store, root / 'results').read_text())
assert report['runs'][0]['objective'] == 14
assert report['runs'][0]['solution_file']
assert 'dimod' not in json.loads((bundle / 'manifest.json').read_text())['versions']
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
