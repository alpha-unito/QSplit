import json
import os

import certifi
import networkx as nx
import pennylane as qml
from dwave.samplers import SimulatedAnnealingSampler
from networkx.algorithms.approximation.maxcut import one_exchange
from tqdm import tqdm


def main():
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ds = qml.data.load("other", name="hamlib-maxcut", attributes=["edges", "ids", "ns"])[0]
    sa = SimulatedAnnealingSampler()

    res = []
    for i in tqdm(range(len(ds.ns))):
        n_nodes = ds.ns[i]
        if n_nodes > 300:
            continue

        clean_edges = list(tuple((int(u), int(v))) for u, v in ds.edges[i] if u != v)
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        g.add_edges_from(clean_edges)
        qubo_triplets = []

        for n in g.nodes:
            degree = g.degree(n)
            if degree > 0:
                qubo_triplets.append((int(n), int(n), int(-degree)))

        for u, v in g.edges:
            qubo_triplets.append((int(u), int(v), 2))

        total_elements = (n_nodes * (n_nodes + 1)) / 2
        sparsity = float(1 - len(qubo_triplets) / total_elements)
        instance = {
            "originale_index": i,
            "original_id": ds.ids[i],
            "dim": int(n_nodes),
            "qubo_mat": qubo_triplets,
            "sparsity": sparsity,
            "max_cut": -one_exchange(g)[0],
            "sa_max_cut": int(sa.sample_qubo({(x[0], x[1]): x[2] for x in qubo_triplets}).first.energy),
        }
        res.append(instance)

    file_path = "qubo_max_cut.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for instance in res:
            json_record = json.dumps(instance)
            f.write(json_record + "\n")


if __name__ == "__main__":
    main()
