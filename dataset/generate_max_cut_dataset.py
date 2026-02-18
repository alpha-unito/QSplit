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

    unique_instances = {}
    for i in tqdm(range(len(ds.ns))):
        n_nodes = ds.ns[i]
        if n_nodes > 100 or n_nodes in unique_instances:
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
        sa_sol = int(sa.sample_qubo({(x[0], x[1]): x[2] for x in qubo_triplets}).first.energy)
        instance = {
            "id": f"{i}-{ds.ids[i]}",
            "dim": int(n_nodes),
            "qubo_mat": qubo_triplets,
            "offset": 0,
            "sa": sa_sol,
            "min": -one_exchange(g)[0],
            "max": 0,
            "sparsity": sparsity,
        }
        unique_instances[n_nodes] = instance

    res = sorted(unique_instances.values(), key=lambda x: x["dim"])
    file_path = "qubo_max_cut.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for instance in res:
            json_record = json.dumps(instance)
            f.write(json_record + "\n")


if __name__ == "__main__":
    main()
