import json

from dimod import cqm_to_bqm
from dimod.generators import random_knapsack
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
from tqdm import tqdm


def main():
    dims = []
    with open("../dataset/qubo_max_cut.jsonl") as f:
        for line in f.readlines():
            dims.append(json.loads(line)["dim"])
    dims.sort()
    sa = SimulatedAnnealingSampler()
    tabu = TabuSampler()

    res = []
    for idx, dim in enumerate(tqdm(dims)):
        cqm = random_knapsack(num_items=dim, seed=idx, tightness_ratio=0.5)
        bqm = cqm_to_bqm(cqm)[0]
        qubo, offset = bqm.to_qubo()
        bqm.offset = 0
        kp_min = int(tabu.sample(bqm).first.energy)
        bqm_inverted = bqm.copy()
        bqm_inverted.scale(-1)
        kp_max = int(-tabu.sample(bqm_inverted).first.energy)
        num_vars = len(bqm.variables)
        qubo_triplets = []
        for (u, v), val in qubo.items():
            var_u = int(u.split("_")[-1]) + (dim if u.startswith("slack") else 0)
            var_v = int(v.split("_")[-1]) + (dim if v.startswith("slack") else 0)
            r, c = (var_u, var_v) if var_u <= var_v else (var_v, var_u)
            qubo_triplets.append((r, c, val))

        sa_sol = int(sa.sample_qubo({(x[0], x[1]): x[2] for x in qubo_triplets}).first.energy)
        instance = {
            "id": f"{idx}-{dim}",
            "dim": num_vars,
            "qubo_mat": qubo_triplets,
            "offset": offset,
            "sa": sa_sol,
            "min": kp_min,
            "max": kp_max,
            "sparsity": 0,
        }
        res.append(instance)

    res.sort(key=lambda x: x["dim"])
    file_path = "qubo_kp.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for instance in res:
            json_record = json.dumps(instance)
            f.write(json_record + "\n")


if __name__ == "__main__":
    main()
