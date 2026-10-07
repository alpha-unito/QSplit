import argparse
import csv
import json
import time
from pathlib import Path

from qsplit import configuration
from qsplit.benchmark.bridge import finalize, prepare, write_json
from qsplit.benchmark.conversion import to_qubo
from qsplit.cwl.methods import AGGREGATORS, SPLIT_METHODS, SPLITTERS, resolve_aggregate


def run(request, output_dir, *, repository=None, split_method="recursive_graph", aggregate_method="auto", config=None):
    from qsplit.aggregation.aggregate_recursive import aggregate_leaves
    from qsplit.local_runner import qsplit_sampler
    from qsplit.splitting.split_recursive import split_leaves

    output = Path(output_dir)
    bundle = output / "bundle"
    start = time.perf_counter()
    dataset = prepare(request, bundle, repository=repository)
    preparation = time.perf_counter() - start
    settings = {"CUT_DIM": 16, "REFINEMENT_LOOPS": 0, "REFINEMENT_METHOD": "conditioned", **configuration.load(config)}
    settings["QSPLIT_BACKEND"] = "dwave"
    settings.pop("QSPLIT_SOLVER_MODULE", None)
    aggregate = resolve_aggregate(split_method, aggregate_method)
    splitter = split_leaves if split_method == "recursive" else SPLITTERS[split_method]
    aggregator = aggregate_leaves if aggregate == "recursive" else AGGREGATORS[aggregate]
    store = output / "solutions"
    store.mkdir(parents=True, exist_ok=True)
    timings = {}
    for line in dataset.read_text().splitlines():
        record = json.loads(line)
        qubo = to_qubo(record)
        qubo.instance_id = record["id"]
        qubo.node_id = "root"

        def split(qubo):
            leaves = splitter(qubo)
            for i, leaf in enumerate(leaves):
                leaf.instance_id, leaf.node_id = record["id"], f"root_{i:06d}"
            return leaves

        started = time.perf_counter()
        solved = qsplit_sampler(
            qubo,
            split=split,
            aggregate=aggregator,
            config=settings,
        )
        timings[record["id"]] = time.perf_counter() - started
        with (store / f"solutions_{record['id']}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "backend", "bitstring", "energy"])
            for _, row in solved.solutions.iterrows():
                bits = "".join(str(int(row[i])) for i in range(record["dim"]))
                writer.writerow(["root", "aggregate", bits, row["energy"]])
        print(f"QSPLIT BENCHMARK completed {record['problem']}/{record['instance']} run={record['run']}", flush=True)
    started = time.perf_counter()
    result = finalize(
        bundle,
        store,
        output / "results",
        timings=timings,
        pipeline={
            "split_method": split_method,
            "aggregate_method": aggregate,
            "cut_dim": settings["CUT_DIM"],
            "refinement_method": settings["REFINEMENT_METHOD"],
            "refinement_loops": settings["REFINEMENT_LOOPS"],
            "backend": "simulated_annealing",
        },
    )
    write_json(
        output / "run.json",
        {
            "preparation_seconds": preparation,
            "pipeline_seconds": timings,
            "export_and_validation_seconds": time.perf_counter() - started,
            "total_seconds": time.perf_counter() - start,
            "configuration": {
                k: v
                for k, v in settings.items()
                if k.startswith("REFINEMENT_")
                or k in {"CUT_DIM", "STRIDE", "EXACT_RATIO", "QSPLIT_BACKEND", "QSPLIT_SOLVER_MODULE"}
            },
            "split_method": split_method,
            "aggregate_method": aggregate,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--repository")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--split-method", choices=SPLIT_METHODS, default="recursive_graph")
    parser.add_argument("--aggregate-method", default="auto")
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    print(
        run(
            args.request,
            args.output_dir,
            repository=args.repository,
            split_method=args.split_method,
            aggregate_method=args.aggregate_method,
            config=args.config,
        )
    )


if __name__ == "__main__":
    main()
