import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from qsplit import _core, configuration
from qsplit.adapters.all_zero import solve as constant_solution
from qsplit.cwl.cli.utils import bitstring_from_row, load_qubo, parse_solved_paths, save_qubo
from qsplit.cwl.methods import AGGREGATORS, resolve_refinement_aggregate
from qsplit.qubo import QUBO
from qsplit.refinement.refine_mean_field import refine_problems as mean_field
from qsplit.refinement.refine_soft_consensus import refine_problems as soft_consensus

REFINEMENT_METHODS = ["none", "conditioned", "consensus", "mean_field", "soft_consensus"]


def load_samples(paths, instance_id, expected=None):
    samples = []
    for path in parse_solved_paths(paths):
        sub = load_qubo(path)
        if not isinstance(sub, QUBO) or getattr(sub, "instance_id", None) != instance_id:
            raise ValueError("Refinement subproblem belongs to a different instance")
        samples.append(sub)
    samples.sort(key=lambda sub: sub.node_id)
    ids = [sub.node_id for sub in samples]
    if len(set(ids)) != len(ids) or (expected is not None and set(ids) != set(expected)):
        raise ValueError("Missing, duplicate or stale refinement subproblems")
    if expected is not None:
        for sub in samples:
            spec = expected[sub.node_id]
            if sub.rows_idx.tolist() != spec[0] or sub.cols_idx.tolist() != spec[1]:
                raise ValueError("Refinement subproblem variable IDs do not match the prepared block")
    return samples


def initialize(qubo, samples, *, method, loops, block_size, aggregate_method):
    if method not in REFINEMENT_METHODS:
        raise ValueError(f"Unknown refinement method: {method}")
    active = method != "none" and loops > 0
    if active and block_size <= 0:
        raise ValueError("Refinement requires a positive CUT_DIM")
    if method == "consensus":
        method = "soft_consensus" if any(getattr(sub, "macro_members", {}) for sub in samples) else "mean_field"
    aggregate_method = resolve_refinement_aggregate(aggregate_method)
    options = {"method": method, "loops": max(loops, 0), "block_size": block_size, "aggregate": aggregate_method}
    if active:
        tolerance = float(configuration.get("REFINEMENT_TOLERANCE", "1e-9"))
        patience = int(configuration.get("REFINEMENT_PATIENCE", "10" if method == "conditioned" else "1"))
        strength = float(configuration.get("REFINEMENT_STRENGTH", "0.1")) if method == "soft_consensus" else 0.0
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("REFINEMENT_TOLERANCE must be finite and non-negative")
        if patience <= 0:
            raise ValueError("REFINEMENT_PATIENCE must be positive")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("REFINEMENT_STRENGTH must be finite and non-negative")
        seed = int(configuration.get("REFINEMENT_SEED", "0")) if method == "conditioned" else 0
        options.update(tolerance=tolerance, patience=patience, strength=strength)
        rng_state = np.random.default_rng(seed).bit_generator.state
    else:
        rng_state = None
    qubo.solutions = _core.validate_solutions(qubo.solutions.copy(deep=True), qubo)
    energy = float(qubo.solutions.energy.min())
    if not np.isfinite(energy):
        raise ValueError("Refinement requires finite global energies")
    qubo.refinement_history = [energy]
    qubo.refinement_state = {
        "options": options,
        "active": active and bool(samples),
        "rounds": 0,
        "cursor": 0,
        "windows": [],
        "rng_state": rng_state,
        "stalled": 0,
        "best_energy": energy,
        "round_start_energy": energy,
        "working_solutions": qubo.solutions.copy(deep=True),
    }
    return qubo


def prepare(qubo, samples):
    info = qubo.refinement_state
    options = info["options"]
    if not info["active"]:
        info["pending"] = {}
        return []
    if options["method"] == "conditioned":
        if not info["windows"]:
            ids = [int(idx) for idx in qubo.rows_idx if idx >= 0]
            rng = np.random.default_rng()
            rng.bit_generator.state = info["rng_state"]
            order = rng.permutation(ids).tolist()
            size = options["block_size"]
            info["windows"] = [order[start : start + size] for start in range(0, len(order), size)]
            info["rng_state"] = rng.bit_generator.state
            info["round_start_energy"] = info["best_energy"]
        working = info["working_solutions"]
        assignment = working.iloc[working.energy.to_numpy().argmin()].drop("energy").to_dict()
        subs = [_core.condition_subproblem(qubo, info["windows"][info["cursor"]], assignment)]
    else:
        info["round_start_energy"] = info["best_energy"]
        strategy = mean_field if options["method"] == "mean_field" else soft_consensus
        subs = strategy(
            samples, qubo, config={"CUT_DIM": options["block_size"], "REFINEMENT_STRENGTH": options["strength"]}
        )
    expected = {}
    for index, sub in enumerate(subs):
        sub.instance_id = qubo.instance_id
        sub.node_id = f"refine_{info['rounds']:06d}_{info['cursor']:06d}_{index:06d}"
        expected[sub.node_id] = [sub.rows_idx.tolist(), sub.cols_idx.tolist()]
    info["pending"] = expected
    return subs


def update(qubo, samples):
    info = qubo.refinement_state
    options = info["options"]
    if not info["active"]:
        return qubo
    if options["method"] == "conditioned":
        if len(samples) != 1:
            raise ValueError("Conditioned refinement requires exactly one solved block per transition")
        working = deepcopy(qubo)
        working.solutions = info["working_solutions"]
        candidates = (
            _core.accept_conditioned(working, samples[0], samples[0].solutions)
            if np.any(samples[0].mat)
            else working.solutions.copy(deep=True)
        )
        info["working_solutions"] = candidates.copy(deep=True)
        info["cursor"] += 1
        finished = info["cursor"] == len(info["windows"])
    else:
        candidates = AGGREGATORS[options["aggregate"]](samples, deepcopy(qubo)).solutions
        finished = True
    candidates = _core.validate_solutions(candidates.copy(deep=True), qubo)
    energy = float(candidates.energy.min())
    if not np.isfinite(energy):
        raise ValueError("Refinement requires finite global energies")
    if energy < info["best_energy"]:
        qubo.solutions = candidates
        info["best_energy"] = energy
    qubo.refinement_subproblems = samples
    if finished:
        qubo.refinement_history.append(energy)
        info["rounds"] += 1
        improvement = info["round_start_energy"] - energy
        info["stalled"] = info["stalled"] + 1 if improvement <= options["tolerance"] else 0
        info["cursor"], info["windows"] = 0, []
        info["active"] = info["rounds"] < options["loops"] and info["stalled"] < options["patience"] and bool(samples)
    return qubo


def write_state(qubo):
    save_qubo("state.pkl", qubo)
    Path("control.json").write_text(json.dumps({"continue": qubo.refinement_state["active"]}), encoding="utf-8")
    Path("refinement_history.json").write_text(json.dumps(qubo.refinement_history), encoding="utf-8")
    ids = [int(idx) for idx in qubo.cols_idx if idx >= 0]
    rows = ["node_id,backend,bitstring,energy"]
    for _, sample in qubo.solutions.nsmallest(10, "energy").iterrows():
        rows.append(f"root,aggregate,{bitstring_from_row(sample, ids)},{float(sample.energy):.12g}")
    Path("solutions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@configuration.cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initialize", "prepare", "update"], required=True)
    parser.add_argument("--input-state", required=True)
    parser.add_argument("--solved-list", action="extend", nargs="+", default=[])
    parser.add_argument("--method", choices=REFINEMENT_METHODS, default="none")
    parser.add_argument("--loops", type=int, default=0)
    parser.add_argument("--cut-dim", type=int, default=16)
    parser.add_argument("--aggregate-method", default="linear")
    args = parser.parse_args()
    qubo = load_qubo(args.input_state)
    expected = qubo.refinement_state["pending"] if args.phase == "update" else None
    samples = (
        load_samples(args.solved_list, qubo.instance_id, expected)
        if args.phase == "update"
        else qubo.refinement_subproblems
    )
    if args.phase == "initialize":
        initialize(
            qubo,
            samples,
            method=args.method,
            loops=args.loops,
            block_size=args.cut_dim,
            aggregate_method=args.aggregate_method,
        )
    elif args.phase == "prepare":
        subs = prepare(qubo, samples)
        for folder in ("subproblems", "solved_constant"):
            Path(folder).mkdir(exist_ok=True)
        for sub in subs:
            if np.any(sub.mat):
                save_qubo(Path("subproblems") / f"{sub.node_id}.pkl", sub)
            else:
                sub.solutions, sub.backend = constant_solution(sub), "constant"
                sub.solutions["energy"] = sub.offset
                save_qubo(Path("solved_constant") / f"{sub.node_id}.pkl", sub)
    else:
        update(qubo, samples)
    write_state(qubo)


if __name__ == "__main__":
    main()
