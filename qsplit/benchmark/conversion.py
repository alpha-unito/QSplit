import itertools
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from qsplit.qubo import QUBO


def integer_weights(span: int) -> list[int]:
    if span < 0:
        raise ValueError("Empty integer domain")
    weights = []
    while span:
        weight = min(1 << len(weights), span)
        weights.append(weight)
        span -= weight
    return weights


def _scaled_constraint(constraint):
    bounds = [constraint.get("lower"), constraint.get("upper")]
    values = list(constraint["coefficients"].values()) + [v for v in bounds if v is not None]
    if all(abs(v) <= 2**53 and v == int(v) for v in values):
        divisor = math.gcd(*(int(v) for v in values)) or 1
        coefficients = {k: int(v) // divisor for k, v in constraint["coefficients"].items()}
        lower, upper = [None if v is None else int(v) // divisor for v in bounds]
    else:
        fractions = {k: Fraction(str(v)) for k, v in constraint["coefficients"].items()}
        values = list(fractions.values()) + [Fraction(str(v)) for v in bounds if v is not None]
        denominator = math.lcm(*(v.denominator for v in values))
        divisor = math.gcd(*(int(v * denominator) for v in values)) or 1
        scale = Fraction(denominator, divisor)
        coefficients = {k: int(v * scale) for k, v in fractions.items()}
        lower, upper = [None if v is None else int(Fraction(str(v)) * scale) for v in bounds]
    return coefficients, lower, upper


def model_qubo_dimension(model, *, max_variables=None):
    dim = 0

    def add(span):
        nonlocal dim
        if span < 0:
            raise ValueError("Empty integer domain / constraint has no feasible assignment")
        dim += int(span).bit_length()
        if max_variables is not None and max(dim, 1) > max_variables:
            raise ValueError(f"QUBO exceeds max_variables={max_variables}; choose a smaller instance")

    for name, (lower, upper) in model["variables"].items():
        if not all(math.isfinite(v) and v == int(v) for v in (lower, upper)):
            raise ValueError(f"{name}: finite integral bounds are required")
        add(upper - lower)
    for constraint in model.get("constraints", []):
        coefficients, lower, upper = _scaled_constraint(constraint)
        if lower == upper:
            continue
        minimum = sum(c * model["variables"][k][0 if c >= 0 else 1] for k, c in coefficients.items())
        maximum = sum(c * model["variables"][k][1 if c >= 0 else 0] for k, c in coefficients.items())
        if upper is not None:
            add(upper - minimum)
        if lower is not None:
            add(maximum - lower)
    add(0)
    return max(dim, 1)


def native_qubo_dimension(slug, instance):
    if slug == "labs":
        return instance * (instance + 1) // 2
    if slug == "marketsplit":
        return max(1, instance.n_cols)
    if slug == "independentset":
        return max(1, instance.n + len(instance.edges))
    if slug == "birkhoff":
        return max(
            1,
            sum(
                math.factorial(int(data["n"])) * (1 + 2 * int(data["scale"]).bit_length())
                for data in instance.instances.values()
            ),
        )
    raise ValueError(f"No native model for {slug}")


def encode_model(model: dict, *, penalty=None, max_variables=4096) -> dict:
    model_qubo_dimension(model, max_variables=max_variables)
    coefficients = defaultdict(float)
    offset = float(model.get("constant", 0))
    dim = 0
    mapping = {}

    def encode(name, lower, upper):
        nonlocal dim
        if not all(math.isfinite(v) and v == int(v) for v in (lower, upper)):
            raise ValueError(f"{name}: finite integral bounds are required")
        weights = integer_weights(int(upper - lower))
        if dim + len(weights) > max_variables:
            raise ValueError(f"QUBO exceeds max_variables={max_variables}; choose a smaller instance")
        ids = list(range(dim, dim + len(weights)))
        dim += len(weights)
        return {"lower": int(lower), "bits": ids, "weights": weights}

    for name, bounds in model["variables"].items():
        mapping[name] = encode(name, *bounds)

    def affine(coefficients, constant=0):
        linear = defaultdict(float)
        for name, coefficient in coefficients.items():
            entry = mapping[name]
            constant += coefficient * entry["lower"]
            for idx, weight in zip(entry["bits"], entry["weights"]):
                linear[idx] += coefficient * weight
        return dict(linear), constant

    def product(a, ac, b, bc, scale):
        nonlocal offset
        offset += scale * ac * bc
        for i, value in a.items():
            coefficients[i, i] += scale * value * bc
        for j, value in b.items():
            coefficients[j, j] += scale * value * ac
        for i, av in a.items():
            for j, bv in b.items():
                coefficients[min(i, j), max(i, j)] += scale * av * bv

    sign = -1 if model.get("sense", "minimize") == "maximize" else 1
    offset *= sign
    a, ac = affine(model.get("linear", {}))
    product(a, ac, {}, 1, sign)
    for u, v, coefficient in model.get("quadratic", []):
        a, ac = affine({u: 1})
        b, bc = affine({v: 1})
        product(a, ac, b, bc, sign * coefficient)
    automatic = 1 + sum(abs(v) for v in coefficients.values())
    penalty = automatic if penalty is None else float(penalty)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")

    for index, constraint in enumerate(model.get("constraints", [])):
        row_coefficients, lower, upper = _scaled_constraint(constraint)
        a, ac = affine(row_coefficients)
        minimum = ac + sum(min(0, v) for v in a.values())
        maximum = ac + sum(max(0, v) for v in a.values())
        for direction, bound in [(1, upper), (-1, lower)]:
            if bound is None:
                continue
            if lower == upper and direction == -1:
                continue
            residual = {i: direction * v for i, v in a.items()}
            constant = direction * (ac - bound)
            if lower != upper:
                slack_span = int(direction * (bound - (minimum if direction == 1 else maximum)))
                if slack_span < 0:
                    raise ValueError(f"Constraint {index} has no feasible assignment")
                slack = encode(f"slack_{index}_{direction}", 0, slack_span)
                residual.update(dict(zip(slack["bits"], slack["weights"])))
            product(residual, constant, residual, constant, penalty)
    return _qubo_record(
        max(dim, 1),
        coefficients,
        offset,
        {"kind": "integer_model", "mapping": mapping, "model": model, "penalty": penalty},
    )


def _qubo_record(dim, coefficients, offset, metadata):
    if not math.isfinite(offset) or any(not math.isfinite(v) for v in coefficients.values()):
        raise ValueError("Nonfinite QUBO coefficient")
    terms = [[i, j, float(v)] for (i, j), v in sorted(coefficients.items()) if v]
    return {"dim": dim, "qubo_mat": terms, "offset": float(offset), "bridge": metadata}


def to_qubo(record):
    n = record["dim"]
    mat = np.zeros((n, n))
    for i, j, value in record["qubo_mat"]:
        mat[i, j] += value
    return QUBO(mat, np.arange(n), np.arange(n), offset=record["offset"])


def read_lp(path: Path, *, max_variables=None) -> dict:
    from pyscipopt import Model

    scip = Model()
    scip.hideOutput()
    scip.readProblem(str(path))
    variables = scip.getVars()
    if (
        max_variables is not None
        and sum(
            v.vtype() in {"BINARY", "INTEGER", "IMPLINT"} and v.getUbOriginal() > v.getLbOriginal() for v in variables
        )
        > max_variables
    ):
        raise ValueError(f"LP requires more than max_variables={max_variables} QUBO variables")
    objective = {v.name: float(v.getObj()) for v in variables if v.getObj()}
    quadratic = []
    auxiliary = None
    constraints = []
    for c in scip.getConss():
        if c.getConshdlrName() == "linear":
            constraints.append(
                {
                    "coefficients": scip.getValsLinear(c),
                    "lower": None if scip.isInfinity(-scip.getLhs(c)) else scip.getLhs(c),
                    "upper": None if scip.isInfinity(scip.getRhs(c)) else scip.getRhs(c),
                }
            )
        elif c.name == "quadobj" and scip.checkQuadraticNonlinear(c):
            bilinear, squares, linear = scip.getTermsQuadratic(c)
            if len(linear) != 1 or linear[0][0].name != "quadobjvar" or linear[0][1] != -1:
                raise ValueError("Unrecognized SCIP quadratic objective epigraph")
            auxiliary = linear[0][0].name
            if objective.pop(auxiliary, None) != 1 or scip.getRhs(c) != 0:
                raise ValueError("Unrecognized SCIP quadratic objective")
            quadratic.extend((u.name, v.name, value) for u, v, value in bilinear)
            for v, square, lin in squares:
                if square:
                    quadratic.append((v.name, v.name, square))
                if lin:
                    objective[v.name] = objective.get(v.name, 0) + lin
        else:
            raise ValueError(f"Unsupported {c.getConshdlrName()} constraint {c.name}")
    bounds = {}
    for v in variables:
        if v.name == auxiliary:
            continue
        if v.vtype() not in {"BINARY", "INTEGER", "IMPLINT"}:
            raise ValueError(f"Continuous variable {v.name}: an explicit integer formulation is required")
        bounds[v.name] = [float(v.getLbOriginal()), float(v.getUbOriginal())]
    return {
        "variables": bounds,
        "linear": objective,
        "quadratic": quadratic,
        "constant": scip.getObjoffset(),
        "sense": scip.getObjectiveSense(),
        "constraints": constraints,
    }


def native_model(slug, instance, *, max_variables=4096):
    if native_qubo_dimension(slug, instance) > max_variables:
        raise ValueError(f"{slug} formulation exceeds max_variables={max_variables}")
    model = {"variables": {}, "linear": {}, "constraints": []}
    if slug == "marketsplit":
        names = [f"x#{i + 1}" for i in range(instance.n_cols)]
        model["variables"] = dict.fromkeys(names, [0, 1])
        for row, rhs in zip(instance.A, instance.b):
            model["constraints"].append({"coefficients": dict(zip(names, row)), "lower": rhs, "upper": rhs})
    elif slug == "independentset":
        model["variables"] = {f"x#{i + 1}": [0, 1] for i in range(instance.n)}
        model["linear"] = dict.fromkeys(model["variables"], -1)
        for u, v in instance.edges:
            model["constraints"].append({"coefficients": {f"x#{u}": 1, f"x#{v}": 1}, "upper": 1})
    elif slug == "birkhoff":
        permutation_map = {}
        for key, data in instance.instances.items():
            n, scale = int(data["n"]), int(data["scale"])
            permutations = list(itertools.permutations(range(1, n + 1)))
            permutation_map[key] = permutations
            weights = [f"w${key}#{i}" for i in range(len(permutations))]
            for i, w in enumerate(weights):
                z = f"z${key}#{i}"
                model["variables"].update({w: [0, scale], z: [0, 1]})
                model["linear"][z] = 1
                model["constraints"].append({"coefficients": {w: 1, z: -scale}, "upper": 0})
            for row in range(n):
                for col in range(1, n + 1):
                    rhs = data["scaled_doubly_stochastic_matrix"][row * n + col - 1]
                    model["constraints"].append(
                        {
                            "coefficients": {w: 1 for w, p in zip(weights, permutations) if p[row] == col},
                            "lower": rhs,
                            "upper": rhs,
                        }
                    )
        model["permutations"] = permutation_map
    else:
        raise ValueError(f"No native model for {slug}")
    return model


def labs_record(n, *, penalty=None, max_variables=4096):
    if n < 1 or n > max_variables:
        raise ValueError("Invalid LABS length / max_variables")
    if native_qubo_dimension("labs", n) > max_variables:
        raise ValueError("LABS quadratization exceeds max_variables")
    model = {"variables": {f"x#{i}": [0, 1] for i in range(n)}, "constraints": []}
    for i in range(n):
        for j in range(i + 1, n):
            model["variables"][f"p#{i}#{j}"] = [0, 1]
    coefficients = defaultdict(float)
    offset = 0.0
    names = list(model["variables"])
    ids = {name: i for i, name in enumerate(names)}
    for lag in range(1, n):
        a = defaultdict(int)
        for i in range(n - lag):
            a[ids[f"p#{i}#{i + lag}"]] += 4
            a[ids[f"x#{i}"]] -= 2
            a[ids[f"x#{i + lag}"]] -= 2
        constant = n - lag
        offset += constant**2
        for i, v in a.items():
            coefficients[i, i] += v * v + 2 * constant * v
        for (i, v), (j, w) in itertools.combinations(a.items(), 2):
            coefficients[min(i, j), max(i, j)] += 2 * v * w
    strength = 1 + sum(abs(v) for v in coefficients.values())
    strength = strength if penalty is None else float(penalty)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("penalty must be finite and positive")
    for i in range(n):
        for j in range(i + 1, n):
            x, y, p = ids[f"x#{i}"], ids[f"x#{j}"], ids[f"p#{i}#{j}"]
            coefficients[x, y] += strength
            coefficients[x, p] -= 2 * strength
            coefficients[y, p] -= 2 * strength
            coefficients[p, p] += 3 * strength
    return _qubo_record(
        len(names),
        coefficients,
        offset,
        {
            "kind": "labs",
            "n": n,
            "penalty": strength,
            "mapping": {name: {"lower": 0, "bits": [i], "weights": [1]} for name, i in ids.items()},
        },
    )


def decode(record, bitstring):
    if len(bitstring) != record["dim"] or set(bitstring) - {"0", "1"}:
        raise ValueError("Expected a complete binary assignment of the original QUBO")
    bits = list(map(int, bitstring))
    return {
        name: entry["lower"] + sum(bits[i] * w for i, w in zip(entry["bits"], entry["weights"]))
        for name, entry in record["bridge"]["mapping"].items()
    }


def model_evaluation(model, values):
    objective = model.get("constant", 0) + sum(c * values[v] for v, c in model.get("linear", {}).items())
    objective += sum(c * values[u] * values[v] for u, v, c in model.get("quadratic", []))
    violations = []
    for i, row in enumerate(model.get("constraints", [])):
        value = sum(c * values[v] for v, c in row["coefficients"].items())
        if row.get("lower") is not None and value < row["lower"] - 1e-7:
            violations.append(f"constraint {i}: {value} < {row['lower']}")
        if row.get("upper") is not None and value > row["upper"] + 1e-7:
            violations.append(f"constraint {i}: {value} > {row['upper']}")
    return float(objective), violations
