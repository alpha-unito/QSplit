#include "core.hpp"

namespace qs {
namespace {
bool binary(double x) { return x == 0 || x == 1; }
double clean(double x) { return std::isfinite(x) ? x : 0; }
void set_column(py::object frame, py::object label, py::object values) {
    auto columns = frame.attr("columns");
    if (columns.attr("__contains__")(label).cast<bool>())
        frame.attr("isetitem")(columns.attr("get_loc")(label), values);
    else
        frame.attr("insert")(py::len(columns), label, values);
}
bool integer_rows(py::object df) {
    auto kind = df.attr("to_numpy")().attr("dtype").attr("kind").cast<std::string>();
    return kind == "i" || kind == "u";
}
Array table_array(py::object df) { return df.attr("to_numpy")("dtype"_a = "float64").cast<Array>(); }
std::vector<double> merge_row(const double *a, const double *b, py::ssize_t size, py::ssize_t energy_col) {
    std::vector<double> out(size);
    bool conflict = false;
    for (py::ssize_t j = 0; j < size; ++j) {
        if (j == energy_col) {
            if (conflict || (std::isnan(a[j]) && std::isnan(b[j])))
                out[j] = nan;
            else if (std::isnan(a[j]))
                out[j] = b[j];
            else if (std::isnan(b[j]))
                out[j] = a[j];
            else
                out[j] = a[j] + b[j];
        } else {
            if (std::isnan(a[j]) || std::isnan(b[j]))
                out[j] = nan;
            else if (std::isinf(a[j]) && std::isinf(b[j]))
                out[j] = inf;
            else if (std::isinf(a[j]))
                out[j] = b[j];
            else if (std::isinf(b[j]) || a[j] == b[j])
                out[j] = a[j];
            else
                out[j] = nan;
        }
        conflict = conflict || std::isnan(out[j]);
    }
    return out;
}
py::object recalculate(py::object df, const Qubo &q, bool validate = false) {
    Table t(df, true);
    View v(q);
    Array energies(t.values.shape(0));
    auto out = energies.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < out.shape(0); ++i) {
            auto assignment = t.assignment(i);
            if (validate) {
                for (py::ssize_t j = 0; j < v.n; ++j)
                    for (auto id : {v.r(j), v.c(j)})
                        if (id >= 0 && (!assignment.contains(id) || !binary(assignment.at(id))))
                            throw py::value_error("Refinement requires binary assignments for every real variable");
            } else {
                for (auto &[id, x] : assignment)
                    x = clean(x);
            }
            out(i) = energy(v, assignment) + q.offset;
        }
    }
    set_column(df, py::str("energy"), energies);
    return df;
}
} // namespace

py::object validate_solutions(py::object df, const Qubo &q) {
    if (df.attr("empty").cast<bool>())
        throw py::value_error("Refinement requires a complete global solution");
    return recalculate(df, q, true);
}

py::object aggregate_votes(py::list subs, py::object obj, const std::string &mode) {
    auto &q = obj.cast<Qubo &>();
    bool bp = mode == "linear_belief_propagation", linear = mode == "linear";
    bool graph = mode == "recursive_graph", interactions = mode == "k_interactions";
    bool quad = mode == "quadtree";
    if (!bp && !linear && !graph && !interactions && !quad)
        throw py::value_error("Unknown aggregation mode");
    auto ids = variables(q, graph || interactions);
    std::map<Id, double> ones, zeros, totals, first;
    for (auto id : ids) {
        ones[id] = 0;
        totals[id] = 0;
    }
    py::ssize_t center = 0;
    for (auto obj_sub : subs) {
        auto &sub = obj_sub.cast<Qubo &>();
        Table t(sub.solutions);
        auto a = t.values.unchecked<2>();
        auto best = linear ? -1 : t.best(bp || quad);
        if (!linear && best < 0) {
            ++center;
            continue;
        }
        std::vector<double> weights(a.shape(0), 0);
        if (bp) {
            auto min_energy = a(best, t.energy_col);
            double sum = 0;
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
                if (std::isfinite(a(i, t.energy_col))) {
                    weights[i] = std::exp(-(a(i, t.energy_col) - min_energy));
                    sum += weights[i];
                }
            for (auto &w : weights)
                w /= sum;
        } else if (linear)
            std::fill(weights.begin(), weights.end(), 1);
        else
            weights[best] = 1;
        auto row_ids = positions(sub.rows_idx);
        auto quad_center = sub.rows_idx.size() ? sub.rows_idx.at(0) : Id(-1);
        {
            py::gil_scoped_release release;
            for (auto [id, col] : t.by_id) {
                if (!ones.contains(id) || (quad && !row_ids.contains(id)))
                    continue;
                double multiplier = (interactions && id == center) || (quad && id == quad_center) ? 2 : 1;
                for (py::ssize_t i = 0; i < a.shape(0); ++i) {
                    auto value = a(i, col);
                    if (!weights[i] || !binary(value))
                        continue;
                    if (!first.contains(id))
                        first[id] = value;
                    auto weight = weights[i] * multiplier;
                    ones[id] += weight * value;
                    zeros[id] += weight * (1 - value);
                    totals[id] += weight;
                }
            }
        }
        ++center;
    }
    std::map<Id, double> assignment;
    for (auto id : ids) {
        auto one = ones[id], total = totals[id];
        if (bp)
            assignment[id] = one >= zeros[id];
        else if (graph && total && one == total / 2)
            assignment[id] = first.at(id);
        else
            assignment[id] = one > total / 2;
    }
    q.solutions = assignment_frame(q, assignment);
    return obj;
}

py::object fill_inf(py::object schema, py::object df) {
    return df.attr("reindex")("columns"_a = schema, "fill_value"_a = inf);
}
py::object combine_ul_lr(const Qubo &ul, const Qubo &lr) {
    std::set<Id> all;
    for (auto ids : {ul.rows_idx, lr.cols_idx}) {
        auto a = ids.unchecked<1>();
        for (py::ssize_t i = 0; i < a.shape(0); ++i)
            all.insert(a(i));
    }
    py::list schema;
    for (auto id : all)
        schema.append(id);
    schema.append("energy");
    // Pandas already implements joins in native code; retain its dtype semantics.
    auto result = ul.solutions.attr("merge")(lr.solutions, "how"_a = "cross");
    set_column(result, py::str("energy"), result[py::str("energy_x")].attr("__add__")(result[py::str("energy_y")]));
    result = result.attr("drop")("columns"_a = py::cast(std::vector<std::string>{"energy_x", "energy_y"}));
    return fill_inf(schema, result);
}
std::vector<double> combine_rows(py::object a, py::object b) {
    Array av = a.attr("to_numpy")("dtype"_a = "float64");
    Array bv = b.attr("to_numpy")("dtype"_a = "float64");
    auto x = av.unchecked<1>(), y = bv.unchecked<1>();
    if (x.shape(0) != y.shape(0))
        throw py::value_error("Rows must have matching lengths");
    std::vector<double> left(x.shape(0)), right(x.shape(0));
    for (py::ssize_t i = 0; i < x.shape(0); ++i) {
        left[i] = x(i);
        right[i] = y(i);
    }
    auto col = a.attr("index").attr("get_loc")("energy").cast<py::ssize_t>();
    return merge_row(left.data(), right.data(), x.shape(0), col);
}
double distance(Array a, Array b) {
    auto x = a.unchecked<1>(), y = b.unchecked<1>();
    if (x.shape(0) != y.shape(0))
        throw py::value_error("Assignments must have matching lengths");
    py::gil_scoped_release release;
    double count = 0, different = 0;
    for (py::ssize_t i = 0; i < x.shape(0); ++i)
        if (std::isfinite(x(i)) && std::isfinite(y(i))) {
            ++count;
            different += x(i) != y(i);
        }
    return count ? different / count : inf;
}
py::object closest(py::object start, py::object candidates) {
    auto aa = table_array(start), bb = table_array(candidates);
    auto a = aa.unchecked<2>(), b = bb.unchecked<2>();
    if (a.shape(1) != b.shape(1))
        throw py::value_error("Assignments must have matching columns");
    if (a.shape(0) && !b.shape(0))
        throw py::value_error("No candidate assignments");
    Array result({a.shape(0), a.shape(1)});
    auto out = result.mutable_unchecked<2>();
    auto energy_col = candidates.attr("columns").attr("get_loc")("energy").cast<py::ssize_t>();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            double minimum = inf;
            py::ssize_t best = 0;
            for (py::ssize_t j = 0; j < b.shape(0); ++j) {
                double count = 0, different = 0;
                for (py::ssize_t k = 0; k < a.shape(1) - 1; ++k)
                    if (std::isfinite(a(i, k)) && std::isfinite(b(j, k))) {
                        ++count;
                        different += a(i, k) != b(j, k);
                    }
                auto d = count ? different / count : inf;
                if (d < minimum) {
                    minimum = d;
                    best = j;
                }
            }
            bool missing = false;
            for (py::ssize_t k = 0; k < a.shape(1); ++k) {
                out(i, k) = b(best, k);
                missing = missing || std::isnan(out(i, k));
            }
            if (missing)
                out(i, energy_col) = nan;
        }
    }
    auto frame = dataframe(result, start.attr("columns"));
    if (integer_rows(candidates))
        return frame.attr("astype")(candidates.attr("to_numpy")().attr("dtype"));
    return frame;
}

py::object resolve_conflicts(py::object df, const Qubo &q, py::function solve, int exact_limit) {
    if (exact_limit < 0 || exact_limit > 20)
        throw py::value_error("Exact conflict limit must be between 0 and 20");
    Table t(df, true);
    View v(q);
    auto a = t.values.unchecked<2>();
    Array result({a.shape(0), a.shape(1)});
    auto out = result.mutable_unchecked<2>();
    std::set<py::ssize_t> changed;
    for (py::ssize_t row = 0; row < a.shape(0); ++row) {
        auto assignment = t.assignment(row);
        std::vector<Id> missing;
        for (auto &[id, value] : assignment) {
            if (id >= 0 && std::isnan(value))
                missing.push_back(id);
            value = clean(value);
        }
        if (!missing.empty()) {
            std::map<Id, py::ssize_t> local;
            for (std::size_t i = 0; i < missing.size(); ++i)
                local[missing[i]] = i;
            auto matrix = zeros(missing.size());
            auto m = matrix.mutable_unchecked<2>();
            {
                py::gil_scoped_release release;
                for (py::ssize_t i = 0; i < v.n; ++i) {
                    if (v.r(i) < 0)
                        continue;
                    for (py::ssize_t j = 0; j < v.n; ++j) {
                        if (v.c(j) < 0 || v.m(i, j) == 0)
                            continue;
                        bool r = local.contains(v.r(i)), c = local.contains(v.c(j));
                        if (r && c)
                            m(local.at(v.r(i)), local.at(v.c(j))) += v.m(i, j);
                        else if (r)
                            m(local.at(v.r(i)), local.at(v.r(i))) += v.m(i, j) * assignment[v.c(j)];
                        else if (c)
                            m(local.at(v.c(j)), local.at(v.c(j))) += v.m(i, j) * assignment[v.r(i)];
                    }
                }
            }
            Qubo sub(matrix, indices(missing), indices(missing));
            View s(sub);
            if (!empty(sub)) {
                if (missing.size() <= static_cast<std::size_t>(exact_limit)) {
                    py::gil_scoped_release release;
                    double best_energy = inf;
                    std::uint64_t best = 0, count = std::uint64_t{1} << missing.size();
                    for (std::uint64_t bits = 0; bits < count; ++bits) {
                        double e = 0;
                        for (std::size_t j = 0; j < missing.size(); ++j) {
                            double column = 0;
                            for (std::size_t i = 0; i < missing.size(); ++i)
                                column += static_cast<double>((bits >> (missing.size() - 1 - i)) & 1) * s.m(i, j);
                            e += column * static_cast<double>((bits >> (missing.size() - 1 - j)) & 1);
                        }
                        if (e < best_energy) {
                            best_energy = e;
                            best = bits;
                        }
                    }
                    for (std::size_t i = 0; i < missing.size(); ++i)
                        assignment[missing[i]] = (best >> (missing.size() - 1 - i)) & 1;
                } else {
                    Table samples(solve(py::cast(std::move(sub))));
                    auto best = samples.assignment(samples.best());
                    for (auto id : missing)
                        assignment[id] = best.contains(id) ? best.at(id) : 0;
                }
            }
        }
        for (py::ssize_t col = 0; col < a.shape(1); ++col)
            out(row, col) = a(row, col);
        for (auto id : missing) {
            auto col = t.by_id.at(id);
            out(row, col) = assignment[id];
            changed.insert(col);
        }
        if (t.energy_col >= 0) {
            py::gil_scoped_release release;
            out(row, t.energy_col) = energy(v, assignment) + q.offset;
        }
    }
    auto frame = df.attr("copy")();
    if (t.energy_col >= 0)
        changed.insert(t.energy_col);
    for (auto col : changed) {
        Array column(a.shape(0));
        for (py::ssize_t row = 0; row < a.shape(0); ++row)
            column.mutable_at(row) = out(row, col);
        set_column(frame, t.columns[col], column);
    }
    if (t.energy_col < 0)
        frame = recalculate(frame, q);
    return frame;
}

py::object aggregate_recursive(py::tuple subs, py::object obj, py::function resolve) {
    if (py::len(subs) != 3)
        throw py::value_error("Recursive aggregation requires three subproblems");
    auto start = combine_ul_lr(subs[0].cast<Qubo &>(), subs[2].cast<Qubo &>());
    auto columns = start.attr("columns");
    if (columns.attr("__contains__")(-1).cast<bool>())
        set_column(start, py::int_(-1), py::int_(0));
    auto candidate = closest(start, fill_inf(columns, subs[1].cast<Qubo &>().solutions));
    auto aa = table_array(start), bb = table_array(candidate);
    auto a = aa.unchecked<2>(), b = bb.unchecked<2>();
    auto energy_col = columns.attr("get_loc")("energy").cast<py::ssize_t>();
    Array result({a.shape(0), a.shape(1)});
    auto out = result.mutable_unchecked<2>();
    bool left_int = integer_rows(start), right_int = integer_rows(candidate);
    std::vector<bool> integer_columns(a.shape(1), true);
    {
        py::gil_scoped_release release;
        std::vector<double> left(a.shape(1)), right(a.shape(1));
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            for (py::ssize_t j = 0; j < a.shape(1); ++j) {
                left[j] = a(i, j);
                right[j] = b(i, j);
            }
            auto merged = merge_row(left.data(), right.data(), a.shape(1), energy_col);
            for (py::ssize_t j = 0; j < a.shape(1); ++j) {
                out(i, j) = merged[j];
                bool from_right = std::isinf(left[j]) && std::isfinite(right[j]);
                integer_columns[j] = integer_columns[j] && std::isfinite(merged[j]) &&
                                     (j == energy_col ? left_int && right_int
                                      : from_right    ? right_int
                                                      : left_int);
            }
        }
    }
    auto combined = dataframe(result, columns);
    for (py::ssize_t j = 0; j < a.shape(1); ++j)
        if (integer_columns[j]) {
            auto col = columns[py::int_(j)];
            set_column(combined, col, combined[col].attr("astype")("int64"));
        }
    auto df = resolve(combined, obj)
                  .attr("reset_index")("drop"_a = true)
                  .attr("drop_duplicates")()
                  .attr("nsmallest")("n"_a = 10, "columns"_a = "energy");
    obj.cast<Qubo &>().solutions = df.attr("drop")("columns"_a = py::cast(std::vector<int>{-1}), "errors"_a = "ignore");
    return obj;
}
py::object aggregate_trivial(const Qubo &ul, const Qubo &lr, py::object obj) {
    auto &q = obj.cast<Qubo &>();
    py::list schema;
    for (auto id : variables(q))
        schema.append(id);
    schema.append("energy");
    auto df = recalculate(fill_inf(schema, combine_ul_lr(ul, lr)), q)
                  .attr("reset_index")("drop"_a = true)
                  .attr("drop_duplicates")();
    if (!df.attr("empty").cast<bool>()) {
        auto energies = df[py::str("energy")];
        df = df[energies.attr("__eq__")(energies.attr("min")())].attr("reset_index")("drop"_a = true);
    }
    q.solutions = df;
    return obj;
}
py::object aggregate_leaves(py::list subs, py::object obj, py::function resolve) {
    auto deepcopy = py::module_::import("copy").attr("deepcopy");
    std::function<py::object(py::object)> visit = [&](py::object tree) -> py::object {
        if (py::isinstance<py::int_>(tree))
            return subs[tree.cast<py::ssize_t>()];
        auto pair = tree.cast<py::tuple>();
        auto children = pair[1].cast<py::tuple>();
        py::tuple results(py::len(children));
        for (py::ssize_t i = 0; i < py::len(children); ++i)
            results[i] = visit(py::reinterpret_borrow<py::object>(children[i]));
        return aggregate_recursive(results, deepcopy(pair[0]), resolve);
    };
    obj.cast<Qubo &>().solutions = visit(obj.attr("split_tree")).cast<Qubo &>().solutions;
    return obj;
}
py::tuple logical_expansion(py::tuple subs) {
    auto &ul = subs[0].cast<Qubo &>(), &ur = subs[1].cast<Qubo &>(), &lr = subs[2].cast<Qubo &>();
    if (ur.solutions.is_none() || ur.solutions.attr("empty").cast<bool>() ||
        !ur.solutions.attr("columns").attr("__contains__")("energy").cast<bool>())
        return subs;
    Table t(ur.solutions);
    auto a = t.values.unchecked<2>();
    auto best = t.best();
    if (best < 0)
        return subs;
    std::map<Id, double> hints;
    for (auto [id, col] : t.by_id) {
        if (id < 0)
            continue;
        double sum = 0, count = 0;
        for (py::ssize_t i = 0; i < a.shape(0); ++i)
            if (a(i, t.energy_col) == a(best, t.energy_col) && std::isfinite(a(i, col))) {
                sum += a(i, col);
                ++count;
            }
        if (count)
            hints[id] = sum / count;
    }
    View u(ul), v(ur), l(lr);
    auto um = ul.mat.mutable_unchecked<2>(), lm = lr.mat.mutable_unchecked<2>();
    auto up = positions(ul.rows_idx), lp = positions(lr.rows_idx);
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < v.n; ++i)
        for (py::ssize_t j = 0; j < v.n; ++j) {
            if (v.m(i, j) == 0)
                continue;
            if (v.r(i) >= 0 && up.contains(v.r(i)) && hints.contains(v.c(j)))
                um(up.at(v.r(i)), up.at(v.r(i))) += v.m(i, j) * hints.at(v.c(j));
            if (v.c(j) >= 0 && lp.contains(v.c(j)) && hints.contains(v.r(i)))
                lm(lp.at(v.c(j)), lp.at(v.c(j))) += v.m(i, j) * hints.at(v.r(i));
        }
    return subs;
}
double calculate_energy(Array x, Array mat) {
    auto a = x.unchecked<1>();
    auto m = mat.unchecked<2>();
    if (m.shape(0) != a.shape(0) || m.shape(1) != a.shape(0))
        throw py::value_error("Incompatible energy dimensions");
    py::gil_scoped_release release;
    double e = 0;
    for (py::ssize_t j = 0; j < a.shape(0); ++j) {
        double column = 0;
        for (py::ssize_t i = 0; i < a.shape(0); ++i)
            column += (std::isinf(a(i)) ? 0 : a(i)) * m(i, j);
        e += column * (std::isinf(a(j)) ? 0 : a(j));
    }
    return e;
}
} // namespace qs
