#include "core.hpp"

namespace qs {
Qubo condition(const Qubo &q, const std::vector<Id> &window, const std::map<Id, double> &beliefs) {
    View v(q);
    auto rows = positions(q.rows_idx), cols = positions(q.cols_idx);
    std::set<Id> inside(window.begin(), window.end());
    std::vector<Id> outside;
    for (py::ssize_t i = 0; i < v.n; ++i)
        if (v.r(i) >= 0 && !inside.contains(v.r(i)))
            outside.push_back(v.r(i));
    for (auto id : window)
        if (!rows.contains(id) || !cols.contains(id))
            throw py::key_error("Unknown refinement variable");
    for (auto id : outside)
        if (!cols.contains(id) || !beliefs.contains(id))
            throw py::key_error("Missing outside belief or column");
    std::vector<py::ssize_t> window_rows, window_cols, outside_rows, outside_cols;
    std::vector<double> probabilities;
    for (auto id : window) {
        window_rows.push_back(rows.at(id));
        window_cols.push_back(cols.at(id));
    }
    for (auto id : outside) {
        outside_rows.push_back(rows.at(id));
        outside_cols.push_back(cols.at(id));
        probabilities.push_back(beliefs.at(id));
    }
    auto matrix = zeros(window.size());
    auto out = matrix.mutable_unchecked<2>();
    double offset = q.offset;
    {
        py::gil_scoped_release release;
        for (std::size_t i = 0; i < window.size(); ++i) {
            for (std::size_t j = 0; j < window.size(); ++j)
                out(i, j) = v.m(window_rows[i], window_cols[j]);
            double coupling = 0;
            for (std::size_t j = 0; j < outside.size(); ++j)
                coupling +=
                    (v.m(window_rows[i], outside_cols[j]) + v.m(outside_rows[j], window_cols[i])) * probabilities[j];
            out(i, i) += coupling;
        }
        std::vector<double> contracted(outside.size(), 0);
        for (std::size_t i = 0; i < outside.size(); ++i)
            for (std::size_t j = 0; j < outside.size(); ++j)
                contracted[j] += probabilities[i] * v.m(outside_rows[i], outside_cols[j]);
        double outside_energy = 0, correction = 0;
        for (std::size_t j = 0; j < outside.size(); ++j) {
            auto p = probabilities[j];
            outside_energy += contracted[j] * p;
            correction += v.m(outside_rows[j], outside_cols[j]) * (p - p * p);
        }
        offset += outside_energy;
        offset += correction;
    }
    return Qubo(matrix, indices(window), indices(window), offset);
}

std::map<Id, double> collect_beliefs(py::list subs, const Qubo &q) {
    Table global(q.solutions);
    auto assignment = global.assignment(global.best());
    std::map<Id, double> beliefs, votes, counts;
    for (auto id : variables(q)) {
        if (!assignment.contains(id))
            throw py::key_error("Missing global assignment");
        beliefs[id] = assignment.at(id);
    }
    for (auto obj : subs) {
        auto &sub = obj.cast<Qubo &>();
        auto df = sub.solutions;
        if (df.is_none() || df.attr("empty").cast<bool>() ||
            !df.attr("columns").attr("__contains__")("energy").cast<bool>())
            continue;
        Table local(df);
        auto best = local.best(true);
        if (best < 0)
            continue;
        auto ids = variables(sub);
        auto a = local.values.unchecked<2>();
        py::gil_scoped_release release;
        for (auto id : ids) {
            if (!beliefs.contains(id) || !local.by_id.contains(id))
                continue;
            auto col = local.by_id.at(id);
            double sum = 0, count = 0;
            for (py::ssize_t row = 0; row < a.shape(0); ++row)
                if (a(row, local.energy_col) == a(best, local.energy_col) && (a(row, col) == 0 || a(row, col) == 1)) {
                    sum += a(row, col);
                    ++count;
                }
            if (count) {
                votes[id] += sum / count;
                ++counts[id];
            }
        }
    }
    for (auto &[id, value] : beliefs)
        if (counts[id])
            value = (value + votes[id] / counts[id]) / 2;
    return beliefs;
}

namespace {
py::list soft_consensus(py::list subs, const Qubo &q, double strength, const std::map<Id, double> &beliefs);

std::vector<std::vector<Id>> real_windows(py::list subs, int block) {
    if (block < 0)
        throw py::value_error("Refinement block_size must be non-negative");
    std::set<std::vector<Id>> seen;
    std::vector<std::vector<Id>> result;
    for (auto obj : subs) {
        auto &sub = obj.cast<Qubo &>();
        View validated(sub);
        for (auto ids : {sub.rows_idx, sub.cols_idx}) {
            auto a = ids.unchecked<1>();
            std::vector<Id> window;
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
                if (a(i) >= 0)
                    window.push_back(a(i));
            auto size = block ? static_cast<std::size_t>(block) : window.size();
            for (std::size_t start = 0; start < window.size(); start += size) {
                std::vector<Id> part(window.begin() + start, window.begin() + std::min(window.size(), start + size));
                if (seen.insert(part).second)
                    result.push_back(std::move(part));
            }
        }
    }
    return result;
}
} // namespace

py::list refine_mean_field(py::list subs, const Qubo &q, int block) {
    auto beliefs = collect_beliefs(subs, q);
    py::list result;
    for (const auto &window : real_windows(subs, block))
        result.append(condition(q, window, beliefs));
    return result;
}

py::list refine_linear(py::list subs, const Qubo &q) { return refine_mean_field(subs, q, 0); }

py::list refine_soft_consensus(py::list subs, const Qubo &q, double strength, int block) {
    py::list templates, real;
    for (auto obj : subs) {
        auto members = py::getattr(obj, "macro_members", py::dict()).cast<py::dict>();
        if (py::len(members)) {
            auto &sub = obj.cast<Qubo &>();
            View s(sub);
            for (py::ssize_t i = 0; i < s.n; ++i)
                if (s.r(i) != s.c(i))
                    throw py::value_error("Macrovariable refinement requires principal templates");
            if (block > 0 && s.n > block)
                throw py::value_error("Macrovariable template exceeds refinement block_size");
            templates.append(obj);
        } else
            real.append(obj);
    }
    auto rows = positions(q.rows_idx), cols = positions(q.cols_idx);
    for (const auto &window : real_windows(real, block)) {
        std::vector<py::ssize_t> rr, cc;
        for (auto id : window) {
            if (!rows.contains(id) || !cols.contains(id))
                throw py::key_error("Unknown refinement variable");
            rr.push_back(rows.at(id));
            cc.push_back(cols.at(id));
        }
        templates.append(subqubo(q, rr, cc, q.offset));
    }
    return soft_consensus(templates, q, strength, collect_beliefs(subs, q));
}

namespace {
py::list soft_consensus(py::list subs, const Qubo &q, double strength, const std::map<Id, double> &beliefs) {
    if (!std::isfinite(strength) || strength < 0)
        throw py::value_error("REFINEMENT_STRENGTH must be finite and non-negative");
    View v(q);
    auto rows = positions(q.rows_idx), cols = positions(q.cols_idx);
    py::list result;
    for (auto obj : subs) {
        auto &sub = obj.cast<Qubo &>();
        View s(sub);
        auto members = py::getattr(obj, "macro_members", py::dict()).cast<std::map<Id, std::vector<Id>>>();
        std::vector<std::vector<Id>> groups;
        std::vector<double> targets;
        for (py::ssize_t i = 0; i < s.n; ++i) {
            auto id = s.r(i);
            if (id < 0 && !members.contains(id))
                throw py::key_error("Missing macro members");
            auto group = id >= 0 ? std::vector<Id>{id} : members.at(id);
            double sum = 0;
            for (auto member : group) {
                if (!rows.contains(member) || !cols.contains(member) || !beliefs.contains(member))
                    throw py::key_error("Unknown macro member");
                sum += beliefs.at(member);
            }
            targets.push_back(group.empty() ? nan : sum / group.size());
            groups.push_back(std::move(group));
        }
        auto matrix = zeros(s.n);
        auto m = matrix.mutable_unchecked<2>();
        {
            py::gil_scoped_release release;
            for (py::ssize_t i = 0; i < s.n; ++i)
                for (py::ssize_t j = 0; j < s.n; ++j)
                    for (auto row : groups[i])
                        for (auto col : groups[j])
                            m(i, j) += v.m(rows.at(row), cols.at(col));
        }
        Qubo base(matrix, sub.rows_idx.attr("copy")().cast<Indices>(), sub.cols_idx.attr("copy")().cast<Indices>(),
                  q.offset);
        auto out = base.mat.mutable_unchecked<2>();
        {
            py::gil_scoped_release release;
            std::vector<double> penalties(s.n);
            for (py::ssize_t i = 0; i < s.n; ++i) {
                double row = 0, col = 0;
                for (py::ssize_t j = 0; j < s.n; ++j) {
                    row += std::abs(out(i, j));
                    col += std::abs(out(j, i));
                }
                penalties[i] = strength * (row + col - std::abs(out(i, i)));
            }
            for (py::ssize_t i = 0; i < s.n; ++i) {
                out(i, i) += penalties[i] * (1 - 2 * targets[i]);
                base.offset += penalties[i] * targets[i] * targets[i];
            }
        }
        auto bound = py::cast(std::move(base));
        bound.attr("macro_members") = py::cast(members);
        result.append(bound);
    }
    return result;
}
} // namespace

py::list refine_quadtree(py::list subs, const Qubo &q, double strength) {
    return soft_consensus(subs, q, strength, collect_beliefs(subs, q));
}

py::object accept_conditioned(const Qubo &q, const Qubo &sub, py::object df) {
    View v(q), local(sub);
    Table global(q.solutions), samples(df);
    auto best = global.best();
    auto current = global.assignment(best);
    auto a = global.values.unchecked<2>(), s = samples.values.unchecked<2>();
    double current_energy = energy(v, current) + q.offset;
    for (py::ssize_t i = 0; i < local.n; ++i)
        if (local.r(i) < 0 || local.r(i) != local.c(i) || !current.contains(local.r(i)))
            throw py::value_error("Conditioned updates require principal real-variable blocks");
    for (py::ssize_t row = 0; row < s.shape(0); ++row) {
        auto candidate = current;
        for (py::ssize_t i = 0; i < local.n; ++i) {
            auto id = local.r(i);
            if (!samples.by_id.contains(id))
                throw py::value_error("Conditioned refinement requires binary samples for every local variable");
            auto value = s(row, samples.by_id.at(id));
            if (value != 0 && value != 1)
                throw py::value_error("Conditioned refinement requires binary samples for every local variable");
            candidate[id] = value;
        }
        double e;
        {
            py::gil_scoped_release release;
            e = energy(v, candidate) + q.offset;
        }
        if (e <= current_energy) {
            current = std::move(candidate);
            current_energy = e;
        }
    }
    Array result({py::ssize_t(1), a.shape(1)});
    auto out = result.mutable_unchecked<2>();
    for (py::ssize_t col = 0; col < a.shape(1); ++col)
        out(0, col) = a(best, col);
    for (auto [id, col] : global.by_id)
        out(0, col) = current.at(id);
    out(0, global.energy_col) = current_energy;
    return dataframe(result, global.columns);
}

py::object refine_conditioned(const Qubo &q, py::function solve, int block, py::object rng) {
    if (block <= 0)
        throw py::value_error("Refinement block_size must be positive");
    View v(q);
    auto working = q;
    Table global(q.solutions);
    auto best = global.best();
    py::list selected;
    selected.append(best);
    working.solutions = global.frame.attr("iloc")[selected].attr("copy")();
    std::vector<Id> ids;
    for (py::ssize_t i = 0; i < v.n; ++i)
        if (v.r(i) >= 0)
            ids.push_back(v.r(i));
    auto order = rng.attr("permutation")(ids).attr("tolist")().cast<std::vector<Id>>();
    for (std::size_t start = 0; start < order.size(); start += block) {
        std::vector<Id> window(order.begin() + start, order.begin() + std::min(order.size(), start + block));
        Table incumbent(working.solutions);
        auto sub = condition(q, window, incumbent.assignment(incumbent.best()));
        if (!empty(sub))
            working.solutions = accept_conditioned(working, sub, solve(py::cast(sub)));
    }
    return working.solutions.attr("reset_index")("drop"_a = true);
}
} // namespace qs
