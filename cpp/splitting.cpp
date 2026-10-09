#include "core.hpp"

namespace qs {
namespace {
void positive_cut(int cut) {
    if (cut <= 0)
        throw py::value_error("CUT_DIM must be positive");
}
std::vector<py::ssize_t> range(py::ssize_t start, py::ssize_t end) {
    std::vector<py::ssize_t> values(end - start);
    std::iota(values.begin(), values.end(), start);
    return values;
}
Qubo padded(const Qubo &q, py::ssize_t n) {
    View v(q);
    auto mat = zeros(n);
    auto out = mat.mutable_unchecked<2>();
    std::vector<Id> rows(n), cols(n);
    for (py::ssize_t i = 0; i < n; ++i) {
        rows[i] = i < v.n ? v.r(i) : -(i - v.n + 1);
        cols[i] = i < v.n ? v.c(i) : -(i - v.n + 1);
    }
    for (py::ssize_t i = 0; i < v.n; ++i)
        for (py::ssize_t j = 0; j < v.n; ++j)
            out(i, j) = v.m(i, j);
    // Do not sanitize here: recursive splitting must detect a modified lower-left block.
    Qubo result(zeros(n), indices(rows), indices(cols), q.offset);
    result.mat = mat;
    return result;
}
} // namespace

py::tuple split_recursive(const Qubo &q) {
    View original(q);
    auto p = padded(q, original.n + original.n % 2);
    View v(p);
    auto half = v.n / 2;
    for (py::ssize_t i = half; i < v.n; ++i)
        for (py::ssize_t j = 0; j < half; ++j)
            if (v.m(i, j) != 0)
                throw py::value_error("Lower left sub-matrix must be 0");
    auto first = range(0, half), last = range(half, v.n);
    return py::make_tuple(subqubo(p, first, first, 0), subqubo(p, first, last, 0), subqubo(p, last, last, 0));
}

py::list split_linear(Qubo &q, int cut, int stride) {
    positive_cut(cut);
    if (stride <= 0)
        stride = cut;
    View initial(q);
    auto n = initial.n;
    auto size = n <= cut ? cut : ((n - cut + stride - 1) / stride) * stride + cut;
    if (size != n) {
        auto p = padded(q, size);
        q.mat = p.mat;
        q.rows_idx = p.rows_idx;
        q.cols_idx = p.cols_idx;
        q.problem_size = size;
    }
    View v(q);
    py::list result;
    for (py::ssize_t i = 0; i <= size - cut; i += stride) {
        for (py::ssize_t j = i; j <= size - cut; j += stride) {
            bool nonzero = false;
            for (py::ssize_t r = i; r < i + cut && !nonzero; ++r)
                for (py::ssize_t c = j; c < j + cut; ++c)
                    if (v.m(r, c) != 0) {
                        nonzero = true;
                        break;
                    }
            if (nonzero)
                result.append(subqubo(q, range(i, i + cut), range(j, j + cut), 0));
        }
    }
    return result;
}

py::list split_interactions(const Qubo &q, int cut) {
    positive_cut(cut);
    View v(q);
    cut = std::min<py::ssize_t>(cut, v.n);
    py::list result;
    for (py::ssize_t i = 0; i < v.n; ++i) {
        auto order = range(0, v.n);
        {
            py::gil_scoped_release release;
            auto weight = [&](auto j) { return i == j ? 0.0 : std::abs(v.m(i, j)) + std::abs(v.m(j, i)); };
            std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                auto wa = weight(a), wb = weight(b);
                return !std::isnan(wa) && (std::isnan(wb) || wa < wb);
            });
            order.erase(order.begin(), order.end() - (cut - 1));
            order.push_back(i);
            std::sort(order.begin(), order.end());
            order.erase(std::unique(order.begin(), order.end()), order.end());
        }
        result.append(subqubo(q, order, order, 0));
    }
    return result;
}

py::list split_quadtree(const Qubo &q, int cut, double ratio) {
    cut -= cut % 2;
    if (cut < 2)
        throw py::value_error("Quadtree CUT_DIM must be at least 2");
    if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
        throw py::value_error("EXACT_RATIO must be finite and between 0 and 1");
    View v(q);
    std::vector<py::ssize_t> real;
    for (py::ssize_t i = 0; i < v.n; ++i)
        if (v.r(i) >= 0)
            real.push_back(i);
    py::list result;
    if (real.size() <= static_cast<std::size_t>(cut)) {
        auto sub = subqubo(q, real, real, q.offset);
        sub.cols_idx = indices([&] {
            std::vector<Id> ids;
            for (auto i : real)
                ids.push_back(v.r(i));
            return ids;
        }());
        result.append(std::move(sub));
        return result;
    }
    auto exact = std::max(1, static_cast<int>(cut * ratio));
    auto macros = cut - exact;
    auto sym = [&](auto i, auto j) {
        auto r = real[i], c = real[j];
        return v.m(r, c) + v.m(c, r) - (r == c ? v.m(r, r) : 0);
    };
    for (py::ssize_t center = 0; center < static_cast<py::ssize_t>(real.size()); ++center) {
        std::vector<std::vector<py::ssize_t>> groups;
        auto order = range(0, real.size());
        order.erase(order.begin() + center);
        {
            py::gil_scoped_release release;
            std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                auto wa = std::abs(sym(center, a)), wb = std::abs(sym(center, b));
                return !std::isnan(wa) && (std::isnan(wb) || wa > wb);
            });
            groups.push_back({center});
            for (int i = 0; i < exact - 1; ++i)
                groups.push_back({order[i]});
            auto remaining = order.size() - (exact - 1);
            std::size_t start = exact - 1;
            for (int m = 0; m < macros; ++m) {
                auto count = remaining / macros + (static_cast<std::size_t>(m) < remaining % macros);
                groups.emplace_back(order.begin() + start, order.begin() + start + count);
                start += count;
            }
        }
        auto mat = zeros(cut);
        auto out = mat.mutable_unchecked<2>();
        std::vector<Id> ids;
        py::dict members;
        for (int i = 0; i < cut; ++i) {
            ids.push_back(i < exact ? v.r(real[groups[i][0]]) : -(i - exact + 1000));
            if (i >= exact) {
                std::vector<Id> values;
                for (auto j : groups[i])
                    values.push_back(v.r(real[j]));
                members[py::int_(ids.back())] = py::cast(values);
            }
        }
        {
            py::gil_scoped_release release;
            for (int i = 0; i < cut; ++i) {
                for (int j = i; j < cut; ++j) {
                    for (auto r : groups[i])
                        for (auto c : groups[j]) {
                            if (i == j)
                                out(i, j) += v.m(real[r], real[c]);
                            else
                                out(i, j) += sym(r, c);
                        }
                }
            }
        }
        auto sub = py::cast(Qubo(mat, indices(ids), indices(ids), q.offset));
        sub.attr("macro_members") = members;
        result.append(sub);
    }
    return result;
}

py::list split_graph(const Qubo &q, int cut, py::function partition) {
    positive_cut(cut);
    View v(q);
    std::vector<std::vector<py::ssize_t>> adjacency(v.n);
    std::vector<py::ssize_t> nodes;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < v.n; ++i) {
            if (v.c(i) == -1)
                continue;
            nodes.push_back(i);
            for (py::ssize_t j = i + 1; j < v.n; ++j)
                if (v.c(j) != -1 && v.m(i, j) != 0) {
                    adjacency[i].push_back(j);
                    adjacency[j].push_back(i);
                }
        }
    }
    py::list result;
    std::function<void(const std::vector<py::ssize_t> &)> visit = [&](const auto &current) {
        if (current.size() <= static_cast<std::size_t>(cut)) {
            result.append(subqubo(q, current, current, q.offset));
            return;
        }
        std::map<py::ssize_t, py::ssize_t> local;
        for (std::size_t i = 0; i < current.size(); ++i)
            local[current[i]] = i;
        std::vector<std::vector<py::ssize_t>> edges(current.size());
        bool has_edges = false;
        for (std::size_t i = 0; i < current.size(); ++i)
            for (auto neighbor : adjacency[current[i]])
                if (local.count(neighbor)) {
                    edges[i].push_back(local.at(neighbor));
                    has_edges = true;
                }
        std::vector<py::ssize_t> first, second;
        if (has_edges) {
            auto parts = partition("nparts"_a = 2, "adjacency"_a = edges);
            auto member = parts[py::int_(1)].template cast<std::vector<int>>();
            if (member.size() != current.size())
                throw py::value_error("Invalid graph partition size");
            for (std::size_t i = 0; i < current.size(); ++i)
                (member[i] == 0 ? first : second).push_back(current[i]);
        }
        // A degenerate METIS result must still make progress.
        if (first.empty() || second.empty()) {
            auto half = current.begin() + current.size() / 2;
            first.assign(current.begin(), half);
            second.assign(half, current.end());
        }
        visit(first);
        visit(second);
    };
    visit(nodes);
    return result;
}

py::list split_leaves(py::object obj, int cut) {
    positive_cut(cut);
    py::list leaves;
    std::function<py::object(py::object)> visit = [&](py::object node) -> py::object {
        auto &q = node.cast<Qubo &>();
        if (q.problem_size <= cut || empty(q) || vars_count(q) <= static_cast<std::size_t>(cut)) {
            leaves.append(node);
            return py::int_(py::len(leaves) - 1);
        }
        auto subs = split_recursive(q);
        py::tuple children(3);
        for (int i = 0; i < 3; ++i)
            children[i] = visit(py::reinterpret_borrow<py::object>(subs[i]));
        return py::make_tuple(node, children);
    };
    obj.attr("split_tree") = visit(py::module_::import("copy").attr("deepcopy")(obj));
    return leaves;
}
} // namespace qs
