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
    std::vector<Id> rows(n), cols(n);
    for (py::ssize_t i = 0; i < n; ++i) {
        rows[i] = i < v.n ? v.r(i) : -(i - v.n + 1);
        cols[i] = i < v.n ? v.c(i) : -(i - v.n + 1);
    }
    if (v.m.sparse) {
        auto mat = v.m.matrix.attr("copy")();
        mat.attr("resize")(py::make_tuple(n, n));
        Qubo result(mat, indices(rows), indices(cols), q.offset);
        result.mat = mat; // Preserve mutable lower-left entries, just as in the dense path.
        return result;
    }
    auto mat = zeros(n);
    auto out = mat.mutable_unchecked<2>();
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
    bool lower_left = false;
    {
        py::gil_scoped_release release;
        v.m.each([&](auto i, auto j, auto) { lower_left = lower_left || (i >= half && j < half); });
    }
    if (lower_left)
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
    if (v.m.sparse) {
        std::set<std::pair<py::ssize_t, py::ssize_t>> blocks;
        {
            py::gil_scoped_release release;
            v.m.each([&](auto r, auto c, auto) {
                auto first_r = std::max<py::ssize_t>(0, (r - cut + stride) / stride);
                auto first_c = std::max<py::ssize_t>(0, (c - cut + stride) / stride);
                for (auto i = first_r; i <= r / stride && i * stride <= size - cut; ++i)
                    for (auto j = std::max(i, first_c); j <= c / stride && j * stride <= size - cut; ++j)
                        blocks.emplace(i * stride, j * stride);
            });
        }
        py::list result;
        for (auto [i, j] : blocks)
            result.append(subqubo(q, range(i, i + cut), range(j, j + cut), 0));
        return result;
    }
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
    const auto n = static_cast<py::ssize_t>(real.size());
    // Contract several centers together so NumPy can use its existing BLAS backend.
    // S = Q + Q.T; macro diagonals must be halved after contraction.
    auto symmetric = macros && !v.m.sparse ? zeros(n) : zeros(0);
    py::object sparse_symmetric;
    auto symmetric_view = symmetric.mutable_unchecked<2>();
    // Direct sums preserve nonfinite behavior; BLAS would spread NaNs through 0 * inf.
    bool finite = macros > 0;
    {
        py::gil_scoped_release release;
        if (macros && !v.m.sparse)
            for (py::ssize_t i = 0; i < n; ++i)
                for (py::ssize_t j = i; j < n; ++j) {
                    auto value = v.m(real[i], real[j]) + v.m(real[j], real[i]);
                    symmetric_view(i, j) = symmetric_view(j, i) = value;
                    finite = finite && std::isfinite(value);
                }
    }
    if (macros && v.m.sparse) {
        auto ids = indices(std::vector<Id>(real.begin(), real.end()));
        auto ix = py::module_::import("numpy").attr("ix_")(ids, ids);
        auto principal = v.m.matrix.attr("__getitem__")(ix);
        sparse_symmetric = principal.attr("__add__")(principal.attr("T"));
        finite = py::module_::import("numpy").attr("isfinite")(sparse_symmetric.attr("data")).attr("all")().cast<bool>();
    }
    // Keep temporary membership/contraction arrays to about 128 columns.
    const auto batch_size = finite ? std::max(1, 128 / macros) : 1;
    using Groups = std::vector<std::vector<py::ssize_t>>;
    // Macro lists can share immutable Python IDs instead of allocating N copies of each ID.
    std::vector<py::int_> variable_ids;
    variable_ids.reserve(n);
    for (auto r : real)
        variable_ids.emplace_back(v.r(r));
    for (py::ssize_t start_center = 0; start_center < n; start_center += batch_size) {
        auto count = std::min<py::ssize_t>(batch_size, n - start_center);
        std::vector<Groups> batch_groups;
        {
            py::gil_scoped_release release;
            for (py::ssize_t center = start_center; center < start_center + count; ++center) {
                Groups groups;
                auto order = range(0, n);
                order.erase(order.begin() + center);
                std::vector<double> weights(n);
                for (auto i : order)
                    weights[i] = std::abs(finite && !v.m.sparse ? symmetric_view(center, i) : sym(center, i));
                std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                    auto wa = weights[a], wb = weights[b];
                    return !std::isnan(wa) && (std::isnan(wb) || wa > wb);
                });
                groups.push_back({center});
                for (int i = 0; i < exact - 1; ++i)
                    groups.push_back({order[i]});
                auto remaining = order.size() - (exact - 1);
                std::size_t start = exact - 1;
                for (int m = 0; m < macros; ++m) {
                    auto size = remaining / macros + (static_cast<std::size_t>(m) < remaining % macros);
                    groups.emplace_back(order.begin() + start, order.begin() + start + size);
                    start += size;
                }
                batch_groups.push_back(std::move(groups));
            }
        }
        auto contracted = zeros(0);
        if (finite) {
            Array membership({n, count * macros});
            std::fill_n(membership.mutable_data(), membership.size(), 0.0);
            auto membership_view = membership.mutable_unchecked<2>();
            {
                py::gil_scoped_release release;
                for (py::ssize_t b = 0; b < count; ++b)
                    for (int m = 0; m < macros; ++m)
                        for (auto r : batch_groups[b][exact + m])
                            membership_view(r, b * macros + m) = 1.0;
            }
            contracted = (v.m.sparse ? sparse_symmetric.attr("__matmul__")(membership)
                                     : symmetric.attr("__matmul__")(membership)).cast<Array>();
        }
        auto contraction = contracted.unchecked<2>();
        for (py::ssize_t b = 0; b < count; ++b) {
            const auto &groups = batch_groups[b];
            auto mat = zeros(cut);
            auto out = mat.mutable_unchecked<2>();
            std::vector<Id> ids;
            py::dict members;
            for (int i = 0; i < cut; ++i) {
                ids.push_back(i < exact ? v.r(real[groups[i][0]]) : -(i - exact + 1000));
                if (i >= exact) {
                    py::list values(groups[i].size());
                    for (std::size_t k = 0; k < groups[i].size(); ++k)
                        values[k] = variable_ids[groups[i][k]];
                    members[py::int_(ids.back())] = values;
                }
            }
            {
                py::gil_scoped_release release;
                for (int i = 0; i < cut; ++i)
                    for (int j = i; j < cut; ++j) {
                        if (finite && j >= exact) {
                            double sum = 0;
                            for (auto r : groups[i])
                                sum += contraction(r, b * macros + j - exact);
                            out(i, j) = i == j ? sum / 2 : sum;
                        } else {
                            for (auto r : groups[i])
                                for (auto c : groups[j])
                                    out(i, j) += i == j ? v.m(real[r], real[c]) : sym(r, c);
                        }
                    }
            }
            auto sub = py::cast(Qubo(mat, indices(ids), indices(ids), q.offset));
            sub.attr("macro_members") = members;
            result.append(sub);
        }
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
        for (py::ssize_t i = 0; i < v.n; ++i)
            if (v.c(i) != -1)
                nodes.push_back(i);
        v.m.each([&](auto i, auto j, auto) {
            if (j > i && v.c(i) != -1 && v.c(j) != -1) {
                adjacency[i].push_back(j);
                adjacency[j].push_back(i);
            }
        });
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
