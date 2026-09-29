#include "core.hpp"

using namespace qs;

PYBIND11_MODULE(_core, m) {
    m.doc() = "QSplit C++23 computational core";
    auto cls =
        py::class_<Qubo>(m, "QUBO", py::dynamic_attr())
            .def(py::init<Array, Indices, Indices, double>(), "mat"_a, "rows_idx"_a, "cols_idx"_a, "offset"_a = 0.0)
            .def_readwrite("mat", &Qubo::mat)
            .def_readwrite("rows_idx", &Qubo::rows_idx)
            .def_readwrite("cols_idx", &Qubo::cols_idx)
            .def_readwrite("offset", &Qubo::offset)
            .def_readwrite("problem_size", &Qubo::problem_size)
            .def_readwrite("solutions", &Qubo::solutions)
            .def_static(
                "sanitize", [](Array a, Indices cols, Indices rows) { return py::make_tuple(sanitize(a), cols, rows); },
                "mat"_a, "cols_idx"_a, "rows_idx"_a)
            .def("__str__",
                 [](const Qubo &q) {
                     return py::str("QUBO(cols: {}, rows: {}, offset: {}, size: {})")
                         .attr("format")(q.cols_idx, q.rows_idx, q.offset, q.problem_size);
                 })
            .def(py::pickle(
                [](py::object self) {
                    auto &q = self.cast<Qubo &>();
                    py::dict state(self.attr("__dict__").attr("copy")());
                    state["mat"] = q.mat;
                    state["rows_idx"] = q.rows_idx;
                    state["cols_idx"] = q.cols_idx;
                    state["offset"] = q.offset;
                    state["problem_size"] = q.problem_size;
                    state["solutions"] = q.solutions;
                    return state;
                },
                [](py::dict state) {
                    // The dictionary schema and module path also read pre-port Python pickles.
                    auto mat = state["mat"].cast<Array>();
                    Qubo q(mat, state["rows_idx"].cast<Indices>(), state["cols_idx"].cast<Indices>(),
                           state["offset"].cast<double>());
                    q.mat = mat; // Mutable matrices must survive pickle without being re-sanitized.
                    q.problem_size = state["problem_size"].cast<py::ssize_t>();
                    q.solutions = state["solutions"];
                    auto attrs = py::dict(state.attr("copy")());
                    for (const auto *key : {"mat", "rows_idx", "cols_idx", "offset", "problem_size", "solutions"})
                        attrs.attr("pop")(key);
                    return std::make_pair(std::move(q), attrs);
                }));
    cls.attr("__module__") = "qsplit.qubo";
    m.def("is_empty", &empty, "qubo"_a);
    m.def("vars_count", &vars_count, "qubo"_a);
    m.def("split_recursive", &split_recursive, "qubo"_a);
    m.def("split_linear", &split_linear, "qubo"_a, "cut"_a, "stride"_a);
    m.def("split_interactions", &split_interactions, "qubo"_a, "cut"_a);
    m.def("split_quadtree", &split_quadtree, "qubo"_a, "cut"_a, "ratio"_a);
    m.def("split_graph", &split_graph, "qubo"_a, "cut"_a, "partition"_a);
    m.def("split_leaves", &split_leaves, "qubo"_a, "cut"_a);
    m.def("validate_solutions", &validate_solutions, "df"_a, "qubo"_a);
    m.def("aggregate_votes", &aggregate_votes, "solutions"_a, "qubo"_a, "mode"_a);
    m.def("fill_inf", &fill_inf);
    m.def("combine_ul_lr", &combine_ul_lr);
    m.def("combine_rows", &combine_rows);
    m.def("distance", &distance);
    m.def("closest", &closest);
    m.def("resolve_conflicts", &resolve_conflicts, "df"_a, "qubo"_a, "solve"_a, "exact_limit"_a);
    m.def("aggregate_recursive", &aggregate_recursive);
    m.def("aggregate_trivial", &aggregate_trivial, "ul"_a, "lr"_a, "qubo"_a);
    m.def("aggregate_leaves", &aggregate_leaves);
    m.def("logical_expansion", &logical_expansion, "subs"_a);
    m.def("calculate_energy", &calculate_energy, "x"_a, "q_mat"_a);
    m.def("collect_beliefs", &collect_beliefs, "subproblems"_a, "qubo"_a);
    m.def("condition_subproblem", &condition, "qubo"_a, "window"_a, "beliefs"_a);
    m.def("refine_linear", &refine_linear, "subproblems"_a, "qubo"_a);
    m.def("refine_quadtree", &refine_quadtree, "subproblems"_a, "qubo"_a, "strength"_a);
    m.def("refine_conditioned", &refine_conditioned, "qubo"_a, "solve"_a, "block_size"_a, "rng"_a);
}
