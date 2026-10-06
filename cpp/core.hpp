#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace py = pybind11;
namespace qs {
using namespace pybind11::literals;
using Id = std::int64_t;
using Array = py::array_t<double, py::array::forcecast>;
using Indices = py::array_t<Id, py::array::forcecast>;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct Qubo {
    Array mat;
    Indices rows_idx, cols_idx;
    double offset;
    py::ssize_t problem_size;
    py::object solutions = py::none();
    Qubo(Array mat, Indices rows, Indices cols, double offset = 0);
};

// Own the array references while unchecked, strided views are in use. The public
// arrays remain writable, so validate dimensions at each numerical entry point.
struct View {
    Array matrix;
    Indices rows, cols;
    py::detail::unchecked_reference<double, 2> m;
    py::detail::unchecked_reference<Id, 1> r, c;
    py::ssize_t n;
    explicit View(const Qubo &q);
};
Array sanitize(Array mat);
Indices indices(const std::vector<Id> &values);
Array zeros(py::ssize_t n);
std::vector<Id> variables(const Qubo &q, bool rows_only = false);
std::map<Id, py::ssize_t> positions(Indices ids);
double energy(const View &v, const std::map<Id, double> &assignment);
bool empty(const Qubo &q);
std::size_t vars_count(const Qubo &q);
Qubo subqubo(const Qubo &q, const std::vector<py::ssize_t> &rows, const std::vector<py::ssize_t> &cols, double offset);

// Pandas conversion happens once per table, outside the computational loops.
struct Table {
    py::object frame;
    py::list columns;
    Array values;
    std::map<Id, py::ssize_t> by_id;
    py::ssize_t energy_col = -1;
    explicit Table(py::object df, bool numeric_strings = false);
    py::ssize_t best(bool finite_only = false) const;
    std::map<Id, double> assignment(py::ssize_t row) const;
};
py::object dataframe(Array values, py::object columns);
py::object assignment_frame(const Qubo &q, const std::map<Id, double> &assignment);

py::tuple split_recursive(const Qubo &q);
py::list split_linear(Qubo &q, int cut, int stride);
py::list split_interactions(const Qubo &q, int cut);
py::list split_quadtree(const Qubo &q, int cut, double ratio);
py::list split_graph(const Qubo &q, int cut, py::function partition);
py::list split_leaves(py::object q, int cut);

py::object validate_solutions(py::object df, const Qubo &q);
py::object aggregate_votes(py::list subs, py::object q, const std::string &mode);
py::object fill_inf(py::object schema, py::object df);
py::object combine_ul_lr(const Qubo &ul, const Qubo &lr);
std::vector<double> combine_rows(py::object a, py::object b);
double distance(Array a, Array b);
py::object closest(py::object start, py::object candidates);
py::object resolve_conflicts(py::object df, const Qubo &q, py::function solve, int exact_limit);
py::object aggregate_recursive(py::tuple subs, py::object q, py::function resolve);
py::object aggregate_trivial(const Qubo &ul, const Qubo &lr, py::object q);
py::object aggregate_leaves(py::list subs, py::object q, py::function resolve);
py::tuple logical_expansion(py::tuple subs);
double calculate_energy(Array x, Array mat);

std::map<Id, double> collect_beliefs(py::list subs, const Qubo &q);
Qubo condition(const Qubo &q, const std::vector<Id> &window, const std::map<Id, double> &beliefs);
py::list refine_linear(py::list subs, const Qubo &q);
py::list refine_mean_field(py::list subs, const Qubo &q, int block);
py::list refine_soft_consensus(py::list subs, const Qubo &q, double strength, int block);
py::list refine_quadtree(py::list subs, const Qubo &q, double strength);
py::object refine_conditioned(const Qubo &q, py::function solve, int block, py::object rng);
} // namespace qs
