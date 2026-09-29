#include "core.hpp"

namespace qs {
namespace {
void require(bool ok, const char *message) {
    if (!ok) {
        PyErr_SetString(PyExc_AssertionError, message);
        throw py::error_already_set();
    }
}
void validate(Array mat, Indices rows, Indices cols) {
    require(mat.ndim() == 2 && mat.shape(0) == mat.shape(1), "Problem matrix must be square");
    require(cols.ndim() == 1 && mat.shape(1) == cols.shape(0),
            "Invalid numer of columns, must be as long as cols_idx array");
    require(rows.ndim() == 1 && mat.shape(0) == rows.shape(0),
            "Invalid numer of rows, must be as long as rows_idx array");
}
} // namespace

Array sanitize(Array mat) {
    require(mat.ndim() == 2 && mat.shape(0) == mat.shape(1), "Problem matrix must be square");
    auto a = mat.unchecked<2>();
    const auto n = a.shape(0);
    Array result({n, n});
    auto out = result.mutable_unchecked<2>();
    {
        py::gil_scoped_release release;
        bool triangular = true;
        for (py::ssize_t i = 0; i < n; ++i)
            for (py::ssize_t j = 0; j < n; ++j)
                if (std::isnan(a(i, j)) || (i > j && std::abs(a(i, j)) > 1e-8))
                    triangular = false;
        for (py::ssize_t i = 0; i < n; ++i) {
            for (py::ssize_t j = 0; j < n; ++j) {
                double value = a(i, j);
                if (!triangular)
                    value = i < j ? a(i, j) + a(j, i) : (i == j ? (a(i, i) + a(i, i)) - a(i, i) : 0.0);
                // NumPy's decimal rounding uses ties-to-even, not std::round.
                value = std::nearbyint(value * 1e9) / 1e9;
                out(i, j) = std::abs(value) < 1e-12 ? 0.0 : value;
            }
        }
    }
    return result;
}

Qubo::Qubo(Array a, Indices rows, Indices cols, double off)
    : mat(std::move(a)), rows_idx(std::move(rows)), cols_idx(std::move(cols)), offset(off), problem_size(0) {
    validate(mat, rows_idx, cols_idx);
    mat = sanitize(mat);
    problem_size = mat.shape(0);
}

View::View(const Qubo &q)
    : matrix(q.mat), rows(q.rows_idx), cols(q.cols_idx), m((validate(matrix, rows, cols), matrix.unchecked<2>())),
      r(rows.unchecked<1>()), c(cols.unchecked<1>()), n(matrix.shape(0)) {
    if (q.problem_size != n)
        throw py::value_error("problem_size must match the QUBO matrix");
}

Indices indices(const std::vector<Id> &values) {
    Indices result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}
Array zeros(py::ssize_t n) {
    Array result({n, n});
    std::fill_n(result.mutable_data(), result.size(), 0.0);
    return result;
}
std::map<Id, py::ssize_t> positions(Indices ids) {
    auto a = ids.unchecked<1>();
    std::map<Id, py::ssize_t> result;
    for (py::ssize_t i = 0; i < a.shape(0); ++i)
        result[a(i)] = i;
    return result;
}
std::vector<Id> variables(const Qubo &q, bool rows_only) {
    std::set<Id> found;
    for (auto ids : {q.rows_idx, q.cols_idx}) {
        auto a = ids.unchecked<1>();
        for (py::ssize_t i = 0; i < a.shape(0); ++i)
            if (a(i) >= 0)
                found.insert(a(i));
        if (rows_only)
            break;
    }
    return {found.begin(), found.end()};
}

double energy(const View &v, const std::map<Id, double> &assignment) {
    std::vector<double> rows(v.n), cols(v.n);
    for (py::ssize_t i = 0; i < v.n; ++i) {
        if (v.r(i) >= 0 && assignment.contains(v.r(i)))
            rows[i] = assignment.at(v.r(i));
        if (v.c(i) >= 0 && assignment.contains(v.c(i)))
            cols[i] = assignment.at(v.c(i));
    }
    // Accumulate in row-major order for cache locality, preserving each column's
    // summation order from rows @ matrix @ cols.
    std::vector<double> contracted(v.n, 0);
    for (py::ssize_t i = 0; i < v.n; ++i)
        for (py::ssize_t j = 0; j < v.n; ++j)
            contracted[j] += rows[i] * v.m(i, j);
    double result = 0;
    for (py::ssize_t j = 0; j < v.n; ++j)
        result += contracted[j] * cols[j];
    return result;
}
bool empty(const Qubo &q) {
    View v(q);
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < v.n; ++i)
        if (v.r(i) >= 0)
            for (py::ssize_t j = 0; j < v.n; ++j)
                if (v.c(j) >= 0 && v.m(i, j) != 0)
                    return false;
    return true;
}
std::size_t vars_count(const Qubo &q) {
    View v(q);
    py::gil_scoped_release release;
    std::vector<bool> active_rows(v.n), active_cols(v.n);
    for (py::ssize_t i = 0; i < v.n; ++i)
        if (v.r(i) >= 0)
            for (py::ssize_t j = 0; j < v.n; ++j)
                if (v.c(j) >= 0 && v.m(i, j) != 0) {
                    active_rows[i] = true;
                    active_cols[j] = true;
                }
    std::set<Id> found;
    for (py::ssize_t i = 0; i < v.n; ++i) {
        if (active_rows[i])
            found.insert(v.r(i));
        if (active_cols[i])
            found.insert(v.c(i));
    }
    return found.size();
}
Qubo subqubo(const Qubo &q, const std::vector<py::ssize_t> &rows, const std::vector<py::ssize_t> &cols, double offset) {
    View v(q);
    if (rows.size() != cols.size())
        throw py::value_error("Subproblem must be square");
    auto mat = zeros(rows.size());
    auto out = mat.mutable_unchecked<2>();
    std::vector<Id> r, c;
    for (auto i : rows) {
        if (i < 0 || i >= v.n)
            throw py::index_error("Invalid row position");
        r.push_back(v.r(i));
    }
    for (auto j : cols) {
        if (j < 0 || j >= v.n)
            throw py::index_error("Invalid column position");
        c.push_back(v.c(j));
    }
    {
        py::gil_scoped_release release;
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t j = 0; j < cols.size(); ++j)
                out(i, j) = v.m(rows[i], cols[j]);
    }
    return Qubo(mat, indices(r), indices(c), offset);
}

Table::Table(py::object df, bool numeric_strings)
    : frame(std::move(df)), columns(frame.attr("columns")), values(frame.attr("to_numpy")("dtype"_a = "float64")) {
    for (py::ssize_t i = 0; i < py::len(columns); ++i) {
        auto col = columns[i];
        if (py::isinstance<py::str>(col)) {
            if (py::cast<std::string>(col) == "energy") {
                energy_col = i;
                continue;
            }
            if (!numeric_strings)
                continue;
        }
        auto integer = py::module_::import("builtins").attr("int")(col);
        by_id[integer.cast<Id>()] = i;
    }
}
py::ssize_t Table::best(bool finite_only) const {
    if (energy_col < 0)
        throw py::key_error("energy");
    auto a = values.unchecked<2>();
    py::ssize_t best_row = -1;
    double best_energy = inf;
    for (py::ssize_t i = 0; i < a.shape(0); ++i) {
        const auto e = a(i, energy_col);
        if (finite_only && !std::isfinite(e))
            continue;
        if (best_row < 0 || (!std::isnan(e) && (std::isnan(best_energy) || e < best_energy))) {
            best_row = i;
            best_energy = e;
        }
    }
    return best_row;
}
std::map<Id, double> Table::assignment(py::ssize_t row) const {
    auto a = values.unchecked<2>();
    if (row < 0 || row >= a.shape(0))
        throw py::index_error("No solution rows");
    std::map<Id, double> result;
    for (auto [id, col] : by_id)
        result[id] = a(row, col);
    return result;
}
py::object dataframe(Array values, py::object columns) {
    return py::module_::import("pandas").attr("DataFrame")(values, "columns"_a = columns);
}
py::object assignment_frame(const Qubo &q, const std::map<Id, double> &assignment) {
    py::dict data;
    for (auto [id, value] : assignment)
        data[py::int_(id)] = py::make_tuple(static_cast<Id>(value));
    View v(q);
    double e;
    {
        py::gil_scoped_release release;
        e = energy(v, assignment) + q.offset;
    }
    data["energy"] = py::make_tuple(e);
    return py::module_::import("pandas").attr("DataFrame")(data);
}
} // namespace qs
