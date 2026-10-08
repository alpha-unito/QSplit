import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

if sys.platform == "win32":
    raise RuntimeError("QSplit supports Linux and macOS; Windows is not supported")

setup(
    ext_modules=[
        Pybind11Extension(
            "qsplit._core",
            ["cpp/bindings.cpp", "cpp/qubo.cpp", "cpp/splitting.cpp", "cpp/aggregation.cpp", "cpp/refinement.cpp"],
            depends=["cpp/core.hpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-ffp-contract=off"],
        )
    ],
)
