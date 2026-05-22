openfold_repo = abspath(joinpath(@__FILE__, "..", "..", "..", "..", "..", "python", "openfold"))
pythonpath = joinpath(openfold_repo, ".venv", "bin", "python")

import PythonTestHelpers: setup

setup(pythonpath)

using PyCall, PythonTestHelpers

# Add openfold repo to path:
pushfirst!(pyimport("sys")."path", openfold_repo)

include("mock.jl")

const openfold = pyimport("openfold")
