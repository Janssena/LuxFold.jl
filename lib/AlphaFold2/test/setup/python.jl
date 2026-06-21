openfold_repo    = abspath(joinpath(@__FILE__, "..", "..", "..", "..", "..", "python", "openfold"))
minalphafold_repo = abspath(joinpath(@__FILE__, "..", "..", "..", "..", "..", "python", "minAlphaFold2"))
pythonpath = joinpath(openfold_repo, ".venv", "bin", "python")

import PythonTestHelpers: setup

setup(pythonpath)

using PyCall, PythonTestHelpers

# Add openfold and minAlphaFold2 repos to path:
pushfirst!(pyimport("sys")."path", openfold_repo)
pushfirst!(pyimport("sys")."path", minalphafold_repo)

include("mock.jl")

const openfold = pyimport("openfold")
