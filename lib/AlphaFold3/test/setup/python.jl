openfold3_repo = abspath(joinpath(@__FILE__, "..", "..", "..", "..", "..", "python", "openfold-3"))
pythonpath = joinpath(openfold3_repo, ".pixi", "envs", "openfold3-cpu", "bin", "python")

import PythonTestHelpers: setup

setup(pythonpath)

using PyCall, PythonTestHelpers

# Add the openfold-3 repo root to sys.path so `import openfold3` works.
pushfirst!(pyimport("sys")."path", openfold3_repo)


include("mock.jl")
