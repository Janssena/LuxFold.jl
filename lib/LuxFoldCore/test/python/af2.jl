# The REAL openfold layers that LuxFoldCore's shared layers are ports of.
#
# These were hand-copied `nn.Module` definitions in a `py"""` block. A copy drifts: the copied
# `OuterProductMean` had already lost upstream's `init.final_init_` on `proj_o` and never gained
# its chunked forward. Importing the real class means an upstream fix reaches these tests for free.
#
# Needs PyCall/PythonCall bound to the openfold venv, see `runtests_af2.jl`.
const PyAF2OuterProductMean = pyimport("openfold.model.outer_product_mean").OuterProductMean
