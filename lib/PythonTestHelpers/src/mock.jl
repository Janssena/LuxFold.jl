"""
    mock_imports(blocked...)

Install an import blocker for GPU-only modules, so the reference models import on a CPU box.

`openfold`/`openfold3` need `deepspeed`, `flash_attn` and `attn_core_inplace_cuda`; `boltz` needs
`cuequivariance_torch`. None are installed here and none are reached by the code under test, so
they are replaced by mock packages rather than allowed to fail at import.

The machinery is `mock_imports.py` next to this file — one definition, where the three test suites
each used to carry a verbatim copy inline.

    mock_imports("deepspeed", "flash_attn", "attn_core_inplace_cuda")
"""
function mock_imports(blocked::AbstractString...)
    sys = pyimport("sys")
    dir = @__DIR__
    dir in pyconvert(Vector{String}, sys.path) || sys.path.insert(0, dir)
    pyimport("mock_imports").install(pylist(collect(blocked)))
    return nothing
end
