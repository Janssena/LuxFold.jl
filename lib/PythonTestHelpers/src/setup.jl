"""
    setup(pythonpath::String)

Verify that this process's Python is `pythonpath`, and fail loudly if it is not.

# Calling contract

The interpreter is chosen when PythonCall initialises, and cannot be changed afterwards. This
module does `using PythonCall` at its own top level, so **importing this function already
initialises Python** — which means `setup` cannot select the interpreter for you. Selecting it is
two lines of plain `Base` code that must come FIRST, before any import:

```julia
pythonpath = joinpath(repo, ".venv", "bin", "python")
ENV["JULIA_PYTHONCALL_EXE"]    = pythonpath
ENV["JULIA_CONDAPKG_BACKEND"]  = "Null"   # or CondaPkg provisions its own env and ignores the venv

import PythonTestHelpers: setup
setup(pythonpath)                          # verifies the above actually took effect
using PythonCall, PythonTestHelpers
```

`setup` sets both variables too, so a caller that gets the order right anyway loses nothing — but
if PythonCall is already live they have no effect, which is exactly the case this function exists
to catch.

# Why this is an assertion and not a build step

It used to be `ENV["PYTHON"] = …; Pkg.build("PyCall")`, which could not work in a single pass:

  * PyCall resolves its interpreter with a plain `include(deps.jl)` — not `include_dependency` —
    and freezes it into `const _current_python` at PRECOMPILE time, so rewriting `deps.jl` left the
    `.ji` cache stale and still valid as far as Julia was concerned.
  * That config is depot-GLOBAL: one build shared by every test environment, so the suites fought
    over a single setting and a child process could not hold a different one.
  * Importing this module loaded PyCall before the build could run.

Switching packages therefore needed the suite run TWICE, and the first run did not stop — it warned
and then executed the whole suite against the previous package's Python.

PythonCall reads `JULIA_PYTHONCALL_EXE` at RUNTIME, per process. Nothing is built, nothing goes
stale, and separate processes can hold separate interpreters — which is what makes a per-env test
split possible at all.

# Errors

Throws if the live interpreter is not `pythonpath`. Hard failure on purpose: the old behaviour was
to warn and carry on, which produced a full, green-looking suite run against the wrong Python.
"""
function setup(pythonpath::String)
    isfile(pythonpath) || throw(ArgumentError("no Python interpreter at $pythonpath"))
    @info "Activating python env at $(abspath(pythonpath, "..", ".."))"

    # No-ops if PythonCall is already live; see the calling contract above.
    ENV["JULIA_PYTHONCALL_EXE"] = pythonpath
    get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")

    # The interpreter PythonCall actually initialised with, not the request just made.
    live = PythonCall.C.CTX.exe_path
    if live !== nothing && abspath(String(live)) != abspath(pythonpath)
        error("""
              PythonCall is initialised with a different interpreter than this suite wants.

                wanted: $(abspath(pythonpath))
                actual: $(abspath(String(live)))

              The interpreter is fixed at PythonCall's initialisation and cannot be changed in this
              process. Set JULIA_PYTHONCALL_EXE before the first import (see the calling contract
              in this function's docstring), or run this suite in its own process.
              """)
    end
    return nothing
end
