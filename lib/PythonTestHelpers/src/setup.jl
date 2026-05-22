# python_path = abspath(joinpath(@__DIR__, "..", "..", ".venv", "bin", "python"))
function setup(pythonpath::String)
    @info "Activating python env at $(abspath(pythonpath, "..", ".."))"
    ENV["PYTHON"] = pythonpath
    try 
        Pkg.build("PyCall")
        @info "PyCall has been rebuild successfully, please run `using PyCall` again."
    catch err
        throw(err)
    end

    return nothing
end