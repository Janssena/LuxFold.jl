"""
    LayerNormNoBias(args...; kwargs...)

There is no standard way of setting use_bias = false for Lux.LayerNorm
so we instead freeze the bias and init with zeros to achieve the same effect.
"""
LayerNormNoBias(args...; kwargs...) = Lux.Experimental.freeze(
    LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...), 
    (:bias, )
)

"""
    Activation(activation::Function)

Simple wrapper around a Lux.WrappedFunction that broadcasts the passed function
over inputs x.
"""
Activation(activation::Function) = 
    Lux.WrappedFunction(Base.Fix1(broadcast, activation))

"""
    ReLU()

Runs the ReLU activation function over inputs.
"""
ReLU() = Activation(Lux.relu)