module LuxFoldCoreReactantExt

using LuxFoldCore: AttentionPairBias
using Reactant: AnyTracedRArray

(l::AttentionPairBias)(x, z, mask::AnyTracedRArray{Bool}, ps, st) =
    l(x, z, nothing, mask, ps, st)

(l::AttentionPairBias)(x, z, cond::AnyTracedRArray{<:Real}, ps, st) =
    l(x, z, cond, nothing, ps, st)

end # module LuxFoldCoreReactantExt
