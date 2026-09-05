# The REAL boltz layers behind LuxFoldCore's shared layers. See `af2.jl` for why these are imported
# rather than copied. Needs the boltz venv — see `runtests_boltz2.jl`.
#
# `attentionv2` / `transformersv2`, not `attention` / `transformers`: boltz ships two generations of
# both and this repo ports the v2 line throughout (see `lib/Boltz2/src`).
const PyBoltz2AdaLN                 = pyimport("boltz.model.modules.transformersv2").AdaLN
const PyBoltz2OuterProductMean      = pyimport("boltz.model.layers.outer_product_mean").OuterProductMean
const PyBoltz2PairWeightedAveraging = pyimport("boltz.model.layers.pair_averaging").PairWeightedAveraging
const PyBoltz2AttentionPairBias     = pyimport("boltz.model.layers.attentionv2").AttentionPairBias
