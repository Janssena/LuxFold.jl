"""
Residue and atom constants for AlphaFold2.

This module currently implements the minimal subset of constants needed by the
StructureModule's backbone (rigid frames, IPA, angle resnet). Sidechain atom
placement (`atom14`/`atom37` tables, `chi_angles_*`, `restype_rigid_group_*`)
is deferred until atom-position utilities (`utils/atom_utils.jl`) are added.

The constant tables here match `openfold.np.residue_constants` for parity testing.
"""

# Standard 20 amino acid one-letter codes, in canonical AlphaFold order.
const restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

# Map one-letter code -> 1-based index (Python uses 0-based; we expose
# `restype_order_py` separately for parity tests that compare indices directly).
const restype_order = Dict{String, Int}(aa => i for (i, aa) in enumerate(restypes))

# 0-based map (matches Python directly — useful when comparing aatype tensors).
const restype_order_py = Dict{String, Int}(aa => i - 1 for (i, aa) in enumerate(restypes))

# 1-letter -> 3-letter
const restype_1to3 = Dict(
    "A" => "ALA", "R" => "ARG", "N" => "ASN", "D" => "ASP", "C" => "CYS",
    "Q" => "GLN", "E" => "GLU", "G" => "GLY", "H" => "HIS", "I" => "ILE",
    "L" => "LEU", "K" => "LYS", "M" => "MET", "F" => "PHE", "P" => "PRO",
    "S" => "SER", "T" => "THR", "W" => "TRP", "Y" => "TYR", "V" => "VAL",
)

# Reverse: 3-letter -> 1-letter
const restype_3to1 = Dict(v => k for (k, v) in restype_1to3)

# Extended with unknown residue type "X"
const restypes_with_x = vcat(restypes, ["X"])
const restype_order_with_x = Dict{String, Int}(aa => i for (i, aa) in enumerate(restypes_with_x))
const restype_order_with_x_py = Dict{String, Int}(aa => i - 1 for (i, aa) in enumerate(restypes_with_x))
