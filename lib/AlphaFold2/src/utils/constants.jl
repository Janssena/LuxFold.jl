using LinearAlgebra: norm, dot, cross, I

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

# ==============================================================================
# Chi Angles — Task 2.3
# ==============================================================================

"""
    chi_angles_atoms

Dict mapping 3-letter residue codes to lists of chi-angle defining atom tuples.
Each tuple is [atom1, atom2, atom3, atom4] defining a dihedral angle.
ALA and GLY have no chi angles (empty list).
"""
const chi_angles_atoms = Dict(
    "ALA" => [],
    "ARG" => [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS" => [["N", "CA", "CB", "SG"]],
    "GLN" => [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU" => [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY" => [],
    "HIS" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE" => [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS" => [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET" => [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER" => [["N", "CA", "CB", "OG"]],
    "THR" => [["N", "CA", "CB", "OG1"]],
    "TRP" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR" => [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL" => [["N", "CA", "CB", "CG1"]],
)

# ==============================================================================
# Chi Angles Mask — Task 2.4
# ==============================================================================

"""
    chi_angles_mask

20×4 Bool array: mask[restype_idx, chi_idx] = 1.0 if chi angle exists.
Order matches `restypes`. For each residue, up to 4 chi angles are possible.
"""
const chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# ==============================================================================
# Atom Types and Order — Task 2.5
# ==============================================================================

"""
    atom_types

List of 37 distinct heavy atom names used in the atom37 representation.
These are indices 0-36 in the standard protein atom encoding.
"""
const atom_types = [
    "N", "CA", "C", "CB", "O",
    "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]

"""
    atom_order

Dict mapping atom names to their 0-based indices in the atom37 representation.
"""
const atom_order = Dict(atom => i - 1 for (i, atom) in enumerate(atom_types))

# ==============================================================================
# Van der Waals Radii — Task 2.6
# ==============================================================================

"""
    van_der_waals_radius

Dict mapping atom element (first character) to van der Waals radius in Ångströms.
Used for clash detection and atomic distance calculations.
"""
const van_der_waals_radius = Dict(
    "C" => 1.7f0,
    "N" => 1.55f0,
    "O" => 1.52f0,
    "S" => 1.8f0,
)

# ==============================================================================
# Backbone Atoms — Task 2.7
# ==============================================================================

"""
    backbone_atoms

List of atom names that form the protein backbone (present in all residues).
"""
const backbone_atoms = ["N", "CA", "C", "O"]

"""
    backbone_atom_order

Dict mapping backbone atom names to 0-based indices.
"""
const backbone_atom_order = Dict(atom => i - 1 for (i, atom) in enumerate(backbone_atoms))

# ==============================================================================
# Residue Atoms — Supporting data for atom37/atom14 derivation
# ==============================================================================

"""
    residue_atoms

Dict mapping 3-letter residue codes to lists of actual atoms present in each AA.
Uses PDB naming convention. Used to compute atom37 and atom14 representations.
"""
const residue_atoms = Dict(
    "ALA" => ["C", "CA", "CB", "N", "O"],
    "ARG" => ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP" => ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN" => ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS" => ["C", "CA", "CB", "N", "O", "SG"],
    "GLU" => ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN" => ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY" => ["C", "CA", "N", "O"],
    "HIS" => ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE" => ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU" => ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS" => ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET" => ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE" => ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO" => ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER" => ["C", "CA", "CB", "N", "O", "OG"],
    "THR" => ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP" => [
        "C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        "N", "NE1", "O",
    ],
    "TYR" => [
        "C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH",
    ],
    "VAL" => ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
)

# ==============================================================================
# Rigid Group Atom Positions — Task 2.8 (supporting data)
# ==============================================================================

"""
    rigid_group_atom_positions

Dict mapping 3-letter residue codes to lists of [atom_name, group_idx, (x, y, z)].
Defines relative coordinates of atoms within 8 rigid groups:
  0: backbone, 1: pre-omega (empty), 2: phi (empty), 3: psi,
  4-7: chi1, chi2, chi3, chi4
Used to compute restype_atom37_rigid_group_positions and similar.
"""
const rigid_group_atom_positions = Dict(
    "ALA" => [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG" => [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN" => [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP" => [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS" => [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN" => [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU" => [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY" => [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS" => [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE" => [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU" => [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS" => [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET" => [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE" => [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO" => [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],
    ],
    "SER" => [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR" => [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP" => [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR" => [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL" => [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
)

# ==============================================================================
# Atom14 Names — Supporting data for atom14 representation
# ==============================================================================

"""
    restype_name_to_atom14_names

Dict mapping 3-letter residue codes to lists of 14 atom names.
A compact representation: positions 0-3 are always backbone (N, CA, C, O),
then side chain atoms up to position 13 (padded with empty strings).
Used to compute restype_atom14_* tables and atom position arrays.
"""
const restype_name_to_atom14_names = Dict(
    "ALA" => ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG" => ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS" => ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY" => ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS" => ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS" => ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET" => ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO" => ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER" => ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR" => ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
)

# ==============================================================================
# Derived Atom37 and Atom14 Tables — Tasks 2.8-2.9
# ==============================================================================

"""
    restype_atom37_to_rigid_group

Array [21, 37] (Int32): maps (restype, atom37_idx) → rigid_group_idx.
Computed from rigid_group_atom_positions.
"""
const restype_atom37_to_rigid_group = let
    arr = zeros(Int32, 21, 37)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        for (atom_name, group_idx, _) in rigid_group_atom_positions[res_name]
            atom_idx = atom_order[atom_name]
            arr[res_idx, atom_idx + 1] = group_idx  # +1 for 1-based Julia indexing
        end
    end
    
    return arr
end

"""
    restype_atom37_mask

Array [21, 37] (Bool): true where an atom exists for each residue in atom37.
"""
const restype_atom37_mask = let
    arr = falses(21, 37)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        for atom_name in residue_atoms[res_name]
            atom_idx = atom_order[atom_name]
            arr[res_idx, atom_idx + 1] = true
        end
    end
    
    return arr
end

"""
    restype_atom37_rigid_group_positions

Array [21, 37, 3] (Float32): relative (x, y, z) positions of atoms within rigid groups.
"""
const restype_atom37_rigid_group_positions = let
    arr = zeros(Float32, 21, 37, 3)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        for (atom_name, _, (x, y, z)) in rigid_group_atom_positions[res_name]
            atom_idx = atom_order[atom_name]
            arr[res_idx, atom_idx + 1, 1] = Float32(x)
            arr[res_idx, atom_idx + 1, 2] = Float32(y)
            arr[res_idx, atom_idx + 1, 3] = Float32(z)
        end
    end
    arr
end

"""
    restype_atom14_to_rigid_group

Array [21, 14] (Int32): maps (restype, atom14_idx) → rigid_group_idx.
Computed from rigid_group_atom_positions and restype_name_to_atom14_names.
"""
const restype_atom14_to_rigid_group = let
    arr = zeros(Int32, 21, 14)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        atom14_names = restype_name_to_atom14_names[res_name]
        atom_name_to_group = Dict(
            atom_name => group_idx
            for (atom_name, group_idx, _) in rigid_group_atom_positions[res_name]
        )
        for (atom14_idx, atom14_name) in enumerate(atom14_names)
            if atom14_name != ""
                arr[res_idx, atom14_idx] = atom_name_to_group[atom14_name]
            end
        end
    end
    arr
end

"""
    restype_atom14_mask

Array [21, 14] (Bool): true where an atom exists in the atom14 representation for
that residue type; false for unused slots. Row 21 (Unk) is all false.
"""
const restype_atom14_mask = let
    arr = falses(21, 14)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        atom14_names = restype_name_to_atom14_names[res_name]
        for (atom14_idx, atom14_name) in enumerate(atom14_names)
            if atom14_name != ""
                arr[res_idx, atom14_idx] = true
            end
        end
    end
    arr
end

"""
    restype_atom14_rigid_group_positions

Array [21, 14, 3] (Float32): relative (x, y, z) positions of atom14 atoms in rigid groups.
"""
const restype_atom14_rigid_group_positions = let
    arr = zeros(Float32, 21, 14, 3)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        atom14_names = restype_name_to_atom14_names[res_name]
        atom_name_to_pos = Dict(
            atom_name => pos
            for (atom_name, _, pos) in rigid_group_atom_positions[res_name]
        )
        for (atom14_idx, atom14_name) in enumerate(atom14_names)
            if atom14_name != ""
                (x, y, z) = atom_name_to_pos[atom14_name]
                arr[res_idx, atom14_idx, 1] = Float32(x)
                arr[res_idx, atom14_idx, 2] = Float32(y)
                arr[res_idx, atom14_idx, 3] = Float32(z)
            end
        end
    end
    arr
end

# ==============================================================================
# Default Rigid Group Frames — restype_rigid_group_default_frame [21, 8, 4, 4]
# ==============================================================================

"""
    _make_rigid_transformation_4x4(ex, ey, translation) -> Matrix{Float32}

Build an orthonormal 4×4 homogeneous transformation matrix from axis vectors
`ex` (x-direction) and `ey` (approximate y-direction) via Gram-Schmidt.

- Column 1: `ex_n = ex / ‖ex‖`
- Column 2: `ey_n = (ey − (ey·ex_n) ex_n) / ‖…‖` (Gram-Schmidt)
- Column 3: `ez_n = ex_n × ey_n`
- Column 4: `translation`
- Bottom row: `[0, 0, 0, 1]`

Matches OpenFold's `_make_rigid_transformation_4x4` in `openfold/np/residue_constants.py`.
"""
function _make_rigid_transformation_4x4(ex::AbstractVector, ey::AbstractVector,
                                          translation::AbstractVector)
    ex_n  = ex / norm(ex)
    ey_n  = ey - dot(ey, ex_n) .* ex_n
    ey_n  = ey_n / norm(ey_n)
    ez_n  = cross(ex_n, ey_n)
    m = zeros(Float32, 4, 4)
    m[1:3, 1] .= Float32.(ex_n)
    m[1:3, 2] .= Float32.(ey_n)
    m[1:3, 3] .= Float32.(ez_n)
    m[1:3, 4] .= Float32.(translation)
    m[4, 4] = 1f0
    return m
end

"""
    _make_rigid_group_default_frames() -> Array{Float32, 4}

Compute `restype_rigid_group_default_frame [21, 8, 4, 4]` from existing geometry
data (`rigid_group_atom_positions`, `chi_angles_atoms`, `chi_angles_mask`).

Mirrors `_make_rigid_group_constants()` in `openfold/np/residue_constants.py`.

Index conventions (1-based):
- Dim 1: residue type (1–20 = standard AA, 21 = Unk)
- Dim 2: rigid group (1=backbone, 2=pre-omega, 3=phi, 4=psi, 5–8=chi1–chi4)
- Dims 3–4: 4×4 homogeneous rotation-translation matrix
"""
function _make_rigid_group_default_frames()
    frames = zeros(Float32, 21, 8, 4, 4)

    # Groups 1 (backbone) and 2 (pre-omega) are identity for the 20 valid residues.
    # Unk (index 21) keeps the zero-initialised default — matches OpenFold's convention.
    identity4 = Matrix{Float32}(I, 4, 4)
    for r in 1:length(restypes)   # 1:20 only
        frames[r, 1, :, :] .= identity4
        frames[r, 2, :, :] .= identity4
    end

    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]

        # Build a position lookup keyed by (atom_name, group_0based)
        pos_by_atom_group = Dict{Tuple{String,Int}, Vector{Float64}}()
        for (atom_name, g, (x, y, z)) in rigid_group_atom_positions[res_name]
            pos_by_atom_group[(atom_name, g)] = [x, y, z]
        end

        # Helper: get position of an atom in a specific (0-based) group
        function atom_pos(name, group_0based)
            key = (name, group_0based)
            haskey(pos_by_atom_group, key) || return nothing
            return pos_by_atom_group[key]
        end

        # Group 3 (phi): ex = N − CA (backbone), ey = [1,0,0], trans = N
        n_pos  = atom_pos("N",  0)
        ca_pos = atom_pos("CA", 0)
        if n_pos !== nothing && ca_pos !== nothing
            ex = n_pos .- ca_pos
            ey = [1.0, 0.0, 0.0]
            frames[res_idx, 3, :, :] .= _make_rigid_transformation_4x4(ex, ey, n_pos)
        end

        # Group 4 (psi): ex = C − CA (backbone), ey = CA − N, trans = C
        c_pos = atom_pos("C", 0)
        if c_pos !== nothing && ca_pos !== nothing && n_pos !== nothing
            ex = c_pos .- ca_pos
            ey = ca_pos .- n_pos
            frames[res_idx, 4, :, :] .= _make_rigid_transformation_4x4(ex, ey, c_pos)
        end

        # Groups 5–8 (chi1–chi4)
        chi_atom_list = chi_angles_atoms[res_name]
        for chi_idx in 1:4
            chi_idx > length(chi_atom_list)   && break
            chi_angles_mask[res_idx][chi_idx] == 0.0 && continue

            atoms = chi_atom_list[chi_idx]   # [atom1, atom2, atom3, atom4]
            julia_group = chi_idx + 4        # chi1→5, chi2→6, chi3→7, chi4→8

            if chi_idx == 1
                # chi1 frame defined in the backbone frame (group 0)
                p1 = atom_pos(atoms[1], 0)
                p2 = atom_pos(atoms[2], 0)
                p3 = atom_pos(atoms[3], 0)
                (p1 === nothing || p2 === nothing || p3 === nothing) && continue
                ex    = p3 .- p2
                ey    = p1 .- p2
                trans = p3
            else
                # chi2–chi4: axis_end = atom3 in the previous chi frame
                # chi1 = 0-based group 4, chi2 = group 5, chi3 = group 6
                prev_group_0based = chi_idx + 2   # chi2→4, chi3→5, chi4→6
                axis_end = atom_pos(atoms[3], prev_group_0based)
                axis_end === nothing && continue
                ex    = axis_end
                ey    = [-1.0, 0.0, 0.0]
                trans = axis_end
            end

            frames[res_idx, julia_group, :, :] .=
                _make_rigid_transformation_4x4(ex, ey, trans)
        end
    end

    return frames
end

"""
    restype_rigid_group_default_frame

Array `[21, 8, 4, 4]` (Float32): default 4×4 homogeneous rigid-body frames for
each residue type and rigid group.

Index conventions (1-based):
- Dim 1: residue type (1=ALA … 20=VAL, 21=Unk)
- Dim 2: rigid group (1=backbone, 2=pre-omega, 3=phi, 4=psi, 5=chi1, …, 8=chi4)
- Dims 3–4: 4×4 rotation-translation matrix

Computed by `_make_rigid_group_default_frames()` at module load time from existing
geometry data — not hardcoded. Matches `openfold.np.residue_constants.restype_rigid_group_default_frame`
to Float32 precision (after 0→1 index base shift).

Forward-pass callers should access this via `st.residue_constants.default_frames`
(device-resident after `Lux.setup`) rather than using this global constant directly.
"""
const restype_rigid_group_default_frame = _make_rigid_group_default_frames()

# ==============================================================================
# atom37_to_atom14 Lookup — restype_atom37_to_atom14 [21, 37]
# ==============================================================================

"""
    restype_atom37_to_atom14

Array `[21, 37]` (Int32): maps `(restype, atom37_slot)` → 1-based atom14 index.

- Row = residue type (1-based, same as `restype_order`)
- Column = atom37 slot (1-based, `atom_order[name] + 1`)
- Value = 1-based atom14 index, or `1` for invalid slots (safe placeholder —
  masked out by `restype_atom37_mask`)

Matches `openfold.np.residue_constants.restype_atom37_to_atom14 + 1` (OpenFold is 0-based).
"""
const restype_atom37_to_atom14 = let
    arr = ones(Int32, 21, 37)   # default 1 (safe placeholder for invalid slots)
    for (res_idx, res_letter) in enumerate(restypes)
        res_name = restype_1to3[res_letter]
        atom14_names = restype_name_to_atom14_names[res_name]
        for (atom14_idx, atom14_name) in enumerate(atom14_names)
            if atom14_name != ""
                atom37_idx = atom_order[atom14_name] + 1  # 0-based → 1-based
                arr[res_idx, atom37_idx] = Int32(atom14_idx)
            end
        end
    end
    arr
end
