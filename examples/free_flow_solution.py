import numpy as np
import matplotlib.pyplot as plt

from lirics import design, fields, grid, transform

# For free flow solution we need to construct proper cell-flow solution for
# coupling

# Setup
RPM = 1500
OMEGA = np.pi*RPM/30
DENSITY = 1000

DALPHA = np.deg2rad(1)
DT = DALPHA/OMEGA

NUM_OF_CELLS = 12

N_R_SEGMENTS = 31
N_PHI_SEGMENTS = 31
SHAPE = (N_R_SEGMENTS, N_PHI_SEGMENTS)

# cell object construction
cell = design.ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    rarch=design.infer_arch_radius(rhub, rrim, np.deg2rad(55)),
    l=100e-3,
    delta=np.deg2rad(360/NUM_OF_CELLS),
    s=5e-3
)

housing = design.CylindricalHousing(
    cell.l,
    0.15*cell.rrim,
    design.infer_housing_radius(cell.rrim, 0.15*cell.rrim, 5e-3))


# Fields construction
VLstar = 0.2*cell.V
astcf = fields.CellField(cell, SHAPE, VLstar, DENSITY, OMEGA)
cf = fields.CellField(cell, SHAPE, VLstar, DENSITY, OMEGA)

# Fields computation
Q = 1.5e-3
cf.t += DT
cf.alpha += DALPHA
cf.VL += Q*DT
cf.solve(astcf)

# Free field definitions
astff = fields.UniformFreeField(cell, housing)
ff = fields.UniformFreeField(cell, housing)
ff.alpha += DALPHA
