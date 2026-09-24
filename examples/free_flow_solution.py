import numpy as np
import matplotlib.pyplot as plt

from lirics import design, fields

# For free flow solution we need to construct proper cell-flow solution for
# coupling

# Setup
RPM = 1500
OMEGA = np.pi*RPM/30
DENSITY = 1000
VISCOSITY = 1e-3

DALPHA = np.deg2rad(1)
DT = DALPHA/OMEGA

NUM_OF_CELLS = 12

N_R_SEGMENTS = 31
N_PHI_SEGMENTS = 31
SHAPE = (N_R_SEGMENTS, N_PHI_SEGMENTS)

# Cell construction
cell = design.ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    rarch=design.infer_arch_radius(rhub, rrim, np.deg2rad(55)),
    l=100e-3,
    delta=np.deg2rad(360/NUM_OF_CELLS),
    s=5e-3
)

# Housing construction
housing = design.CylindricalHousing(
    cell.l,
    0.15*cell.rrim,
    design.infer_housing_radius(cell.rrim, 0.15*cell.rrim, 5e-3))


# Cell fields construction
astcf = fields.CellField(cell, SHAPE)
cf = fields.CellField(cell, SHAPE)

# Cell fields initialization
astVL = 0.12*cell.V
Q = -2.27e-4  # adjusted manually to produce smallest imbalance, test more
dVL = Q*DT
VL = astVL + dVL
astcf.t = cf.t = 0
astcf.alpha = cf.alpha = 0
astcf.omega = cf.omega = OMEGA
astcf.rhoL = cf.rhoL = DENSITY
astcf.VL = cf.VL = astVL
astcf.u[:] = astcf.w[:] = 0

# Cell fields solution
cf.t += DT
cf.alpha += DALPHA
cf.VL += Q*DT
cf.solve(astcf)

# Free field construction
astff = fields.UniformFreeField(cell, housing)
ff = fields.UniformFreeField(cell, housing)

# Free fields initialization
astff.rho = ff.rho = DENSITY
astff.mu = ff.mu = VISCOSITY
astff.avW = 0.5 * OMEGA * cell.rrim
astff.avP = 1e5 + astff.rho*astff.avW**2*astff.kWCF(astff.alpha+astff.phir)
ff.solve(astff, cf)

# Imbalance computations, added here preliminary, should be separate example
# coupled_flow_solution.py?
#
# What is important now - procedure goes through without errors, we are finally in the
# position to resolve coupled flow in the liquid ring machine with 2D model.
#
# For accurate results we still need good assumption about velocity field
print(fields.imbalance(astcf, cf, ff))
