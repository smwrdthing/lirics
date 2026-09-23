import numpy as np
import matplotlib.pyplot as plt

from lirics import design, fields, grid, transform


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

# Fields construction
VLstar = 0.12*cell.V
old_cell_field = fields.CellField(cell, SHAPE, VLstar, DENSITY, OMEGA)
cell_field = fields.CellField(cell, SHAPE, VLstar, DENSITY, OMEGA)

# Fields computation
Q = 1.5e-3
cell_field.t += DT
cell_field.VL += Q*DT
cell_field.solve(old_cell_field)

# Plotting interface capturing results
fig, ax = plt.subplots()
ax.set_title("Interface capturing in the cell\nof the liquid ring machine")
ax.set_xlabel("x, mm")
ax.set_ylabel("y, mm")
ax.set_aspect("equal")

# Impeller bounds
xhub, yhub = transform.rphi_to_xy(
    cell.rhub*np.ones(100), np.linspace(0, 2*np.pi, 100))
xrim, yrim = transform.rphi_to_xy(
    cell.rrim*np.ones(100), np.linspace(0, 2*np.pi, 100))
# Grid
x_grid, y_grid = transform.rphi_to_xy(cell_field.r, cell_field.phi)

# Interface
xif, yif = transform.rphi_to_xy(cell_field.rif, cell_field.phiif)

xlims = (np.min(x_grid)*1e3-20, np.max(x_grid)*1e3+20)
ylims = (np.min(y_grid)*1e3-20, np.max(y_grid)*1e3+20)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.plot(xhub*1e3, yhub*1e3, 'k')
ax.plot(xrim*1e3, yrim*1e3, 'k')
ax.plot(x_grid*1e3, y_grid*1e3, '0.5', linewidth=0.8, alpha=0.4)
ax.plot(x_grid.T*1e3, y_grid.T*1e3, '0.5', linewidth=0.8, alpha=0.4)
ax.plot(xif*1e3, yif*1e3, 'C0')

plt.show()
