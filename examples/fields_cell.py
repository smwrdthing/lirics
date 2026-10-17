import numpy as np
import matplotlib.pyplot as plt
from lirics import design, fields, transform


# Setup
RPM = 1500
OMEGA = np.pi*RPM/30
LIQ_DENSITY = 1000
VAP_MOLMASS = 28.97e-3
VAP_HCRATIO = 1.4

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

# Fields initialization
astVL = 0.15*cell.V
Q = 5e-3
dVL = Q*DT
VL = astVL + dVL

starred = fields.CellField(
    {
        "cell": cell,
        "shape": SHAPE,

        "t": 0.0,
        "alpha": 0.0,
        "omega": OMEGA,

        "u": 0.0,

        "VL": astVL,
        "rhoL": LIQ_DENSITY,
        "pV": 101325,
        "TV": 20 + 273.15,
        "MV": VAP_MOLMASS,
        "nV": VAP_HCRATIO,
    }
)
current = fields.CellField(
    {
        "cell": cell,
        "shape": SHAPE,

        "t": starred.t + DT,
        "alpha": starred.alpha + DALPHA,
        "omega": OMEGA,

        "VL": VL,
        "rhoL": LIQ_DENSITY,
        "MV": VAP_MOLMASS,
        "nV": VAP_HCRATIO,
    }
)

current.U(starred)
current.dUdt(starred)
current.dUdr()
current.gradP()

# grid points
x_grid, y_grid = transform.rphi_to_xy(current.r, current.phi)
x_grid, y_grid = transform.scale(x_grid, y_grid, 1e3)

# radial velocity
fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.plot(x_grid, y_grid, 'k', linewidth=0.5)
ax.plot(x_grid.T, y_grid.T, 'k', linewidth=0.5)
cf = ax.contourf(x_grid, y_grid, current.u, levels=100)
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel("u [m/s]")

# tangent velocity
fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.plot(x_grid, y_grid, 'k', linewidth=0.5)
ax.plot(x_grid.T, y_grid.T, 'k', linewidth=0.5)
cf = ax.contourf(x_grid, y_grid, current.w, levels=100)
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel("w [m/s]")

# pressure radial derivative
fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.plot(x_grid, y_grid, 'k', linewidth=0.5)
ax.plot(x_grid.T, y_grid.T, 'k', linewidth=0.5)
cf = ax.contourf(x_grid, y_grid, current.dpdr/1e6, levels=100)
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel(r"$\partial p / \partial r$ [MPa/m]")

# pressure tangent derivative
fig, ax = plt.subplots()
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.plot(x_grid, y_grid, 'k', linewidth=0.5)
ax.plot(x_grid.T, y_grid.T, 'k', linewidth=0.5)
cf = ax.contourf(x_grid, y_grid, current.dpdphi/1e3, levels=100)
cbar = fig.colorbar(cf)
cbar.ax.set_ylabel(r"$\partial p / \partial \varphi$ [kPa/rad]")


plt.show()
