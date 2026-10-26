import numpy as np
import matplotlib.pyplot as plt

from lirics import design, fields, transform

# For free flow solution we need to construct proper cell-flow solution for
# coupling

# Setup
RPM = 1500
OMEGA = np.pi*RPM/30
LIQ_DENSITY = 1000
LIQ_VISCOSITY = 1e-3
VAP_MOLMASS = 28.97e-3
VAP_HCRATIO = 1.4

DALPHA = np.deg2rad(3)
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
    L=cell.l, epsilon=0.0,
    e=0.15*cell.rrim,
    Rc=design.infer_housing_radius(cell.rrim, 0.15*cell.rrim, 5e-3))


# Cell fields initialization
astVL = 0.2*cell.V
Qs = np.linspace(0, 2, 5)*1e-3

VLs = []
dVs = []

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


for Q in Qs:

    dVL = Q*DT
    VL = astVL + dVL
    VLs.append(VL)

    starred_cf = fields.CellField(
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
    cf = fields.CellField(
        {
            "cell": cell,
            "shape": SHAPE,

            "t": starred_cf.t + DT,
            "alpha": starred_cf.alpha + DALPHA,
            "omega": OMEGA,

            "VL": VL,
            "rhoL": LIQ_DENSITY,
            "MV": VAP_MOLMASS,
            "nV": VAP_HCRATIO,
        }
    )
    starred_cf.GV = cf.GV = 0.0

    # Cell fields solution
    cf.solve(starred_cf)

    # Free field construction
    starred_ff = fields.UniformFreeField(
        {
            "cell": cell,
            "housing": housing,
            "alpha": starred_cf.alpha,
            "rhoL": LIQ_DENSITY,
            "muL": LIQ_VISCOSITY,

            "avW": OMEGA * cell.rrim,
            "Pr": 1e5,
        }
    )
    ff = fields.UniformFreeField(
        {
            "cell": cell,
            "housing": housing,
            "alpha": cf.alpha,
            "rhoL": LIQ_DENSITY,
            "muL": LIQ_VISCOSITY,
        }
    )

    ff.solve(starred_ff, cf)
    dV = fields.imbalance(starred_cf, cf, ff)
    dVs.append(dV)

    # Grid
    if len(ax.lines) == 0:
        x_grid, y_grid = transform.rphi_to_xy(cf.r, cf.phi)
        x_grid, y_grid = x_grid*1e3, y_grid*1e3

        xlims = (np.min(x_grid)*1e3-20, np.max(x_grid)*1e3+20)
        ylims = (np.min(y_grid)*1e3-20, np.max(y_grid)*1e3+20)

        m, n = np.shape(cf.r)

        ax.plot(x_grid, y_grid, "0.5", linewidth=0.75)
        ax.plot(x_grid.T, y_grid.T, "0.5", linewidth=0.75)
        ax.plot(
            x_grid[:, int(n/2)],
            y_grid[:, int(m/2)],
            "C1-.", linewidth=3)

    xif, yif = transform.rphi_to_xy(cf.rif, cf.phiif)
    xif, yif = xif*1e3, yif*1e3
    ax.plot(xif, yif)

VLs = np.ravel(VLs)
dVs = np.ravel(dVs)

VLsol = np.interp(0, dVs, VLs)

fig, ax = plt.subplots()
ax.plot(VLs/cell.V*100, dVs/cell.V*100)
ax.plot(VLsol/cell.V*100, 0, "ro")

plt.show()
