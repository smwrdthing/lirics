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

# cell object construction
cell = design.ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    rarch=design.infer_arch_radius(rhub, rrim, np.deg2rad(55)),
    l=100e-3,
    delta=np.deg2rad(360/NUM_OF_CELLS),
    s=3e-3
)

r = np.linspace(cell.rhub, cell.rrim)
mu = cell.mu(r)

fig, ax = plt.subplots()
ax.set_xlabel("r, mm")
ax.set_ylabel(r"$\mu$")
ax.grid(True)
ax.plot(r*1e3, mu)
ax.plot(r*1e3, np.ones_like(r)*cell.avmu, 'r--')
ax.set_xlim(1e3*r[[0, -1]])
ax.set_ylim((0.95*np.min(mu), 1))
plt.show()
