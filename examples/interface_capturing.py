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

# Cell construction
cell = design.ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    rarch=design.infer_arch_radius(rhub, rrim, np.deg2rad(55)),
    l=100e-3,
    delta=np.deg2rad(360/NUM_OF_CELLS),
    s=5e-3
)

# Fields construction
starred = fields.CellField(cell, (N_R_SEGMENTS, N_PHI_SEGMENTS))
current = fields.CellField(cell, (N_R_SEGMENTS, N_PHI_SEGMENTS))

# Fields initialization
astVL = 3e-4
Q = 5e-3
dVL = Q*DT
VL = astVL + dVL
starred.t = current.t = 0
starred.alpha = current.alpha = 0
starred.omega = current.omega = OMEGA
starred.rhoL = current.rhoL = DENSITY
starred.VL = current.VL = astVL
starred.u[:] = starred.w[:] = 0

# Fields computation
current.t += DT
current.alpha += DALPHA
current.VL += dVL
current.U(starred)
current.dUdt(starred)
current.dUdr()
current.gradP()

rref = 0.6*(cell.rhub+cell.rrim)
current.capture_inteface(rref)
loop = current.evalvof()
actVL = current.actVL

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
x_grid, y_grid = transform.rphi_to_xy(current.r, current.phi)

# Interface
xif, yif = transform.rphi_to_xy(current.rif, current.phiif)

xlims = (np.min(x_grid)*1e3-20, np.max(x_grid)*1e3+20)
ylims = (np.min(y_grid)*1e3-20, np.max(y_grid)*1e3+20)

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.plot(xhub*1e3, yhub*1e3, 'k')
ax.plot(xrim*1e3, yrim*1e3, 'k')
ax.plot(x_grid*1e3, y_grid*1e3, '0.5', linewidth=0.8, alpha=0.4)
ax.plot(x_grid.T*1e3, y_grid.T*1e3, '0.5', linewidth=0.8, alpha=0.4)
ax.plot(xif*1e3, yif*1e3, 'C0')

xloop, yloop = transform.rphi_to_xy(*loop)
ax.plot(xloop*1e3, yloop*1e3, "r.")

xcenter = float(np.mean(xlims))
ycenter = float(np.mean(ylims))
xtxt = xlims[0]+15
ytxt = [ylims[-1]-10]
dytxt = 8
for i in range(5):
    ytxt.append(ytxt[-1]-dytxt)

s = 8
ax.text(xtxt, ytxt[0], r"$\Delta t=$"+f"{DT*1e3:.2f} ms", {"size": s})
ax.text(xtxt, ytxt[1], r"$Q^{(L)}=$"+f"{Q*1e3*60:.2f} L/min", {"size": s})
ax.text(xtxt, ytxt[2], r"$V^{(L)}=$"+f"{VL*1e3:.2e} L", {"size": s})
ax.text(xtxt, ytxt[3], r"$V^{*(L)}=$"+f"{astVL*1e3:.2e} L", {"size": s})
ax.text(xtxt, ytxt[4], r"$\Delta V^{(L)}=$"+f"{dVL*1e3:.2e} L", {"size": s})
ax.text(xtxt, ytxt[5], r"$V^{(L)}_{act.}=$"+f"{actVL*1e3:.2e} L", {"size": s})

plt.show()
