import numpy as np
import matplotlib.pyplot as plt

from lirics.design import ArchImpellerCell, CylindricalHousing
from lirics.design import infer_arch_radius, infer_housing_radius
from lirics.ring import PGRGenralized, PGRClassic, PGRModified, PGRParams


cell = ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    l := 200e-3,
    delta=np.deg2rad(360/12),
    s=5e-3,
    rarch=infer_arch_radius(rhub, rrim, np.deg2rad(45))
)
housing = CylindricalHousing(
    L := 1.01*l,
    e := 11e-3,
    Rc=infer_housing_radius(rrim, e, Smin=5e-3)
)

params: PGRParams = {
    "alphadis": np.deg2rad(270),
    "alphamax": np.deg2rad(360),
    "omega": np.pi*1500/30,
    "pVsuc": 101_325,
    "rhoL": 1000
}

model_calssic = PGRClassic(cell, housing, params)
model_modified = PGRModified(cell, housing, params)

alpha = np.linspace(0, 2*np.pi, 100)
Rhousing = housing.R(alpha-np.pi)
rhub = cell.rhub*np.ones_like(alpha)
rrim = cell.rrim*np.ones_like(alpha)

rhub = 1e3*rhub
rrim = 1e3*rrim
Rhousing = 1e3*Rhousing


def process_model(model: PGRGenralized, linestyle: str, every: int = 1):

    rif, alpha = model.interfaceRPHI()

    # Filetering regions and preparing refpoints
    alphasuc = model_calssic.alphasuc()
    alphacom = model_calssic.alphacom()
    alphadis = model_calssic.alphadis()

    suc = alpha <= alphasuc[-1]
    com = (alphasuc[-1] <= alpha) * (alpha <= alphacom[-1])
    dis = (alphacom[-1] <= alpha) * (alpha <= alphadis[-1])

    alpharef = np.concatenate((
        alpha[suc][[0]],
        alpha[com][[0, -1]]))
    rref = np.concatenate((
        rif[suc][[0]],
        rif[com][[0, -1]]))

    rif = rif*1e3
    rref = rref*1e3

    ax[0].plot(alpha[suc], rif[suc], 'g'+linestyle, markevery=every)
    ax[0].plot(alpha[com], rif[com], 'y'+linestyle, markevery=every)
    ax[0].plot(alpha[dis], rif[dis], 'r'+linestyle, markevery=every)
    ax[0].plot(alpharef, rref, "ko")

    return rif, alpha


to_inches = 1/25.4
w = 210*to_inches
h = 100*to_inches
fig, ax = plt.subplots(
    ncols=2, subplot_kw={"projection": "polar"},
    figsize=(w, h), layout="constrained")

ax[0].set_title("Interface recontruction with 1D PGR models")
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)
ax[0].set_aspect("equal")

ax[1].set_title("Modified model error estimation [mm]")
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)

ax[0].plot([], "g")
ax[0].plot([], "y")
ax[0].plot([], "r")
# ax[0].legend(["suction", "compression", "discharge"], loc="center")

ax[0].plot(alpha, rhub, "k", linewidth=1)
ax[0].plot(alpha, rrim, "k", linewidth=1)
ax[0].plot(alpha, Rhousing, "k", linewidth=1)

rif_calssic, alpha_classic = process_model(model_calssic, '-')
rif_modified, alpha_modified = process_model(model_modified, '.', every=5)

ax[1].plot(alpha_classic, (rif_modified-rif_calssic))

plt.show()

# TODO : interface reconstruction for double acting lrc
# TODO : normalized ranges comparison
