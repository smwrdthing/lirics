import numpy as np
import matplotlib.pyplot as plt

from lirics.design import ArchImpellerCell, CylindricalHousing, EllipticHousing
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
housing_single = CylindricalHousing(
    L := 1.01*l,
    e := 12e-3,
    Rc=infer_housing_radius(rrim, e, Smin := 5e-3)
)
housing_double = EllipticHousing(
    L := L,
    A=housing_single.e+housing_single.Rc,
    B=cell.rrim+Smin
)
params_single: PGRParams = {
    "alphadis": np.deg2rad(270),
    "alphamax": np.deg2rad(360),
    "omega": np.pi*1500/30,
    "pVsuc": 101_325,
    "rhoL": 1000
}
params_double: PGRParams = {
    "alphadis": params_single["alphadis"]/2,
    "alphamax": np.deg2rad(180),
    "omega": np.pi*1500/30,
    "pVsuc": 101_325,
    "rhoL": 1000
}


classic = PGRClassic(cell, housing_single, params_single)
modified = PGRModified(cell, housing_single, params_single)
double = PGRModified(cell, housing_double, params_double)


alpha = np.linspace(0, 2*np.pi, 100)
Rsingle = housing_single.R(alpha-np.pi)
Rdouble = housing_double.R(alpha-np.pi/2)
rhub = cell.rhub*np.ones_like(alpha)
rrim = cell.rrim*np.ones_like(alpha)

rhub = 1e3*rhub
rrim = 1e3*rrim
Rsingle = 1e3*Rsingle
Rdouble = 1e3*Rdouble


def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


def rescale(x, scale):
    return scale*normalize(x)


def process_model(model: PGRGenralized, ax, **kwargs):

    rif, alpha = model.interfaceRPHI()

    # Filetering regions and preparing refpoints
    alphasuc = model.alphasuc()
    alphacom = model.alphacom()
    alphadis = model.alphadis()

    suc = alpha <= alphasuc[-1]
    com = (alphasuc[-1] <= alpha) * (alpha <= alphacom[-1])
    dis = (alphacom[-1] <= alpha) * (alpha <= alphadis[-1])

    alpharef = np.concatenate((
        alpha[suc][[0]],
        alpha[com][[0, -1]],
        alpha[dis][[-1]]))
    rref = np.concatenate((
        rif[suc][[0]],
        rif[com][[0, -1]],
        rif[dis][[-1]]))

    rif = rif*1e3
    rref = rref*1e3

    ax.plot(alpha[suc], rif[suc], 'g', **kwargs)
    ax.plot(alpha[com], rif[com], 'y', **kwargs)
    ax.plot(alpha[dis], rif[dis], 'r', **kwargs)
    ax.plot(alpharef, rref, "ko")

    if alphadis[-1] == np.pi:
        ax.plot(alpha[suc]+np.pi, rif[suc], 'g', **kwargs)
        ax.plot(alpha[com]+np.pi, rif[com], 'y', **kwargs)
        ax.plot(alpha[dis]+np.pi, rif[dis], 'r', **kwargs)
        ax.plot(alpharef+np.pi, rref, "ko")

    return rif, alpha


# Common sizes for plot figures
to_inches = 1/25.4
w = 240*to_inches
h = 120*to_inches


fig, ax = plt.subplots(
    ncols=2, subplot_kw={"projection": "polar"},
    figsize=(w, h), layout="constrained")

# Configuring axes for classic vs modified PGR model comparison
# for single-acting machine
ax[0].set_title("Interface recontruction with 1D PGR models")
ax[0].set_theta_zero_location("N")
ax[0].set_theta_direction(-1)
ax[0].set_aspect("equal")
ax[0].plot([], "g")
ax[0].plot([], "y")
ax[0].plot([], "r")
ax[0].legend(["suction", "compression", "discharge"],
             loc="center", fontsize=8)

ax[0].plot(alpha, rhub, "k", linewidth=1)
ax[0].plot(alpha, rrim, "k", linewidth=1)
ax[0].plot(alpha, Rsingle, "k", linewidth=1)

rif_calssic, alpha_classic = process_model(
    classic, ax[0], linestyle="-")
rif_modified, alpha_modified = process_model(
    modified, ax[0], linestyle='none', marker='.', markevery=5)

# Configuring axes for classic vs modified PGR model error estimation
ax[1].set_title("Modified model error estimation [mm]")
ax[1].set_theta_zero_location("N")
ax[1].set_theta_direction(-1)

# Error plotting for modified vs classic PGR models
ax[1].plot(alpha_classic, (rif_modified-rif_calssic))


# Configuring axes for single-acting vs double-acting modified PGR model comparison
fig, ax = plt.subplot_mosaic(
    "AB",
    per_subplot_kw={
        "A": {"projection": "polar"},
    },
    figsize=(w, h), layout="constrained")

ax["A"].plot([], "g")
ax["A"].plot([], "y")
ax["A"].plot([], "r")
ax["A"].legend(["suction", "compression", "discharge"],
               loc="center", fontsize=8)

ax["A"].set_title("Interface reconstruction for double-acting machine")
ax["A"].set_theta_zero_location("N")
ax["A"].set_theta_direction(-1)

ax["A"].plot(alpha, rhub, "k", linewidth=1)
ax["A"].plot(alpha, rrim, "k", linewidth=1)
ax["A"].plot(alpha, Rdouble, "k", linewidth=1)

rif_double, alpha_double = process_model(double, ax["A"])

# Error plotting for normalized double acting vs modified single acting PGR models
ax["B"].set_title("Normalizer interface profiles comparison")
ax["B"].set_xlabel(r"$\hat{\alpha}\degree$")
ax["B"].set_ylabel(r"$\hat{r}_{if}$")
ax["B"].set_xlim()
ax["B"].grid(True)

alpha_double_norm = normalize(alpha_double)
alpha_single_norm = normalize(alpha_double)
rif_double_norm = normalize(rif_double)
rif_single_norm = normalize(rif_modified)

alpha_sample_norm = np.linspace(0, 1, len(alpha_double_norm))
rif_normalized_error = (
    np.interp(alpha_sample_norm, alpha_double_norm, rif_double_norm)
    - np.interp(alpha_sample_norm, alpha_single_norm, rif_single_norm))


ax["B"].plot(alpha_single_norm, rif_double_norm, "C0")
ax["B"].plot(alpha_double_norm, rif_single_norm, "C1")
ax["B"].plot(alpha_double_norm, rif_normalized_error, "m--")
ax["B"].legend(["single", "double", "error"], loc="upper left", fontsize=10)

plt.show()
