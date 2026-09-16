import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lirics import design, fields, transform, calculus
import pathlib


"""Comparison of the interface capturing algorithm application results and interface shape
predicted in the paper:
    "Evaluation of different turbulence models on simulation of gas-liquid transient flow
    in a liquid-ring vacuum pump" by G. Guo, R. Zhang and H. Yu
    (DOI: 10.1016/j.vacuum.2020.109586)

Code in this file performs solution of multiple interface capturing problems and dumps
results to the /data/IndependentCFDComparison directory of the repo.

Liquid ring machine dimensions are adopted from referred paper as well as necessary
operation data. Reuired parameters for interface capturing are reconstructed by means of
CAD processing of pictures in the referred paper and curve fitting.

Interface capturing data is dumped as set of csv files with interface location points,
each csv file corresponds to one of the cells in referred paraper.

Generated csv files are then used in CAD to produce splines and visually approve interface
capturing by comparison with interface constructed from pictures in the referred paper."""


# Storage path handling
examples = pathlib.Path(__file__).resolve().parent
repo = examples.parent
data = repo / "data"
storage = data / "IndependentCFDComparison"


# Dimensions of the liquid ring machine
n = 18
l = 130e-3
rhub = 91e-3
rrim = 183e-3
betarim_deg = 50
s = 6.3e-3
R = 212e-3
e = 23.1e-3

delta_deg = 360/n
delta = np.deg2rad(delta_deg)
betarim = np.deg2rad(betarim_deg)


# Operation data
rpm = 1450
omega = np.pi*rpm/30
rho = 1000


# Data extracted with CAD

# Liquid-occupied area
ALdata = np.array([
    7.5656590,  # cell0
    11.887639,  # cell1
    17.241374,  # cell2
    22.429805,  # cell3
    26.266830,  # cell4
    28.719064,  # cell5
    31.963378,  # cell6
    33.864694,  # cell7
    34.773390,  # cell8
    34.162394,  # cell9
    31.183360,  # cell10
    27.731013,  # cell11
    22.291546,  # cell12
    18.220097,  # cell13
    13.978069,  # cell14
    10.835036,  # cell15
    7.3954270,  # cell16
    6.5034830,  # cell17
])*1e-4

# Liquid-occupied volume
VLdata = ALdata*l

# Interface locatio at the midline for each cell
rifdata = np.array([
    169.60,  # cell0
    162.00,  # cell1
    150.63,  # cell2
    138.39,  # cell3
    128.02,  # cell4
    121.13,  # cell5
    110.43,  # cell6
    106.99,  # cell7
    106.08,  # cell8
    106.38,  # cell9
    111.23,  # cell10
    117.95,  # cell11
    131.20,  # cell12
    146.70,  # cell13
    157.85,  # cell14
    167.31,  # cell15
    170.90,  # cell16
    169.76,  # cell17
])*1e-3

# Angular position
alpha0data_deg = 7.28413
alphadata_deg = np.arange(alpha0data_deg, alpha0data_deg+360, delta_deg)
alphadata = np.deg2rad(alphadata_deg)


# Curve fitting

nterms = 4
# visual example for nterms = 4:
# V0 + V1*cos(alpha + alph1) + V2*cos(alpha+alph2) + V3*cos(alpha+alph3)
# 4 terms ->
# 4 values of Vi (nterms)
# 3 values of alphi (nterms-1)
# 7 parameter values overall (2*nterms - 1)
# params = [V0, V1, V2, V3, alph1, alph2, alph3]
# idx:      0   1   2   3   4      5      6


def VLmodel(alpha, *params):
    """Models equation representing VL(alpha). Trigonometric series is chosen
    for model equation, number of considered terms is controlled by nterms varibale.
    Four terms proved to be sufficient to capture measured VL(alpha) variation with
    acceptable accuracy."""

    VLparams = params[:nterms]
    alphaparams = params[nterms:]

    VL = VLparams[0]
    for n, (V, alph) in enumerate(zip(VLparams[1:], alphaparams)):
        # n starts from zero, so we must use n+1
        VL += V*np.cos((n+1)*alpha+alph)

    return VL


# Fit the model for VL
params = curve_fit(VLmodel, alphadata, VLdata, p0=np.zeros(2*nterms-1))[0]


def VLfit(alpha):
    """Represents fitted model for VL(alpha). See VLmodel for further details"""
    return VLmodel(alpha, *params)


# Fitteing error evaluation
abs_error = np.abs(VLdata - VLfit(alphadata))
rel_error = abs_error/VLdata


def QLfit(alpha):
    """Computes flow rate value from VLfit, uses functional numeric derivative from
    calculus module and VLfit. Step is controlled with dalpha variable in the script."""

    dVLdt = omega*calculus.dfdx(VLfit, alpha, dalpha)

    return dVLdt


# Interface capturing
rarch = design.infer_arch_radius(rhub, rrim, betarim)
cell = design.ArchImpellerCell(rhub, rrim, l, delta, s, rarch)
shape = (31, 31)

dalpha = np.deg2rad(1)
dt = dalpha/omega

for i, (alph, rref) in enumerate(zip(alphadata, rifdata)):
    print(f"Capturing interface in the cell {i}")

    prior = fields.RotatingField(cell, shape, VLfit(alph-dalpha), rho, omega)
    field = fields.RotatingField(cell, shape, VLfit(alph), rho, omega)

    prior.alpha = alph-dalpha
    prior.t = prior.alpha/omega

    field.alpha = alph
    field.t = field.alpha/omega

    # Manually setting prior velocity field acc. to fitted curves
    # to get proper velocity derivatives
    prior.u = -1/prior.Af * QLfit(alph)
    prior.w = prior.r * prior.u * prior.dphidr

    field.U(prior)
    field.dUdt(prior)
    field.dUdr()
    field.gradP()

    field.capture_inteface(rref)

    xif, yif = transform.rphi_to_xy(field.rif, field.phiif)
    xif, yif = transform.scale(xif, yif, 1e3)
    xif, yif = transform.rotate(xif, yif, alph-np.pi/2)
    # pi/2 due to some differences in coordinate system orientation in lirics and
    # referred paper

    interface = np.array([xif, yif]).T

    fname = "cell"+str(i)+".csv"
    fpath = storage / fname
    np.savetxt(fpath, interface, delimiter=',')


# Fitted curve plot for visual confirmation:

w = 190
h = 100
mm_to_inch = 1/25.4
fig, ax = plt.subplots(figsize=(w*mm_to_inch, h*mm_to_inch))

ax.set_xlabel(r"$\alpha \degree$")
ax.set_ylabel(r"$V^{(L)}$ [L]")
ax.set_xlim((0, 360))
ax.grid(True)
ax.plot([], "ro")
ax.plot([], "C0-")
ax.plot([], "k--")
ax.legend(["data", "fit", "max"])

nmodel = 100
alphamodel = np.linspace(0, 2*np.pi, nmodel)
alphamodel_deg = np.rad2deg(alphamodel)
ax.plot(alphamodel_deg, VLfit(alphamodel)*1e3, "C0")
ax.plot(alphadata_deg, VLdata*1e3, "ro")
ax.plot(alphamodel_deg, np.ones_like(alphamodel_deg)*cell.V*1e3, "k--")

xtxt = 110
ytxt1 = 0.26
ytxt2 = ytxt1 - 0.03
ax.text(xtxt, ytxt1, f"max. abs. error : {np.max(abs_error)*1e6:.2f} [mL]",
        bbox={"facecolor": "w", "edgecolor": "none"})
ax.text(xtxt, ytxt2, f"max. rel. error : {np.max(rel_error)*100:.2f} [%]",
        bbox={"facecolor": "w", "edgecolor": "none"})

plt.show()
