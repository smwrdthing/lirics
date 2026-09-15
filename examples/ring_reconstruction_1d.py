import numpy as np
import matplotlib.pyplot as plt

from lirics.design import ArchImpellerCell, CylindricalHousing
from lirics.design import infer_arch_radius, infer_housing_radius
from lirics.ring import ClassicPfleiderer

from lirics import transform

cell = ArchImpellerCell(
    rhub := 100e-3,
    rrim := 200e-3,
    l := 2*rhub,
    delta=360/12,
    s=5e-3,
    rarch=infer_arch_radius(rhub, rrim, np.deg2rad(45))
)
housing = CylindricalHousing(
    l,
    e := 10e-3,
    Rc=infer_housing_radius(rrim, e)
)

model = ClassicPfleiderer(
    cell, housing,
    {
        "a": 5e-3,
        "alphadis": np.deg2rad(150),
        "alphamax": np.deg2rad(360),
        "omega": np.pi*1500/30,
        "pVsuc": 101325,
        "rhoL": 1000
    }
)

alpha = np.linspace(0, 2*np.pi)
rif = model.rif(alpha)

xif, yif = transform.rphi_to_xy(rif, alpha)
xif, yif = transform.rotate(xif, yif, np.pi/2)
xif, yif = transform.scale(xif, yif, 1e3)

rhub = cell.rhub*np.ones_like(alpha)
rrim = cell.rrim*np.ones_like(alpha)

xhub, yhub = transform.rphi_to_xy(rhub, alpha)
xhub, yhub = transform.rotate(xhub, yhub, np.pi/2)
xhub, yhub = transform.scale(xhub, yhub, 1e3)

xrim, yrim = transform.rphi_to_xy(rrim, alpha)
xrim, yrim = transform.rotate(xrim, yrim, np.pi/2)
xrim, yrim = transform.scale(xrim, yrim, 1e3)


fig, ax = plt.subplots()
ax.set_title("Reconstruction of the interface in the liquid ring machine \n" +
             "with 1D model of prof. Pfleiderer")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

ax.plot(xhub, yhub, "k-")
ax.plot(xrim, yrim, "k-")
ax.plot(xif, yif, 'b--')

ax.grid(True)
