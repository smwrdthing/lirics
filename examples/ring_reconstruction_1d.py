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
    delta=np.deg2rad(360/12),
    s=5e-3,
    rarch=infer_arch_radius(rhub, rrim, np.deg2rad(45))
)
housing = CylindricalHousing(
    L := l,
    e := 20e-3,
    Rc=infer_housing_radius(rrim, e, Smin=5e-3)
)

model = ClassicPfleiderer(
    cell, housing,
    {
        "a": 10e-3,
        "alphadis": np.deg2rad(235),
        "alphamax": np.deg2rad(360),
        "omega": np.pi*3000/30,
        "pVsuc": 101_325,
        "rhoL": 1000
    }
)


alpha = np.linspace(0, 2*np.pi, 100)
rif = model.rif(alpha)
cylR = housing.Rc*np.ones_like(alpha)


xif, yif = transform.rphi_to_xy(rif, alpha)
xif, yif = transform.rotate(xif, yif, np.pi/2)
xif, yif = transform.reflect(xif, yif, "y")
xif, yif = transform.scale(xif, yif, 1e3)


rhub = cell.rhub*np.ones_like(alpha)
xhub, yhub = transform.rphi_to_xy(rhub, alpha)
xhub, yhub = transform.scale(xhub, yhub, 1e3)


rrim = cell.rrim*np.ones_like(alpha)
xrim, yrim = transform.rphi_to_xy(rrim, alpha)
xrim, yrim = transform.scale(xrim, yrim, 1e3)


xcyl, ycyl = transform.rphi_to_xy(housing.R(alpha), alpha)
xcyl, ycyl = transform.rotate(xcyl, ycyl, -np.pi/2)
xcyl, ycyl = transform.reflect(xcyl, ycyl, "y")
xcyl, ycyl = transform.scale(xcyl, ycyl, 1e3)


fig, ax = plt.subplots()
ax.set_title("Reconstruction of the interface in the liquid ring machine \n" +
             "with 1D model of prof. Pfleiderer")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")

ax.plot(xhub, yhub, "0.75", linewidth=1)
ax.plot(xrim, yrim, "0.75", linewidth=1)
ax.plot(xcyl, ycyl, "k", linewidth=1)
ax.plot(xif, yif, 'C0--')

ax.grid(True)

plt.show()

# NOTE : issues are in the compression region, take a closer look at the equatoin
#        for pressure ratio, for some reasons it's solution returns negative values
#
#        And some parts really could be implemented much better, of course.
#        I'll try to improve it if/when I'll have time. Basis is solid though
