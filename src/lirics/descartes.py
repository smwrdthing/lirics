from typing import Literal
import numpy as np
from numpy import float64
from numpy.typing import NDArray

"""2D coordinates manipulation and transformation."""


def cartesian_to_polar(x: NDArray[float64], y: NDArray[float64]
                       ) -> tuple[NDArray[float64], NDArray[float64]]:
    return np.sqrt(x**2+y**2), np.arctan2(y, x)


def polar_to_cartesian(r: NDArray[float64], phi: NDArray[float64]
                       ) -> tuple[NDArray[float64], NDArray[float64]]:
    return r*np.cos(phi), r*np.sin(phi)


def translate(
        x: NDArray[float64], y: NDArray[float64],
        shift_x: float, shift_y: float) -> tuple[NDArray[float64], NDArray[float64]]:
    return x+shift_x, y+shift_y


def rotate(x: NDArray[float64], y: NDArray[float64], theta: float
           ) -> tuple[NDArray[float64], NDArray[float64]]:

    x_rotated = x*np.cos(theta) - y*np.sin(theta)
    y_rotated = x*np.sin(theta) + y*np.cos(theta)

    return x_rotated, y_rotated


def rotate_polar(r: NDArray[float64], phi: NDArray[float64], theta: float
                 ) -> tuple[NDArray[float64], NDArray[float64]]:
    return r, phi+theta


def mirrox(x: NDArray[float64], y: NDArray[float64], axis: Literal["x", "y", "xy"] = "x"
           ) -> tuple[NDArray[float64], NDArray[float64]]:

    if axis not in ("x", "y", "xy"):
        raise ValueError("Provide one of valid axis to mirror")

    if axis == "x":
        y = -y
    elif axis == "y":
        x = -x
    elif axis == "xy":
        x = -x
        y = -y

    return x, y
