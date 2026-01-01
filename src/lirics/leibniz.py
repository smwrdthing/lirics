from __future__ import annotations
from numpy import float64
from numpy.typing import NDArray

import numpy as np


def derivative(y: NDArray[float64], x: NDArray[float64]) -> NDArray[float64]:
    """Computes derivative with tabulated function values y corresponding
    to points x with second order of error."""
    dydx = np.zeros_like(x)

    dydx[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    dydx[0] = 1/2*(-3*y[0]+4*y[1]-y[2])/(x[1]-x[0])
    dydx[-1] = -1/2*(-3*y[-1]+4*y[-2]-y[-3])/(x[-1]-x[-2])

    return dydx


def time_derivative(y, prior_y, time_step):
    return (y-prior_y)/time_step
