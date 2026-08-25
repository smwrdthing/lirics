from __future__ import annotations
from numpy.typing import NDArray
import numpy as np

from typing import Callable


type Numeric = float | NDArray


def function_derivative(func: Callable, x: Numeric, dx: Numeric) -> Numeric:
    return (func(x+dx)-func(x-dx))/(2*dx)


def tabular_derivative(y: NDArray, x: NDArray) -> NDArray:

    # For derivative along X pass field F and X grids as usual, for derivative
    # along Y transpose inputs and transpose results back because this works with rows

    dydx = np.zeros_like(x)

    dydx[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    dydx[0] = 1/2*(-3*y[0]+4*y[1]-y[2])/(x[1]-x[0])
    dydx[-1] = -1/2*(-3*y[-1]+4*y[-2]-y[-3])/(x[-1]-x[-2])

    return dydx


def time_derivative(y: Numeric, prior_y: Numeric, time_step: float) -> Numeric:
    return (y-prior_y)/time_step


def tabular_line_integral(path, components):
    return np.sum(np.trapezoid(components, path))
