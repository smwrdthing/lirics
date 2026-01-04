from __future__ import annotations
from typing import Callable, overload
from numpy import float64
from numpy.typing import NDArray

import numpy as np


@overload
def function_derivative(
        func: Callable[[float], float], x: float, dx: float) -> float:
    ...


@overload
def function_derivative(
        func: Callable[[NDArray[float64]], NDArray[float64]],
        x: NDArray[float64], dx: float) -> NDArray[float64]:
    ...


def function_derivative(func, x, dx):
    return (func(x+dx)-func(x-dx))/(2*dx)


def tabular_derivative(y: NDArray[float64], x: NDArray[float64]) -> NDArray[float64]:
    """Computes derivative with tabulated function values y corresponding
    to points x with second order of error."""

    # This is lighter than numpy's gradient, so it runs faster, thus retained.
    #
    # For derivative along X just pass field F and X grids as usual, for derivative
    # along Y transpose inputs and transpose results back
    #
    # (just remember, this works with rows, thus all transpostitions)

    dydx = np.zeros_like(x)

    dydx[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    dydx[0] = 1/2*(-3*y[0]+4*y[1]-y[2])/(x[1]-x[0])
    dydx[-1] = -1/2*(-3*y[-1]+4*y[-2]-y[-3])/(x[-1]-x[-2])

    return dydx


def time_derivative(y, prior_y, time_step):
    return (y-prior_y)/time_step


def tabular_line_integral(path, components):
    return np.sum(np.trapezoid(components, path))
