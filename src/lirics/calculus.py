from __future__ import annotations
from numpy.typing import NDArray
import numpy as np

from typing import Callable


type Numeric = float | NDArray


def dfdx(func: Callable, x: Numeric, dx: Numeric) -> Numeric:
    """Represents functional numeric derivative. Accepts function of single argument
    as a variable and computes derivative numerically with step dx over all x values
    by means of central differencing."""

    return (func(x+dx)-func(x-dx))/(2*dx)


def dydx(y: NDArray, x: NDArray) -> NDArray:
    """Numerical derivative over (y,x) arrays. Derivative dy/dx is computed with the
    second order scheme, for internal array points central differencing utillized,
    boundary points in the arrays are treated separately to maintin second oreder
    accuracy when central differencing is not applicable.

    For further details on boundary points derivatives computation refer to:
        "Computational Fluid Dynamics : The Basics With Applications"
                                                   by J. D. Anderson

    Handles numerical partial derivatives over 2D filed F(X,Y) as well.
        > For partial derivative of F wrt X pass F as y and X as x:
            dFdx = dydx(F,X)

        > For partial derivative of F wrt Y pass F.T as y and Y.T as x AND transpose
          returned array:
            dFdy = dydx(F.T,Y.T).T

    Transposition is required in the second case because internally "x" is assumed
    to cahnge along rows."""

    dydx = np.zeros_like(x)

    dydx[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    dydx[0] = 1/2*(-3*y[0]+4*y[1]-y[2])/(x[1]-x[0])
    dydx[-1] = -1/2*(-3*y[-1]+4*y[-2]-y[-3])/(x[-1]-x[-2])

    return dydx


def linetrapz(path, components):
    """Calculates line integral of vector field with trapezoid rule. Hadles
    integrals of the expressions in the following form:

        f0(x0,x1,...,xn)*dx0 + f1(x0,x1,...,xn)*dx1 + ... fn(x0,x1,...,xn)*dxn

    Supports multidimensional vector fields. Theoretically. Though practical application
    in this codebase is limited to 2D case only.
    """

    # For integral(P(x,y)*dx + Q(x,y)*dy + ...) calculation like this is reduced
    # to simple call to np.sum over arrays of integrals computed with
    # np.trapezoid for each component (P,Q,...) and corresponding path (x,y,...)

    return np.sum(np.trapezoid(components, path))
