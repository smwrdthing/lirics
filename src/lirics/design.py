from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lirics import transform
from lirics import calculus


DR = 1e-3  # default differentioation step for functional derivative


@dataclass
class ImpellerCell(ABC):
    '''Base class for representation of the liquid ring machine cell.

    Attributes holds main dimensions:
        > hub radius : rhub
        > rim radius : rrim
        > length : l
        > angular width : delta
        > vane thickness : s
        > total area : A
        > total volume : V

    Methods encapsulate some geometrical functions and calculations related to
    the cell.
    '''

    rhub: float
    rrim: float
    l: float
    delta: float
    s: float

    A: float = field(init=False)
    V: float = field(init=False)

    @abstractmethod
    def phi(self, r) -> np.ndarray:
        """Represents cell midline equation in the form of angular coordinate
        vs radial coordinate phi(r). Returns numeric value of phi for given radial
        coordinate

        Must be overriden by specific cell definition by means of inheritance."""
        raise

    def theta(self, r, dr=DR) -> np.ndarray:
        '''Computes and returns numeric value of anlge of the cell midline tangent line.
        Not common in literature. Computations are preformed in the usual manner for
        general parametric curve with radial coordinates being the parameter'''

        phi = self.phi(r)

        # We need functional derivative for this to work properly with 1D and 2D inputs
        dphidr = calculus.dfdx(self.phi, r, dr)

        # Cartesian coordinate derivatives are writeen in expanded form, using
        # transformation functions is not practical here
        dxdr = np.cos(phi) - r*dphidr*np.sin(phi)
        dydr = np.sin(phi) + r*dphidr*np.cos(phi)

        theta = np.arctan(dydr/dxdr)

        return theta

    def beta(self, r) -> np.ndarray:
        """Represents angle between tangents line of cell midline and circle of given
        raidus. For rim radius this quantity is usually denoted in the literature as
        beta_2. Returns numeric value of this angle for given radial coordinate"""

        # From geometric consideration beta is basically phi_mid + pi/2 - theta
        # where theta is an angle of the tangent line to cell midline

        beta = self.phi(r) + np.pi/2 - self.theta(r)

        return beta

    def midline_length(self, r: np.ndarray):
        """Computes and returns length of cell midline. Computation is
        performed as follows:

            > radial coordinates array r is used to compute corresponding cell
              midline angles array phi
            > (r,phi) is converted into (x,y)
            > dxdr and dydr are computed numerically with obtained arrays
            > dxdr and dydr are used to compute standard integral for parametric
              curve length considering r as a parameter

        Integration is carried with trapezoid rule, for further details refer to
        numpy.trapezoid"""

        # Expecting r as an array here eliminates tedious processing of
        # integration boundaries definition, arrays generation etc etc
        #
        # This way user explicitly provides an array over which we must integrate
        # to obtain length and thus user gains more control over desired accuracy,
        # integration span etc
        #
        # If user wishes to get length as a function of r, then 2D array could be
        # provided, with each column spanning from starting radius to final (var.) radius.
        # NumPy integration with trapezoid supports such cases
        # NOTE : reflect in docstring, ensure that this works properly with tests

        phi = self.phi(r)
        x, y = transform.rphi_to_xy(r, phi)
        dxdr = calculus.dydx(x, r)
        dydr = calculus.dydx(y, r)

        # Computing function to integrate
        f = np.sqrt(dxdr**2 + dydr**2)

        # axis spec. here to treat cases with 2D r array
        l = np.trapezoid(f, r, axis=0)

        return l

    # NOTE thsi dr passing gets annoying really fast, probably should find workaround
    def mu(self, r, dr=DR):
        """Computes and returns cell cluttering coefficient caused by finitness
        of vanes occupying cell space"""

        mu = 1 - self.s / self.delta / \
            np.sin(self.theta(r, dr))

        return mu

    def duct_area(self, r):
        '''Computes and returns duct area (area of the radial flow in the cell) for given
        radial coordinate r'''

        A = self.l * self.delta * r * self.mu(r)

        return A


@dataclass
class Housing(ABC):

    length: float

    @abstractmethod
    def R(self, angle):
        pass


@dataclass
class CylindricalHousing(Housing):

    excentricity: float

    def R(self, angle):
        pass


@dataclass
class EllipticHousing(Housing):

    major_semiaxis: float
    minor_semiaxis: float

    def R(self, angle):
        pass
