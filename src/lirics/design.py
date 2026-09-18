from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from lirics import transform
from lirics import calculus


DR = 1e-3  # default step for functional numeric derivative


@dataclass
class ImpellerCell(ABC):
    """Base class for representation of the liquid ring machine cell.

    Attributes hold main dimensions:
        > hub radius : rhub
        > rim radius : rrim
        > length : l
        > angular width : delta
        > vane thickness : s
        > total area : A
        > total volume : V

    Methods encapsulate functions performing calculations related to the cell geometry.

    Specific cell description should be implemeted by inheriting base class and
    implemeting midline equation method phi(r) in dedicated method."""

    rhub: float
    rrim: float
    l: float
    delta: float
    s: float

    A: float = field(init=False)
    V: float = field(init=False)
    avmu: NDArray | float = field(init=False)

    def __post_init__(self):

        n_vol_integration = round((self.rrim-self.rhub)/DR)
        r = np.linspace(self.rhub, self.rrim, n_vol_integration)

        # Vol. computation
        self.V = np.trapezoid(self.Af(r), r)
        self.A = self.V/self.l

        # Average cluttering coefficient for further volume computations
        self.avmu = 1/(self.rrim-self.rhub) * np.trapezoid(self.mu(r), r)

    @abstractmethod
    def phi(self, r) -> np.ndarray:
        """Represents cell midline equation in the form of angular coordinate
        vs radial coordinate phi(r). Returns numeric value of phi for given radial
        coordinate

        Must be overriden by specific cell definition by means of inheritance."""
        raise

    def theta(self, r, dr=DR) -> np.ndarray:
        """Computes and returns numeric value of anlge of the cell midline tangent line.
        Not common in literature. Computations are preformed in the usual manner for
        general parametric curve with radial coordinates being the parameter"""

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
        beta2. Returns numeric value of this angle for given radial coordinate"""

        # From geometric consideration beta is basically phi_mid + pi/2 - theta
        # where theta is an angle of the tangent line to cell midline

        beta = self.phi(r) + np.pi/2 - self.theta(r)

        return beta

    def midline_length(self, r: np.ndarray):
        """Computes and returns length of cell qmidline. Computation is performed
        as follows:

            > radial coordinates array r is used to compute corresponding cell
              midline angles array phi
            > (r,phi) is converted into (x,y)
            > dxdr and dydr are computed numerically with obtained arrays
            > dxdr and dydr are used to compute standard integral for parametric
              curve length considering r as a parameter

        Integration is carried with trapezoid rule, for further details refer to
        numpy.trapezoid function."""

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

    # NOTE this dr passing gets annoying really fast, probably should find workaround
    def mu(self, r, dr=DR):
        """Computes and returns cell cluttering coefficient accounting for the finitness
        of vanes thickness inside the cell."""

        mu = 1 - self.s / self.delta / \
            (r*np.sin(self.beta(r)))

        return mu

    def Af(self, r):
        """Computes and returns cross-area of the radial flow in the cell for given
        radial coordinate r."""

        Af = self.l * self.delta * r * self.mu(r)

        return Af


@dataclass
class StraightImpellerCell(ImpellerCell):
    """Class representing cell with straight midline. This class re-implemets midline
    equation so that it always returns array of zeros compliant with the shape of the
    provided radial coordinates array r."""

    def phi(self, r):
        return np.zeros_like(r)


@dataclass
class ArchImpellerCell(ImpellerCell):
    """Class representing cell with arch-shaped midline. Adds arch radius attribute rarch
    and arch center radius-vectro attribut rcenter used in the phi(r).

    rcenter is computed internally and should not be modified after object creation."""

    rarch: float

    def __post_init__(self):
        self.rcenter = np.sqrt(self.rhub**2+self.rarch**2)
        super().__post_init__()

    def phi(self, r):

        phi = (
            np.arcsin(self.rarch/self.rcenter)
            - np.arccos((r**2 + self.rcenter**2 - self.rarch**2) /
                        (2*r*self.rcenter)))

        return phi


@dataclass
class Housing(ABC):
    """Base class for representation of the liquid ring machine housing.

    Attributes hold main dimensinos:
        > length : L

    Methods encapsulate some geometrical functions and calculations related to
    the housing.

    Specific housings should be implemeted by inheriting base class and implementing
    R(alpha) function for the profile."""

    L: float

    @abstractmethod
    def R(self, alpha):
        """Represents housing profile equation. Returns radius-vector length for given
        rotational angle.Rotational angle is seeded from x-axis.

        Chosen generic coordinate system is aligned as follows:
            x-axis passes throguh location corresponding to max. vapor volume in the cell
            y-axis is oritned from intake port to discharge port

        Coordinate system alignment must be considered for proper implementation of
        R-function for specific housings."""

        raise


@dataclass
class CylindricalHousing(Housing):
    """Class representing cylindrical housing of the single-acting liquid ring machine.
    Adds excentricity and cylinder radius attributes e and Rc used in the R(alpha)."""

    e: float
    Rc: float

    def R(self, alpha):
        """Represents cylindrical housing equation. Derivation is based on triangle
        formed by radius-vector, cylindrincal housing radius and straight line connecting
        centers of housing and impeller. Relationship between sides of such triangle
        leads to quadratic equatino in R. Solution of this equation and selection of
        physically sensibel root leads to implemented equation:"""

        R = self.e*np.cos(alpha) + np.sqrt(
            self.Rc**2 - self.e**2*(1-np.cos(alpha)**2))

        return R


@dataclass
class EllipticHousing(Housing):
    """Class representing elliptic housing of the double-acting liquid ring machine.
    Adds additional attributes A and B for major and minor semi-axes used in R(alpha)."""

    A: float
    B: float

    def R(self, alpha):
        """Represents elliptic housing equation. Derived from canonical equation of the
        ellipse by substitution of Cartesian-to-polar transformation equations, such
        substitution allows then to express R(alpha)."""

        R = np.sqrt(
            1 / (np.cos(alpha)**2/self.A**2 +
                 np.sin(alpha)**2/self.B**2))

        return R


# Convenience functions ahead

def infer_arch_radius(rhub, rrim, betarim):

    rarch = (rrim**2 - rhub**2) /\
        (2*rrim*np.cos(betarim))

    return rarch


def infer_housing_radius(rrim: float, e: float, Smin: float = 0):

    Rc = Smin + rrim + e

    return Rc
