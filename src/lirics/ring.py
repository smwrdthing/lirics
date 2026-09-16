from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.polynomial.polynomial import polyroots
from scipy.constants import g
from scipy.optimize import newton

from lirics.design import ImpellerCell, Housing, CylindricalHousing
from lirics import transform


NUM_REGION_POINTS = 50


type PGRParamKey = Literal["pVsuc", "rhoL", "omega", "alphamax", "alphadis"]
type PGRParams = dict[PGRParamKey, float]


class PGRGenralized(ABC):
    """Represents generalize Pfleiderer-Golovincov-Rumyancev 1D model for interface
    reconstruction in liquid ring machine."""

    def __init__(self, cell: ImpellerCell, housing: Housing,
                 params: PGRParams) -> None:

        self._cell = cell
        self._housing = housing
        self._params = params

        self.Hsuc = params["pVsuc"]/(params["rhoL"]*g)

        # Computing necessary coefficients and parameters
        s = cell.s
        rrim = cell.rrim
        rhub = cell.rhub
        delta = cell.delta
        l = cell.l
        L = housing.L

        self.urim = params["omega"]*rrim

        self.nu = rhub/rrim
        self.psi = 1.0
        self.zeta = L/l
        self.epsilon = self.urim**2 / (2*g*self.Hsuc)
        self.mu = 1 - s / (delta/2 * rrim**2 * (1-self.nu**2))

        self.a = rrim - self.rifsuc(params["alphamax"]/2)
        if self.a < 0:
            msg = "Vane tip is not submerged into liquid surface for given model setup."
            raise ValueError(msg)
        self.alpha = self.a/rrim

        # Checking limits of the model
        self.sigmamax = 2/3 * (self.epsilon * self.psi**2 + 1)

        alphadis_max = newton(
            lambda alpha:
            self.sigmamax**3 * self.A(alpha)**2
            - (self.epsilon*self.psi**2+1)*self.sigmamax**2 * self.A(alpha)**2
            + self.epsilon,
            3/4*params["alphamax"])

        if params["alphadis"] > alphadis_max:

            alpha_given = np.rad2deg(params["alphadis"])
            alpha_possible = np.rad2deg(alphadis_max)

            msg = "Value of discharge angle exceeds limits of the model: "\
                f"current discharge angle is {alpha_given:.2f} (deg), "\
                f"though with given parameters only ~{alpha_possible:.2f} (deg) "\
                "is permitted within model application range."

            raise ValueError(msg)

        self.sigmad = self.sigma(params["alphadis"])

    @abstractmethod
    def S(self, alpha):
        """Returns thickness of the out-of-impeller region in the liquid ring machine.
        Diifferent implement their own rule for S(alpha). There is only one restriction -
        S(0) must return smallest possible value."""

        raise

    def lowS(self, alpha):
        """Returns thickness of the out-of-impeller region for lowered housing profile.
        Lowered profile is, by definition, a housing profile for which minimal thickness
        of the out-of-impeller region is zero.

        Minimal thickness for lowered profile is zero by definition, so
        generalized model implementation is:

                                lowS(alpha) = S(alpha) - S(0)*.

        * - refer to S(alpha) method for further explanation

        Generally, lowered profile can be constructed in various ways for arbitrary
        profile of the housing. To apply special rule this method must be overriden
        with corresponding implementation."""

        return self.S(alpha) - self.S(0)

    def A(self, alpha):
        """Returns dimensionless coefficient in the equation for pressure ratio vs
        rotational angle dependency."""

        k = 2*self.zeta/self.mu * 1/((1-self.alpha)**2 - self.nu**2)
        rrim = self._cell.rrim

        A = k * self.lowS(alpha)/rrim

        return A

    def sigma(self, alpha):
        """Returns pressure ratio for given rotational angle. Pressure ratio is obtained
        as solution of cubic equation for pressure ratio. Equation is solved numerically
        with polyroots() function from numpy.polynomial.polynomial module, polyroots()
        return is porcessed appropriately to select proper root out of the three.

        polyroots() handles one polynomial at a tmie, so array input makes native python
        loop unavoidable, this could hinder performance for large input arrays.

        Equation for pressure ratio is obtained with assumptoin of watertight cell and
        isothermal compression."""

        alpha = np.atleast_1d(alpha)

        ones = np.ones_like(alpha)
        zeros = np.zeros_like(alpha)
        coeffs = np.array([  # ascending power order!
            self.epsilon*ones,
            zeros,
            - self.A(alpha)**2 * (self.epsilon*self.psi**2 + 1),
            self.A(alpha)**2,
        ]).T

        sigma = []
        for c in coeffs:
            r = polyroots(c)
            sigma.append(r[1])
        sigma = np.array(sigma).flatten()

        return sigma

    def alphasuc(self, n: int = NUM_REGION_POINTS):
        """Provides n angular coordinates belonging to suction region."""
        return np.linspace(0, self._params["alphamax"]/2, n)

    def alphacom(self, n: int = NUM_REGION_POINTS):
        """Provides n angular coordinates belonging to compression region."""
        return np.linspace(self._params["alphamax"]/2, self._params["alphadis"], n)

    def alphadis(self, n: int = NUM_REGION_POINTS):
        """Provides n angular coordinates belonging to discharge region."""
        return np.linspace(self._params["alphadis"], self._params["alphamax"], n)

    def rifsuc(self, alpha):
        """Returns radius-vector of the interface for given rotational angle in the
        suction region.

        Equation for rifsuc(alpha) is obtained with assumption of watertight cell and
        constant velocity and pressure in the suction region."""

        rrim = self._cell.rrim

        rifs = rrim * np.sqrt(
            2*self.zeta/self.mu * self.lowS(alpha)/rrim * self.psi
            + self.nu**2)

        return rifs

    def rifcom(self, alpha):
        """Returns radius-vector of the interface for given rotational angle in the
        compression region.

        Resolves pressure ratio on the fly by calling sigma(alpha)."""

        rifc = self._cell.rrim * np.sqrt(
            ((1-self.alpha)**2 - self.nu**2)/self.sigma(alpha)
            + self.nu**2)

        return rifc

    def rifdis(self, alpha):
        """Returns radius-vector of the interface for given rotational angle in the
        discharge region.

        Equation for rifdis(alpha) is obtained with assumption of watertight cell and
        constant velocity and pressure in the discharge region. Also derivation takes
        advantage of lowered profile concept."""

        rrim = self._cell.rrim

        rifd = rrim * np.sqrt(
            2*self.zeta/self.mu * self.lowS(alpha)/rrim
            * np.sqrt(self.psi**2 - (self.sigmad - 1)/self.epsilon)
            + self.nu**2)

        return rifd

    def interfaceRPHI(self, n: int = NUM_REGION_POINTS):
        """Provides polar interface coordinates (r, alpha). Uses coordinate providers
        and profile methods under the hood."""

        alphasuc = self.alphasuc(n)
        alphacom = self.alphacom(n+1)[1:]
        alphadis = self.alphadis(n+1)[1:]

        alpha = np.concatenate((
            alphasuc,
            alphacom,
            alphadis))

        rif = np.concatenate((
            self.rifsuc(alphasuc),
            self.rifcom(alphacom),
            self.rifdis(alphadis)))

        return rif, alpha

    def interfaceXY(self, n: int = NUM_REGION_POINTS):
        """Provides cartesion interface coordinates (x, y). Uses interfaceRPHI and
        transform.rphi_to_xy under the hood."""

        return transform.rphi_to_xy(*self.interfaceRPHI(n))


class PGRClassic(PGRGenralized):
    """Represents classic Pfleiderer-Golovincov-Rumyancev 1D model for interface
    reconstruction in liquid ring machine.

    Supports only single acting machines with cylindrical housings."""

    def __init__(self, cell: ImpellerCell, housing: CylindricalHousing,
                 params: PGRParams) -> None:

        if not (isinstance(housing, CylindricalHousing)):
            raise ValueError(
                "Classic Pfleiderer model supports only cylindrical casing of the" +
                "single-acting liquid ring machine.")
        self._housing: CylindricalHousing

        super().__init__(cell, housing, params)

    def housR(self, alpha):
        """Computes radius-vector from the axis of the cylindircal housing
        to the rim of the impeller for given rotational angle. Applied for
        out-of-impeller region thickness computation on the classical model
        of liquid ring machine by prof. Pfleiderer."""

        rrim = self._cell.rrim
        e = self._housing.e

        housR = np.sqrt(rrim**2 + e**2 + 2*e*rrim*np.cos(alpha))

        return housR

    def S(self, alpha):
        return self._housing.Rc - self.housR(alpha)


class PGRModified(PGRGenralized):
    """Represents modified Pfleiderer-Golovincov-Rumyancev 1D model for interface
    reconstruction in liquid ring machine.

    Supprots single and double acting liquid ring machines with arbitrary profile."""

    def R(self, alpha):
        return self._housing.R(alpha-self._params["alphamax"]/2)

    def S(self, alpha):
        return self.R(alpha) - self._cell.rrim


# NOTE : On latest debugging (commits around 16.09.26)
#        well, it works now, but stuff is incredibly messy, would not hurt to refine
#        implementation, allow different strategies of model application, add convenience
#        functions to infer some parameters when others are fixed etc etc.
#
# NOTE : currently operational parameters and port edges are passed
#        in common dictionary, probably not the most elegant approach,
#        but it was fast in terms of implementation. I'll fix it later,
#        when I'll have more time for this.
#
# NOTE : I think implementation could be generalized even further to accept custom
#        hydraulic losses model, but let's not rush with it for now, maybe if I have
#        enough time...
#
# NOTE : checking for max. possible discsharge angle and vanes submersion directly during
#        object creation is not the most gracefull thing either, should encapsulate
#        validation checks in separate ("_private"?) method.
