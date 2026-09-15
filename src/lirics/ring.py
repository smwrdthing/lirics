from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.polynomial.polynomial import polyroots
from scipy.constants import g

from lirics.design import ImpellerCell, Housing, CylindricalHousing


# Access indices
START = 0
END = 1


# NOTE : currently operational parameters and port edges are passed
#        in common dictionary, probably not the most elegant approach,
#        but it is fast. I'll fix it later, when I'll have more time for
#        this.
type PfleidereModelParamKey = Literal[
    "pVsuc", "rhoL", "a", "omega", "alphamax", "alphadis"]
type PfleidereModelParams = dict[PfleidereModelParamKey, float]


class GeneralizedPfleiderer(ABC):

    def __init__(self, cell: ImpellerCell, housing: Housing,
                 params: PfleidereModelParams) -> None:

        self._cell = cell
        self._housing = housing
        self._params = params

        Hsuc = params["pVsuc"]/params["rhoL"]/g

        # Computing necessary coefficients and parameters
        s = cell.s
        rrim = cell.rrim
        rhub = cell.rhub
        delta = cell.delta
        l = cell.l

        L = housing.L

        urim = params["omega"]*rrim

        self.nu = rhub/rrim
        self.psi = 1.0
        self.zeta = L/l
        self.epsilon = urim**2 / (2*g*Hsuc)
        self.alpha = params["a"]/rrim
        self.mu = 1 - s / (delta/2 * rrim**2 * (1-self.nu**2))

        self.sigmad = self.sigma(params["alphadis"])

        # Checking limits of the model
        sigmamax = 2/3 * (self.epsilon*self.psi**2 + 1)
        epsilonmin = 1/self.psi**2 * (3/2*self.sigmad - 1)

        if self.epsilon < epsilonmin:

            msg = (
                "Value of velocity coefficient falls outside of the model limits.\n" +
                f"Min. acceptable value is {epsilonmin}, but model parameters " +
                f"produce {self.epsilon}.\nConsider increasing suction pressure or " +
                "increasing circumferential velocity.")

            raise ValueError(msg)

        if self.sigmad > sigmamax:

            msg = (
                "Value of overall pressure ratio falls outside of the model limits.\n" +
                f"Max. acceptable value is {sigmamax}, but model parameters " +
                f"produce {self.sigmad}.\nConsider increasing velocity coefficient or " +
                "psi coefficient of the vane.")

            raise ValueError(msg)

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
            self.epsilon/self.A(alpha)**2,
            zeros,
            -(self.epsilon*self.psi**2 + 1)*ones,
            ones,
        ]).T

        sigma = []
        for c in coeffs:
            r = polyroots(c)
            r = np.real(r[np.isreal(r)])
            sigma.append(np.min(r))
        sigma = np.array(sigma).flatten()

        return sigma

    def rifsuc(self, alpha):
        """Returns radius-vector of the interface for given rotational angle in the
        suction region.

        Equation for rifsuc(alpha) is obtained with assumption of watertight cell and
        constant velocity and pressure in the suction region."""

        rrim = self._cell.rrim

        rifs = rrim * np.sqrt(
            2*self.zeta*self.psi/self.mu * (self.S(alpha)-self.S(0))/rrim
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

    def rif(self, alpha):
        """Returns radius-vector of the interface for given rotational angle.
        Resolves regions (suction, compression, discharge) internally, makes call to
        respective methods to compute radius-vectors for each region."""

        alphamax = self._params["alphamax"]
        alphadis = self._params["alphadis"]
        alphamid = alphamax/2

        suc = (alpha >= 0) * (alpha < alphamid)
        com = (alpha >= alphamid) * (alpha < alphadis)
        dis = (alpha >= alphadis) * (alpha <= alphamax)

        rif = np.concatenate(
            (self.rifsuc(alpha[suc]),
             self.rifcom(alpha[com]),
             self.rifdis(alpha[dis]))
        )

        return rif


class ClassicPfleiderer(GeneralizedPfleiderer):

    def __init__(self, cell: ImpellerCell, housing: CylindricalHousing,
                 params: PfleidereModelParams) -> None:

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


class ModifiedPfleiderer(GeneralizedPfleiderer):

    def R(self, alpha):
        return self._housing.R(alpha-np.pi)

    def S(self, alpha):
        return self.R(alpha) - self._cell.rrim


# I think implementation could be generalized even further to accept custom
# hydraulic losses model, but let's not rush with it for now,
# maybe if I have enough time...
