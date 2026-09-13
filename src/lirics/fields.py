from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray

import numpy as np
from scipy.constants import g
from scipy.interpolate import LinearNDInterpolator as linearNDintp
from scipy.optimize import fsolve, newton

from lirics import calculus
from lirics import grid
from lirics.design import ImpellerCell, Housing

import matplotlib.pyplot as plt
from lirics import transform


type ScipyInterpolator = Callable[[tuple[NDArray, NDArray]], NDArray]


BACK_PHI_CORRECTION = np.deg2rad(0.5)

# Container access keys
HUB = 0
RIM = -1
BACK = 0
FRONT = -1
MID = 1
ANY = -1


class RotatingField:
    """Class for representation of the flow filed in the impeller cell of the
    liquid ring machine.

    Field class handles flow field-related data management and is responsible for
    cell flow resolution for the next spatio-temporal step of the modelling procedure
    (for the next moment in time when cell moves to the next angular position).

    Attributes of the calss hold:
        > current time : t
        > current angular position of the domain : alpha
        > rotational velocity of the domain : omega
        > overall flow domain volume : V
        > volume of the domain occupied with liquid : VL
        > volume of the domain occupied with vapor : VP
        > vapor pressure in the domain : pV
        > vapor temperature in the domain : TV
        > fluid density : rho
        > grid points : (r,phi)
        > velocity field : (u,w)
        > velocity field temporal derivatives : (dudt,dwdt)
        > velocity field spatial derivatives : (dudr,dwdr)
        > pressure "gradient" : (dpdr,dpdphi)

    """

    def __init__(
            self,
            cell: ImpellerCell,
            shape: tuple[int, int],
            VL: float,
            rho: float,
            omega: float
    ) -> None:

        self._cell = cell

        self.V = cell.V

        # Vapor parameters?
        self.pV = np.nan
        self.VV = np.nan
        self.TV = np.nan

        self.VL = VL
        self.actualVL = np.nan

        # Considered flow is incompressible, so density "field" is constant
        self.rho = rho

        # Field exists in time and space
        self.omega = omega
        self.alpha = 0
        self.t = 0
        self.r, self.phi = grid.generate(cell, shape)

        self.Af = cell.Af(self.r)
        self.dphidr = calculus.dydx(self.phi, self.r)

        self.u = np.zeros_like(self.r)
        self.w = np.zeros_like(self.r)

        self.dudr = np.zeros_like(self.r)
        self.dwdr = np.zeros_like(self.r)

        self.dudt = np.zeros_like(self.r)
        self.dwdt = np.zeros_like(self.r)

        self.dpdr = np.zeros_like(self.r)
        self.dpdphi = np.zeros_like(self.r)

        self.dprim = np.zeros_like(self.phi[RIM])

        # Attributes to hold interface points
        self.rif = np.zeros_like(self.phi[RIM])
        self.phiif = np.zeros_like(self.phi[RIM])

        # phi correction (necessary for surface capturing)
        self.phicorr = np.zeros_like(self.phi[RIM])
        self.phicorr[0] = BACK_PHI_CORRECTION

    def U(self, prior: RotatingField):
        """Calculates velocity field components with volumetric flow rate computed
        from backward derivative approximation for liquid volume and cell midline
        tangency assumption."""

        dVL = self.VL - prior.VL
        dt = self.t - prior.t

        self.u = - 1 / self.Af * dVL / dt
        self.w = self.u * self.r * self.dphidr

    def dUdt(self, prior: RotatingField):
        """Calclates temporal derivative of the velocity field with backward approximation
        of the dreivative and prior spatio-temporal field."""

        dt = self.t - prior.t
        self.dudt = (self.u - prior.u)/dt
        self.dwdt = (self.w - prior.w)/dt

    def dUdr(self):
        """Calculates spatial derivatives of the velocity field numerically
        on the grid (r,phi)."""

        self.dudr = calculus.dydx(self.u, self.r)
        self.dwdr = calculus.dydx(self.w, self.r)

    def gradP(self):
        """Calculates pressure "gradient" from the governing equation for incompressibel,
        inviscid fluid flow in the uniformly rotating frame of reference."""

        omega_t = self.omega * self.t

        self.dpdr = self.rho * (
            g * np.cos(omega_t + self.phi) -
            (
                self.dudt
                + self.u * self.dudr
                - self.w**2 / self.r
                - 2 * self.w * self.omega
                - self.omega**2 * self.r
            )
        )

        self.dpdphi = - self.rho * self.r * (
            g * np.sin(omega_t + self.phi) +
            (
                self.dwdt
                + self.u*self.dwdr
                + self.u * self.w / self.r
                + 2 * self.u * self.omega
            )
        )

    def capture_inteface(self, rref):
        """Captures interface points in the cell for given refrence radius.

        Capturing is formulated as a search of (r,phi) points where pressure
        difference relative to the refernce point turns into zero along
        reference-to-rim-and-back paths. Algorithm solves multiple rootfinding
        problems for each angular shift relative to the midline on the grid."""

        cell = self._cell

        dpdr_intp = linearNDintp(
            (self.r.ravel(), self.phi.ravel()), self.dpdr.ravel())
        dpdphi_intp = linearNDintp(
            (self.r.ravel(), self.phi.ravel()), self.dpdphi.ravel())
        gradPintp = [dpdr_intp, dpdphi_intp]

        up = grid.pave_radial_path(
            cell,
            start=(rref, cell.phi(rref)),
            stop=(cell.rrim, cell.phi(cell.rrim)))
        dpup = pathtrapz(up, gradPintp)

        for i, phi in enumerate(self.phi[RIM]):

            side = grid.pave_angular_path(
                start=(cell.rrim, cell.phi(cell.rrim)),
                stop=(cell.rrim, phi))
            dpside = pathtrapz(side, gradPintp)

            self.dprim[i] = dpup + dpside

        for i, phi in enumerate(self.phi[RIM]+self.phicorr):

            dphi = phi-cell.phi(cell.rrim)
            self.rif[i] = newton(
                lambda rdown:
                    self.dprim[i]
                    + pathtrapz(
                        grid.pave_radial_path(
                            cell,
                            start=(cell.rrim, phi),
                            stop=(rdown, cell.phi(rdown)+dphi)),
                        gradPintp),
                0.5*(cell.rrim+cell.rhub))

            # phi corrector is applied because otherwise interpolated curve
            # points fall outside of the domain which causes scipy interpolator
            # to return nan
            self.phiif[i] = cell.phi(self.rif[i]) + dphi - self.phicorr[i]

    def evalvof(self):
        """Compute interface-based volume of fluids in the domain of the field.
        Computation is based on the Green-Gauss are for arbitrary polygon.
        Interface, frontal cell line, rim arch and back cell line are assembeled
        into unified looped path over which appropriate integration is preformed.

        For further details refer to areaGreenGauss in calculus module, Green theorem,
        Gauss area formula (also known as shoelaces formula)."""

        cell = self._cell

        interface = self.rif, self.phiif

        # black magic with array filtering ahead, hold on to your hats,
        # ladies and gentlemen
        frontfilter = (self.r[:, FRONT] > self.rif[FRONT])
        rfront = [self.rif[FRONT], *self.r[frontfilter, FRONT]]
        phifront = [self.phiif[FRONT], *self.phi[frontfilter, FRONT]]
        frontline = rfront[:-1], phifront[:-1]

        rimarch = self.r[RIM, ::-1], self.phi[RIM, ::-1]

        backfilter = (self.r[:, BACK] > self.rif[BACK])
        rback = [*self.r[backfilter, BACK][::-1], self.rif[BACK]]
        phiback = [*self.phi[backfilter, BACK][::-1], self.phiif[BACK]]
        backline = rback, phiback
        # Last point duplication is intentional, do not touch!

        loop = np.hstack((interface, frontline, rimarch, backline))

        self.actualVL = abs(
            cell.l * cell.avmu * calculus.areaGreenGauss(transform.rphi_to_xy(*loop)))

        return loop  # return looped path for debugging and test purposes

    def errvof(self, rref):
        """Aids in flow-field resolution in the cell. Captures interface location for
        given rref and then evaluates volume of fluid in the cell based on the location
        of the interface.

        Used in solution algorithm as a function for rootfinding with Newton method."""

        self.capture_inteface(rref)
        self.evalvof()

        return self.V-self.actualVL

    def solve(
            self,
            prior: RotatingField,
    ):
        """Implements solution algorithm for the flow field in the cell of the
        liquid ring machine. Sets new (guessed) value of liquid volume in the cell
        VLnew  and new value for time t. Uses prior-state field for temporal derivative.

        Flow field is resolved by means of solving two rootfinding problems:

        > First problem corresponds to interface capturing for given reference point
          on the midline of the cell. This problem actually comprises of multiple
          rootfinding problems, each searching for location where pressure difference
          turns zero.
            For further details on this part of algorithm refer to capture_interface

        > Second problem correspond to the search of correct rref value which will,
          in fact, ensure that computed interface corresponds to given VLnew value.

        Only resolves flow field in the cell of the liquid ring machine.
        For full-featured quasi-2D modelling this must be coupled with stationary-field
        solver to formulate mass-balance residual-based procedure which will ensure
        correct VLnew for the cell."""

        self.U(prior)
        self.dUdr()
        self.dUdt(prior)
        self.gradP()

        # Using newton optimizer should cut it, calls ro errvof lead to
        # calls to capture_interface (re-calc. interface location) and
        # calls to evalvof() (re-calc. actaual interface-based vof).
        # So last call during which convergence is achieved we should
        # get proper interface location
        rguesse = 0.5*(self._cell.rhub+self._cell.rrim)
        newton(self.errvof, rguesse)


class StationaryField(ABC):

    # NOTE : maybe a usefull idea
    #        we could precompute all points here with known step,
    #        I wonder what prosprects this framework would open

    # NOTE : insight
    #        Cubic equation in average velocity is possible to derive,
    #        but not in the form I was expecting before
    #
    #        lamP does not automatically follows from lamW, but it is
    #        possible to express lamP*lamW via quantities which
    #        are defined by fixed lamW. This leads to cubic equation
    #        in average W.
    #
    #        I still have some concerns on coupling and consistency
    #        of fixing lamW. For example fixed linear profile makes
    #        no sense for straight cells, because average velocity
    #        is instantly defined by tangent velocity on the rim, which
    #        is constant, so we get constant average velocity across
    #        sections automatically.
    #
    #        Day was long, I guesse I just need to sleep on this,
    #        one happy sunny day I will defend this PhD and rest
    #        calmly

    def __init__(self, cell: ImpellerCell, housing: Housing) -> None:

        # We hold parameters for three radial sections for calculations
        # of fluid flow in stationary domain of liquir-ring machine:
        # middle, back and front, thus [0.0]*3 things

        # Sectors geometrical parameters
        self.dphi = cell.phi(cell.rrim)
        self.alpha = [0.0]*3
        self.r = cell.rhub
        self.delta = cell.delta
        self.R = [0.0]*3  # sectors radial bounds
        self.S = [0.0]*3  # sectors radial span
        self.RH = housing.R

        # Flow parameters
        self.avPSI = [0.0]*3  # potential field contribution
        self.avW = [0.0]*3  # velocity contribution
        self.Prim = [0.0]*3  # rim pressure
        self.avP = [0.0]*3  # pressure contribution
        self.avJ = [0.0]*3  # overall energy flux

    @abstractmethod
    def lamW(self):
        raise

    def lamCF(self):
        pass

    def lamP(self):
        pass

    def lamPsi(self):
        pass

    def lamJ(self):
        pass

    def avlamW3(self):
        pass

    def avlamWP(self):
        pass

    def avlamWPsi(self):
        pass

    def avlamWJ(self):
        pass

    def xi(self):
        pass

    def next(self, rotating_field: RotatingField):

        mid = rotating_field.omega * rotating_field.t + self.dphi
        half = self.delta/2
        self.alpha = [mid - half, mid, mid + half]

        for i, alpha in enumerate(self.alpha):
            self.R[i] = self.RH(alpha)
            self.S[i] = self.R[i] - self.r

    def propagate(self):
        pass

    def solve(
            self,
            rotating_field: RotatingField,
            prior_field: StationaryField):
        pass


# Some fresh ideas further


class LinearStationaryFiled(StationaryField):
    pass


class QuadraticStationaryField(StationaryField):
    pass


def pathinterp(
        path: tuple[np.ndarray, np.ndarray],
        intps: list[ScipyInterpolator]):
    """Convenience field-on-path interpolator. Handles generic 2D field interpolation
    for interface reconstruction. Accepts desired path for interpolation,
    field-to-be interpolated and interpolator as inputs.

    Field is considered to be vector field, thus each entry represenst component of
    such field. Components themselves are scalar fields. Function performs interpolation
    of each component of vector field over provided path and returns list with each entry
    corresponding to interpolation of components.

    Mostly used for pressure "gradient" field interpolation.
    """

    f_interp = []
    for i in intps:
        f_interp.append(i(path))

    return tuple(f_interp)


def pathtrapz(path: tuple[np.ndarray, np.ndarray], intps: list[ScipyInterpolator]):
    """Convinience function for line integral along path with interpolated field values.
    Uses pathinterp, so pathinterp restrictions and features must be considered.

    Mostly used for pressure "gradient" field integration."""

    return calculus.linetrapz(path, pathinterp(path, intps))
