from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray

import numpy as np
from scipy.constants import g
from scipy.interpolate import LinearNDInterpolator as linearNDintp
from scipy.optimize import fsolve

from lirics import calculus
from lirics import grid
from lirics.design import ImpellerCell, Housing


type ScipyInterpolator = Callable[[tuple[NDArray, NDArray], NDArray], NDArray]


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
        self.VLinterface = np.nan

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

        # We must solve multiple minimization problems for this to work
        #
        # Algorithm is as follows:
        # 1. Compute pressure diference along midline
        # 2. Compute pressure diference from midline to all angular shifts
        # 3. Compute points downward shifted midlines for which pressure difference is
        #    zero
        #
        # Then write points. Use them to evaluate volume of fluid, solve cell flow for
        # different reference radiuses until new given VL and evaluated volume of fluid
        # match within required tolerance
        #
        # Should restrict this to grid points shifts, general arbitrary interpolation on
        # the rectilinear grid is possible (and is implemented in other branch),
        # but I doubt that it is practical

        cell = self._cell
        gradP = [self.dpdr, self.dpdphi]

        dpdr_intp = linearNDintp(
            self.r.ravel(), self.phi.ravel(), self.dpdr.ravel())
        dpdphi_intp = linearNDintp(
            self.r.ravel(), self.phi.ravel(), self.dpdphi.ravel())
        gradPintp = [dpdr_intp, dpdphi_intp]

        up = grid.pave_radial_path(
            cell,
            start=(rref, cell.phi(rref)),
            stop=(cell.rrim, cell.phi(cell.rrim)))

        # NOTE : retain this until pathinterp and pathtrapz tested
        # dpdrup = dpdr_intp(up)
        # dpdphiup = dpdphi_intp(up)
        # dpup = calculus.linetrapz(up, (dpdrup, dpdphiup))
        dpup = pathtrapz(up, gradP, gradPintp)

        for i, phi in enumerate(self.phi[RIM]):

            side = grid.pave_angular_path(
                start=(cell.rrim, cell.phi(cell.rrim)),
                stop=(cell.rrim, phi))

            # dpdrside = dpdr_intp(side)
            # dpdphiside = dpdphi_intp(side)
            # dpside = calculus.linetrapz(side, (dpdrside, dpdphiside))
            dpside = pathtrapz(side, gradP, gradPintp)

            self.dprim[i] = dpup + dpside

        for i, phi in enumerate(self.phi[RIM]):

            # This should do the trick, it does, however, look like
            # debugging/maintenance hell. I guess we should test it with fire,
            # not sure if we need additional functions/methods if this works fine
            dphi = phi-cell.phi(cell.rrim)
            self.rif[i] = fsolve(
                lambda rdown:
                    self.dprim[i]
                    + pathtrapz(
                        grid.pave_radial_path(
                            cell,
                            start=(cell.rrim, phi),
                            stop=(rdown, cell.phi(rdown)+dphi)),
                        gradP,
                        gradPintp),
                0.5*(cell.rrim+cell.rhub)
            )
            self.phiif[i] = cell.phi(self.rif[i])+dphi

    def evaluate_liquid_volume(self):
        """Evaluate volume of liquid residing within a field."""

        # For this we must process surface points on domain boundaries correctly and
        # evaluate area of the domain occupied by liquid with Gauss's area formula

        pass

    def solve(
            self,
            volume_of_liquid: float,
            prior: RotatingField,
            time_step: float,
            tol: float
    ):
        """Solve flow field for the next spatio-temporal state of the domain."""

        self.VL = volume_of_liquid
        self.t = prior.t + time_step

        self.U(prior)
        self.dUdr()
        self.dUdt(prior)
        self.gradP()

        # we can use prior field surface position for initial guesse
        r_ref = 0.5*(self.r[HUB, ANY] + self.r[RIM, ANY])
        while abs(self.VL - self.VLinterface) > tol:
            # interface resolution loop here, something like this:
            self.capture_inteface(r_ref)
            self.evaluate_liquid_volume()

            # actual logic for r_ref adjjustment must be there
            if self.VLinterface > self.VL:
                # Interface-based liquid volume evaluation overshoot target
                # liquid volume, so we must move our interface up
                r_ref += 0.1*r_ref
            else:
                # Interface-based liquid volume evaluation did not reach target
                # liquid volume, so we must move our interface down
                r_ref -= 0.1*r_ref

        # NOTE : preformance considerations
        # We can pose this as function minimisation prbolem actually,
        # probably scipy-rootfinding will be more performant than this direct loop
        #
        # maybe JIT with numba?
        #
        # Also we must ensure robust surface tracking for this to work smoothly


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
        field: list[np.ndarray],
        intp: list[ScipyInterpolator]):
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
    for f, i in zip(field, intp):
        f_interp.append(i(path, f))

    return tuple(f_interp)


def pathtrapz(path, field, intp):
    """Convinience functino for line integral along path with interpolated field values.
    Uses pathinterp, so pathinterp restrictions and features must be considered.

    Mistly used for pressure "gradient" field integration."""

    f_interp = pathinterp(path, field, intp)
    integral = calculus.linetrapz(path, f_interp)

    return integral
