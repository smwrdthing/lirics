from __future__ import annotations
import numpy as np
from scipy.constants import g

from lirics import calculus
from lirics import grid
from lirics.design import ImpellerCell, Housing

# Container access keys
HUB = 0
RIM = -1
BACK = 0
FRONT = -1
MID = 1
ANY = -1


class RotatingField:

    def __init__(
            self,
            volume_of_liquid: float,
            density: float,
            angular_speed: float,
            cell: ImpellerCell,
            shape: tuple[int, int]
    ) -> None:

        self.V = cell.volume

        # Vapor parameters?
        self.pV = np.nan
        self.VV = np.nan
        self.TV = np.nan

        self.VL = volume_of_liquid
        self.VLinterface = np.nan

        # Considered flow is incompressible, so density "field" is constant
        self.rho = density

        # Field exists in time and space
        self.omega = angular_speed
        self.alpha = 0
        self.t = 0
        self.r, self.phi = grid.generate(cell, shape)

        self.A = cell.duct_area(self.r)
        self.dphidr = calculus.dydx(self.phi, self.r)

        self.u = np.zeros_like(self.r)
        self.w = np.zeros_like(self.r)

        self.dudr = np.zeros_like(self.r)
        self.dwdr = np.zeros_like(self.r)

        self.dudt = np.zeros_like(self.r)
        self.dwdt = np.zeros_like(self.r)

        self.dpdr = np.zeros_like(self.r)
        self.dpdphi = np.zeros_like(self.r)

        self.r_interface = np.nan
        self.phi_interface = np.nan

    def U(self, prior_field: RotatingField):
        '''Determine velocity field components'''

        dVL = self.VL - prior_field.VL
        dt = self.t - prior_field.t

        self.u = - 1 / self.A * dVL / dt
        self.w = self.u * self.r * self.dphidr

    def dUdt(self, prior_field: RotatingField):
        '''Determine temporal derivative of the velocity field'''

        dt = self.t - prior_field.t
        self.dudt = (self.u - prior_field.u)/dt
        self.dwdt = (self.w - prior_field.w)/dt

    def dUdr(self):
        '''Determine spatial derivatives of the velocity field'''

        self.dudr = calculus.dydx(self.u, self.r)
        self.dwdr = calculus.dydx(self.w, self.r)

    def gradP(self):
        '''Determine pressure gradien components from governing equation for fluid flow'''

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

    def capture_inteface(self, reference_radius):
        '''Capture interface points in the cell for given refrence radius'''

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

        pass

    def evaluate_liquid_volume(self):
        '''Evaluate volume of liquid residing within a field'''

        # For this we must process surface points on domain boundaries correctly and
        # evaluate area of the domain occupied by liquid with Gauss's area formula

        pass

    def solve(
            self,
            volume_of_liquid: float,
            time_step: float,
            prior_field: RotatingField,
            tol: float
    ):
        '''Solve time step for provided new value of volume of liquid in field domain'''

        self.VL = volume_of_liquid
        self.t = prior_field.t + time_step

        self.U(prior_field)
        self.dUdr()
        self.dUdt(prior_field)
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


class StationaryField:

    def __init__(self, cell: ImpellerCell, housing: Housing) -> None:

        # We hold parameters for three radial sections for calculations
        # of fluid flow in stationary domain of liquir-ring machine:
        # middle, back and front, thus [0.0]*3 things

        # Sectors geometrical parameters
        self.midline_dphi = cell.midline_angle(cell.rim_radius)
        self.alpha = [0.0]*3
        self.r = cell.hub_radius
        self.R = [0.0]*3  # sectors radial bounds
        self.S = [0.0]*3  # sectors radial span
        self.housingR = housing.profile

        # Flow parameters
        self.avPSI = [0.0]*3  # potential field contribution
        self.avW = [0.0]*3  # velocity contribution
        self.avP = [0.0]*3  # pressure contribution
        self.avJ = [0.0]*3  # overall energy flux

    def sector_geometry_bump(self, rotating_field: RotatingField):

        mid = rotating_field.omega * rotating_field.t + self.midline_dphi
        half = self.beta/2
        self.alpha = [mid - half, mid, mid + half]

        for i, alpha in enumerate(self.alpha):
            self.R[i] = self.housingR(alpha)
            self.S[i] = self.R[i] - self.r

    def propagate(self):
        pass

    def solve(
            self,
            rotating_field: RotatingField,
            prior_field: StationaryField):
        pass
