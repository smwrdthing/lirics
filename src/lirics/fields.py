from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray

import numpy as np
from numpy.polynomial.polynomial import polyroots, polyval
from scipy.constants import g
from scipy.interpolate import LinearNDInterpolator as linearNDintp
from scipy.optimize import newton

from lirics import calculus
from lirics import grid
from lirics.design import ImpellerCell, Housing

import matplotlib.pyplot as plt
from lirics import transform


type ScipyInterpolator = Callable[[tuple[NDArray, NDArray]], NDArray]


BACK_PHI_CORRECTION = np.deg2rad(0.5)
NUM_INTEGRATION = 100

_SENTINTEL = -1.0

# Container access keys
HUB = 0
RIM = -1
BACK = 0
FRONT = -1
MID = 1
ANY = -1


class CellField:
    """Class for representation of the flow field in the impeller cell of the
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
        self.phir = cell.phi(cell.rrim)
        self.delta = cell.delta

        # Vapor parameters?
        self.pV = np.nan
        self.VV = np.nan
        self.TV = np.nan
        self.rhoV = np.nan
        self.nV = np.nan
        self.mV = np.nan
        self.GV = np.nan
        self.RV = np.nan

        self.VL = VL
        self.actVL = np.nan

        # Considered flow is incompressible, so density "field" is constant
        self.rhoL = rho

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
        self.prim = np.zeros_like(self.phi[RIM])

        # Attributes to hold interface points
        self.rif = np.zeros_like(self.phi[RIM])
        self.phiif = np.zeros_like(self.phi[RIM])

        # phi correction (necessary for surface capturing)
        self.phicorr = np.zeros_like(self.phi[RIM])
        self.phicorr[0] = BACK_PHI_CORRECTION

    def U(self, starred: CellField):
        """Calculates velocity field components with volumetric flow rate computed
        from backward derivative approximation for liquid volume and cell midline
        tangency assumption."""

        dVL = self.VL - starred.VL
        dt = self.t - starred.t

        self.u = - 1 / self.Af * dVL / dt
        self.w = self.u * self.r * self.dphidr

    def dUdt(self, starred: CellField):
        """Calclates temporal derivative of the velocity field with backward approximation
        of the dreivative and prior spatio-temporal field."""

        dt = self.t - starred.t
        self.dudt = (self.u - starred.u)/dt
        self.dwdt = (self.w - starred.w)/dt

    def dUdr(self):
        """Calculates spatial derivatives of the velocity field numerically
        on the grid (r,phi)."""

        self.dudr = calculus.dydx(self.u, self.r)
        self.dwdr = calculus.dydx(self.w, self.r)

    def gradP(self):
        """Calculates pressure "gradient" from the governing equation for incompressibel,
        inviscid fluid flow in the uniformly rotating frame of reference."""

        omega_t = self.omega * self.t

        self.dpdr = self.rhoL * (
            g * np.cos(omega_t + self.phi) -
            (
                self.dudt
                + self.u * self.dudr
                - self.w**2 / self.r
                - 2 * self.w * self.omega
                - self.omega**2 * self.r
            )
        )

        self.dpdphi = - self.rhoL * self.r * (
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

            # Searching for dp = 0 when moving from rim to the centerline.
            # This will fail to converge without good initial guesse.
            # Using rref as initial guesse proved to be acceptable
            dphi = phi-cell.phi(cell.rrim)
            try:
                self.rif[i] = newton(
                    lambda rdown:
                        self.dprim[i]
                        + pathtrapz(
                            grid.pave_radial_path(
                                cell,
                                start=(cell.rrim, phi),
                                stop=(rdown, cell.phi(rdown)+dphi)),
                            gradPintp),
                    rref)
            except RuntimeError:
                print(
                    "WARNING : Failed to converge dp=0 problem, opting to fallback values"
                    + "(use for debugging only!)")
                if i == 0:
                    print("Fallback value : rref")
                    rfallback = rref
                else:
                    print("Fallback value: last written rif")
                    rfallback = self.rif[i-1]
                self.rif[i] = rfallback

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

        self.actVL = abs(
            cell.l * cell.avmu * calculus.areaGreenGauss(transform.rphi_to_xy(*loop)))

        return loop  # return looped path for debugging and test purposes

    def errvof(self, rref):
        """Aids in flow-field resolution in the cell. Captures interface location for
        given rref and then evaluates volume of fluid in the cell based on the location
        of the interface.

        Used in solution algorithm as a function for rootfinding with Newton method."""

        self.capture_inteface(rref)
        self.evalvof()

        return self.VL-self.actVL

    def solve(
            self,
            starred: CellField,
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

        self.U(starred)
        self.dUdr()
        self.dUdt(starred)
        self.gradP()

        # Using newton optimizer from scipy cuts it, calls ro errvof lead to
        # calls to capture_interface (re-calc. interface location) and
        # calls to evalvof() (re-calc. actaual interface-based vof).
        # So we should get everything last during last call when convergence
        # is achieved.
        # Initial guesse for interface loaction on the cell midline is based on
        # cylindrical interface shape assumption.
        cell = self._cell
        rguesse = np.sqrt(cell.rrim**2 - 2*self.VL /
                          (cell.delta * cell.l * cell.avmu))
        newton(self.errvof, rguesse)

        # At this point interface and liquid vof are resolved, so we can
        # compute vapor properties and then pressure on the rim
        self.VV = self.V - self.VL
        self.mV = starred.mV + starred.GV*(self.t-starred.t)
        self.rhoV = self.mV/self.VV
        self.pV = starred.pV * (starred.VV/self.VV)**self.nV
        self.TV = self.pV / (self.rhoV*self.RV)
        self.prim = self.pV + self.dprim


class FreeField(ABC):

    def __init__(self, cell: ImpellerCell, housing: Housing) -> None:

        self._cell = cell
        self._housing = housing

        # Sectors geometrical parameters
        self.epsilon = 0  # housing wall roughness
        self.L = housing.L
        self.R = housing.R
        self.r = cell.rrim

        # NOTE : phir must be added every time we pass alpha / automation?
        self.phir = cell.phi(cell.rrim)
        self.delta = cell.delta

        # NOTE : on field initialization
        #        Probably it would be more convenient to set values of param containers to
        #        some sentinel during object creation and adjust them afterwards with
        #        special functionality. This goes both for free and cell fields, should
        #        make "constructors" call signature lighter

        self.alpha = _SENTINTEL

        self.rho = _SENTINTEL
        self.mu = _SENTINTEL

        self.Pr = _SENTINTEL

        self.avPsi = _SENTINTEL
        self.avP = _SENTINTEL
        self.avW = _SENTINTEL

        # Back field section-averaged necessary parameters
        self.avWB = _SENTINTEL
        self.QB = _SENTINTEL
        self.GB = _SENTINTEL

        # Front field section-averaged necessary parameters
        self.avWF = _SENTINTEL
        self.QF = _SENTINTEL
        self.GF = _SENTINTEL

        self.roots = np.full((1, 3), _SENTINTEL)  # Cubic equation -> 3 roots

    def S(self, alpha):
        """Computes out-of-impeller region thickness for given rotational angle."""
        return self.R(alpha) - self.r

    def hydD(self, alpha):
        """Computes hydraulic diameter of the radial section in the free region."""

        L = self.L
        S = self.S(alpha)
        Dh = 4*L*S / (L+2*S)

        return Dh

    def V(self, alpha, n=NUM_INTEGRATION):

        delta = self.delta
        alpha = np.linspace(alpha-delta/2, alpha+delta/2, n)

        f = 1/2 * (self.R(alpha)**2 - self.r**2) * self.L

        integral = np.trapezoid(f, alpha, axis=0)

        return integral

    def avR(self, alpha):
        """Computes average radial coordinate in the out-of-impeller region
        for given rotational angle."""
        return 0.5*(self.R(alpha) + self.r)

    @abstractmethod
    def lamW(self, R, alpha):
        """Represents velocity distribution in radial section of the out-of-impeller
        region. By definition:

                                    lamW = avW / W

        This distribution must be defined to close system of equations. Thus different
        models are distinguished by different shapes of velocity profile in the out-of
        impeller region of liquid ring machine.

        ! Important : some velocity profiles will produce blatantly wrong results;
                      for example linear velocity profile will yield constant tangent
                      velocity of the flow for all rotational angles when considering
                      straight-midline cell, which clearly does not make much sense

        Other than that, the only mathematical restriction for this function, followng
        definition, is that it's average value must always yield 1."""

        # NOTE : on lamW design
        #        This should be designed to handle various models of lamW. (R,alpha)
        #        dependency is very basic and should be included everywhere, specific
        #        model can (and will), however, accept more parameters to for lamW.
        #        Base class should anticipate and expect this behavior.
        #
        # Add **kwargs/**modelparams/**params? Rely on overrides?

        raise

    def lamPsi(self, R, alpha):
        """Represents potential field related specific flow energy distribution in radial
        section of the out-of-impeller region.

        Curent implementation implies specific orientation of the machine in space,
        more gemeral approach would be introducing i.e. housing tilt"""

        gR0 = g*self.R(0)

        Psi = gR0 - g*R*np.cos(alpha)
        avPsi = gR0 - g*self.avR(alpha)*np.cos(alpha)

        return Psi/avPsi

    def lamCF(self, R, alpha, n=NUM_INTEGRATION):
        """Represents function that relates centrifugal force-field related pressure
        contribution and average velocity in the radial section. Computed numerically
        with defined lamW. Could be overriden for optimisation purposes or specific
        analytical definition of lamW."""

        x = np.linspace(self.r, R, n)
        integral = np.trapezoid(self.lamW(x, alpha)**2/x, x, axis=0)

        return integral

    def lamf(self, R, alpha):
        """Represents friction loss distribution along radial section in the free flow
        region. Base class stipulates uniform loss distribution."""

        return 1

    def kWPsi(self, alpha, n=NUM_INTEGRATION):
        """Compmutes average for convolution-like integral for velocity profile shape and
        specific potential-field related energy of the flow. Determines coefficent in the
        cubic equation in average velocity."""

        x = np.linspace(self.r, self.R(alpha), n)
        integral = np.trapezoid(
            self.lamW(x, alpha) * self.lamPsi(x, alpha), x, axis=0)

        return integral/self.S(alpha)

    def kWCF(self, alpha, n=NUM_INTEGRATION):
        """Compmutes average for convolution-like integral for velocity profile shape and
        centrifugal-field related pressure contribution. Determines coefficent in the
        cubic equation in average velocity."""

        x = np.linspace(self.r, self.R(alpha), n)
        integral = np.trapezoid(
            self.lamW(x, alpha)*self.lamCF(x, alpha), x, axis=0)

        return integral/self.S(alpha)

    def kWf(self, alpha, n=NUM_INTEGRATION):
        """Compmutes average for convolution-like integral for velocity profile shape and
        friction-related energy loss. Determines coefficent in the cubic equation in
        average velocity."""

        x = np.linspace(self.r, self.R(alpha), n)
        integral = np.trapezoid(
            self.lamW(x, alpha) * self.lamf(x, alpha), x, axis=0)

        return integral/self.S(alpha)

    def kWWW(self, alpha, n=NUM_INTEGRATION):
        """Compmutes average for integral of the cubed velocity profile shape function.
        Determines coefficent in the cubic equation in average velocity."""

        x = np.linspace(self.r, self.R(alpha), n)
        integral = np.trapezoid(self.lamW(x, alpha)**3, x, axis=0)

        return integral/self.S(alpha)

    def xi(self, alpha, dalpha):
        """Represents energy loss coefficient in the Darcy-Weisbach equation.
        Current implementation is after Raizman et al. Blends friction and local
        duct change contributions.

        Friction contribution is determined as ususall for the circular pipe with
        hydraulic diameter of rectangular duct formed by radial sections of the
        free flow region. Length is computed as arch-length for average section radius
        and specified angular shift.

        Local contribution is determined as proposed by Abramovich.
        (see Raizman's mmonograph and Idelchik's hydraulic loss handbook for details)."""

        r = self.r
        L = self.L
        epsilon = self.epsilon

        Dh = self.hydD(alpha)
        avR = self.avR(alpha)
        avS = (self.S(alpha)+self.S(alpha+dalpha))/2

        Re = self.Re(alpha)

        # Friction contribution
        # NOTE : more complicated friction factor correlation could be used,
        #        consider adjusting this
        f = 0.1 * (1.46*self.epsilon/Dh + 100/Re)**0.25
        xif = f * avR/Dh * dalpha

        # Local contribution
        A = 0.445
        B = 0.21 / (r/Dh)**(2.5*(r/Dh < 1.0) + 0.5*(r/Dh >= 1.0))
        C = (L/avS/2)**0.95 * (L/avS/2 < 3) + \
            0.223*(L/avS/2)**0.448 * (L/avS/2 >= 3)

        kRe = 20.3/Re**0.25 * (Re < 2e5) + 1 * (Re >= 2e5)
        kEps = 1 * (Re <= 4e4) + (1+epsilon/Dh*2e6) * (Re > 4e4)

        xil = kEps * kRe * A * B * C

        xi = xif + xil

        return xi

    def coeffs(self, alpha):
        """Partially fills coefficients for cubic equation of the free field model.
        Omits rhs completely, namely :
             > starred field contribution
             > coupled field contribution
             > friction contribution.

        Said contributions are augmented into coefficients array during solution
        procedure."""

        c = np.array([
            0,  # W^0 / should be based on prior field values
            (self.kWPsi(alpha)*self.avPsi + self.Pr/self.rho)*self.S(alpha),  # W^1
            0,  # W^2
            (self.kWCF(alpha) + self.kWWW(alpha)/2)*self.S(alpha)  # W^3
        ])

        return c

    @abstractmethod
    def updateWmodel(self, starred: FreeField, coupled: CellField):
        """Updates necessary parameters for velocity shape model (if any presented).
        Different models might propose different set of parameters, depending on
        different factors, to keep solution algorithm generic this separate function
        handles model parameters renewal.

        This function is called prior to anything in solve(). Thus to specify W-shape
        model user must:
            > introduce new attributes to record model parameters in inheriting class
            > override updateWmodel so that it handles parameters renewal wrt starred
              and coupled fields
            > use introduced model parameters to override lamW()

        Primary hunch is that W-shape model would mainly depend on parameters of starred
        (prior) free field and current coupled cell field, thus generic signature call."""

        return

    def solve(self, starred: FreeField, coupled: CellField):
        """Solves flow in the free-flow region of the liquid ring machine.

        Section-averaged parameters are determined as a solution of the main cubic
        equation of the model describing flow energy evolution in the free region.

        When averaged parameters are resolved - distributed parameters could be
        restored with aid of the defined distribution functions.

        To finalize computation procedure resolved section-averaged parameters are
        propagated to back ann front boundaries of the dynamic domain. Propagation
        algorithm assumes that Bernoulli's principle is satisfied locally within the cell.
        Then propagated parameters are used to determine residual of the mass balance
        equation. Minimization of that residual constitutes field-coupled computation
        procedure for cell and free field in the liquid ring machines.
        """

        self.updateWmodel(starred, coupled)

        self.alpha = coupled.alpha

        alphar = self.alpha + self.phir
        astalphar = starred.alpha + starred.phir
        dalpha = alphar - astalphar

        gR0 = g*self.R(0)  # for Psi computations

        c = self.coeffs(alphar)

        # Starred fixed field contribution
        c[0] -= polyval(starred.avW, starred.coeffs(astalphar))

        # Coupled rotating field contribution
        Ur = coupled.u[RIM, ANY]
        Wr = coupled.w[RIM, ANY] + coupled.omega*coupled.r[RIM, ANY]
        Pr = np.interp(self.phir, coupled.phi[RIM], coupled.prim)
        Psir = gR0 - g*self.r*np.cos(alphar)
        Jr = Psir + Pr/self.rho + (Ur**2 + Wr**2)/2
        c[0] -= Ur*Jr*self.r*dalpha
        # TODO : Extend CellField class to handle proper vapor parameters computations

        # Friction contribution
        astkWf = starred.kWf(astalphar)
        astxi = starred.xi(self.alpha, dalpha)
        astS = starred.S(astalphar)
        c[0] -= astkWf*astxi/2*astS * starred.avW**3

        self.roots = polyroots(c)  # roots of main equation

        # Getting 3 roots from cubic equtaion leads to root selection, currently it is
        # not known in what form roots are, we can get different values:
        #   > imaginary
        #   > real negative
        #   > real positive
        #
        # Of course only real positive roots are meaningfull, but it is not impossible to
        # get three real positive roots too
        #
        # Simplest selection just strips imaginary and negative numbers away, if all
        # roots are positive - we may face more complex root selection algorithm problem

        real_roots = np.real(self.roots[np.isreal(self.roots)])
        positive_real_root = real_roots[real_roots > 0]

        # Now it is possible to set section-average values
        self.avW = positive_real_root
        self.avP = Pr + self.rho*self.avW**2 * self.kWCF(alphar)
        self.avPsi = gR0 - g*self.avR(alphar)*np.cos(alphar)

        # Further course of action is to propagate parameters to boundaries of the
        # considred domain. Then results must be passed to mass imbalance check
        self.propagate(coupled)

    def propagate(self, coupled: CellField):

        # NOTE : some quick dirty code here, should rewrite

        alphar = self.alpha+self.phir
        alphab = (alphar-self.delta/2,
                  alphar+self.delta/2)
        gR0 = g*self.R(0)

        kWCF = self.kWCF(alphar)

        Wb = []
        for alpha, key in zip(alphab, (BACK, FRONT)):

            kWCFb = self.kWCF(alphab)
            avPsib = gR0 - g*self.avR(alpha)*np.cos(alpha)
            Prb = coupled.prim[RIM, key]

            dPsi = self.avPsi - avPsib
            dPr = self.Pr - Prb

            Wb.append(
                np.sqrt(
                    1/(kWCFb+1/2) * (dPsi + dPr/self.rho +
                                     (kWCF + 1/2)*self.avW**2)
                )
            )

        # Unpacking values to attributes
        self.avWB, self.avWF = Wb

    def Psi(self, R, alpha):
        """Computes potential-field related energy contribution.
        Uses average value and distribution function under the hood."""
        return self.avPsi * self.lamPsi(R, alpha)

    def W(self, R, alpha):
        """Computes flow velocity.
        Uses average value and distribution function under the hood."""
        return self.avW * self.lamW(R, alpha)

    def P(self, R, alpha):
        """Computes flow pressure.
        Uses average value and distribution function under the hood."""
        return self.Pr + self.rho * self.avW**2 * self.lamCF(R, alpha)

    def Re(self, alpha):
        return self.avW * self.hydD(alpha) * self.rho / self.mu


# Some fresh ideas further

class UniformFreeField(FreeField):

    def updateWmodel(self, starred: FreeField, coupled: CellField):
        return

    def lamW(self, R, alpha):
        return np.ones_like(R)


class LinearFreeFiled(FreeField):

    def __init__(self, cell: ImpellerCell, housing: Housing) -> None:
        super().__init__(cell, housing)
        self.k = _SENTINTEL
        self.b = _SENTINTEL

    def updateWmodel(self, starred: FreeField, coupled: CellField):

        alphar = self.alpha + self.phir

        S = self.S(alphar)
        R = self.R(alphar)

        self.k = -2/S
        self.b = 2/S*R

    def lamW(self, R, alpha):
        return self.k*R+self.b


class QuadFreeField(FreeField):

    def __init__(self, cell: ImpellerCell, housing: Housing) -> None:
        super().__init__(cell, housing)

        # lamW profile coefficents
        self.a = _SENTINTEL
        self.b = _SENTINTEL
        self.c = _SENTINTEL

        # Prescribed values for derivative constraint
        self.Rp = _SENTINTEL
        self.dWdRp = _SENTINTEL

    def updateWmodel(self, starred: FreeField, coupled: CellField):

        alphar = self.alpha + self.phir

        Wr = coupled.w[RIM, ANY] + coupled.omega*self.r
        r = self.r
        R = self.R(alphar)
        S = self.S(alphar)

        Rp = self.Rp
        dWdRp = self.dWdRp

        self.a = a = (Wr + self.dWdRp*S)/((R**2-r**2) - 2*Rp*S)
        self.b = b = -2*a*Rp - dWdRp
        self.c = -a*R**2 - b*R

    def lamW(self, R, alpha):

        a, b, c = self.a, self.b, self.c
        S = self.S(alpha)

        lamW = (a*R**2 + b*R + c) / (a/3*S**2 + b/2*S + c)

        return lamW


def imbalance(astcf: CellField, cf: CellField, ff: FreeField):
    # Function to compute mass imbalance of the overall solution step

    alphar = cf.alpha + cf.phir
    astalphar = astcf.alpha + astcf.phir

    dt = cf.t - astcf.t

    dVLcdt = (cf.VL-astcf.VL)/dt
    dVLfdt = (ff.V(alphar)-ff.V(astalphar))/dt
    dVLdt = dVLcdt + dVLfdt

    delta = cf.delta

    alpharB = alphar - delta/2
    alpharF = alphar + delta/2

    L = ff.L
    SB = ff.S(alpharB)
    SF = ff.S(alpharF)

    # Relative flow velocity
    avwB = ff.avWB - cf.omega*ff.avR(alpharB)
    avwF = ff.avWF - cf.omega*ff.avR(alpharB)

    QB = avwB*L*SB
    QF = avwF*L*SF
    Qsum = QB - QF

    return dVLdt - Qsum

# Auxiliary functions


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
