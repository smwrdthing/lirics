from __future__ import annotations

from typing import Iterable

from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import g as gravity
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import newton
from lirics.euclid import Vane, Cell, STEP, pave_vane_path, pave_arch_path
from lirics.leibniz import tabular_derivative, time_derivative, tabular_line_integral


@dataclass
class FlowField:

    time: float
    radius: NDArray[float64]
    angle: NDArray[float64]

    angular_velocity: float
    fluid_density: float

    translation_velocity: float = field(init=False)

    radial_velocity: NDArray[float64] = field(init=False)
    tangent_velocity: NDArray[float64] = field(init=False)
    pressure: NDArray[float64] = field(init=False)

    d_radial_dr: NDArray[float64] = field(init=False)
    d_radial_dt: NDArray[float64] = field(init=False)

    d_tangent_dr: NDArray[float64] = field(init=False)
    d_tangent_dt: NDArray[float64] = field(init=False)

    d_pressure_dr: NDArray[float64] = field(init=False)
    d_pressure_dphi: NDArray[float64] = field(init=False)

    surface_radius: list[float] = field(init=False)
    surface_angle: list[float] = field(init=False)

    mid: int = field(init=False)
    angular_shifts: NDArray[float64] = field(init=False)
    _rim_pressure_diffs: list[float] = field(init=False)
    _interpolator_d_pressure_dr: LinearNDInterpolator = field(init=False)
    _interpolator_d_pressure_dphi: LinearNDInterpolator = field(init=False)

    # This is needed because otherwise last path falls outside the domain
    _last_path_correction: float = field(default=np.deg2rad(0.05))

    def __post_init__(self):
        # we expect odd number of columns to capture midline, this is a convenience
        # variable for corresponding index
        self.mid = (self.radius.shape[1]-1)//2
        self.angular_shifts = self.angle[0, self.mid] - self.angle[0]
        self.angular_shifts[-1] += self._last_path_correction

        self.translation_velocity = (
            self.angular_velocity*self.radius[-1, self.mid])

    def compute_velocity_components(
            self, domain: tuple[Vane, Cell], flow: float, step=STEP) -> None:

        vane, cell = domain

        self.radial_velocity = -flow/cell.flow_area(self.radius)

        self.tangent_velocity = (
            self.radius * self.radial_velocity * vane.derivative(self.radius, step))

    def compute_velocity_space_derivatives(self) -> None:

        self.d_radial_dr = tabular_derivative(
            self.radial_velocity, self.radius)

        self.d_tangent_dr = tabular_derivative(
            self.tangent_velocity, self.radius)

    def compute_velocity_time_derivatives(
            self, prior_flow_field: FlowField) -> None:

        time_step = self.time-prior_flow_field.time

        self.d_radial_dt = time_derivative(
            self.radial_velocity, prior_flow_field.radial_velocity, time_step)

        self.d_tangent_dt = time_derivative(
            self.tangent_velocity, prior_flow_field.tangent_velocity, time_step)

    def compute_pressure_derivatives(self):

        time = self.time
        density = self.fluid_density
        angular_velocity = self.angular_velocity
        radius, angle = self.radius, self.angle
        vr, vt = self.radial_velocity, self.tangent_velocity
        dvr_dt, dvt_dt = self.d_radial_dt, self.d_tangent_dt
        dvr_dr, dvt_dr = self.d_radial_dr, self.d_tangent_dr

        self.d_pressure_dr = density*(
            gravity*np.cos(angular_velocity*time + angle) -
            (dvr_dt + vr*dvr_dr - 2*vt*angular_velocity -
             vt**2/radius - angular_velocity**2*radius))

        self.d_pressure_dphi = density*radius*(
            -gravity*np.sin(angular_velocity*time + angle) -
            (dvt_dt + vr*dvt_dr + 2*vr*angular_velocity + vr*vt/radius))

    def _prepare_surface_tracking(self, vane: Vane, from_radius: float):

        self._rim_pressure_diffs = []
        self.surface_radius = []
        self.surface_angle = []

        self._interpolator_d_pressure_dr = LinearNDInterpolator(
            (self.radius.ravel(), self.angle.ravel()), self.d_pressure_dr.ravel())
        self._interpolator_d_pressure_dphi = LinearNDInterpolator(
            (self.radius.ravel(), self.angle.ravel()), self.d_pressure_dphi.ravel())

        path = pave_vane_path(vane, from_radius, vane.end_radius)

        d_pressure_dr_path = self._interpolator_d_pressure_dr(path)
        d_pressure_dphi_path = self._interpolator_d_pressure_dphi(path)

        pressure_diff_up = tabular_line_integral(
            path, (d_pressure_dr_path, d_pressure_dphi_path))

        for shift in self.angular_shifts:
            if shift == 0:
                pressure_diff_arch = 0
            else:
                path = pave_arch_path(
                    vane.end_radius, vane.angular_width, vane.angular_width+shift)

                d_pressure_dr_path = self._interpolator_d_pressure_dr(path)
                d_pressure_dphi_path = self._interpolator_d_pressure_dphi(path)

                pressure_diff_arch = tabular_line_integral(
                    path, (d_pressure_dr_path, d_pressure_dphi_path))

            self._rim_pressure_diffs.append(
                pressure_diff_up + pressure_diff_arch)

    def _dp_contribution(self, vane: Vane, shift: float, radius: float):

        path = pave_vane_path(vane, vane.end_radius, radius, shift)
        d_pressure_dr_path = self._interpolator_d_pressure_dr(path)
        d_pressure_dphi_path = self._interpolator_d_pressure_dphi(path)

        pressure_diff = tabular_line_integral(
            path, (d_pressure_dr_path, d_pressure_dphi_path))

        return pressure_diff

    def track_surface(self, vane: Vane, from_radius: float):

        self._prepare_surface_tracking(vane, from_radius)

        for shift, pressure_diff in zip(self.angular_shifts, self._rim_pressure_diffs):
            if shift == 0:
                surface_radius = from_radius
            else:
                surface_radius = newton(
                    lambda radius:
                        self._dp_contribution(vane, shift, radius) +
                        pressure_diff, from_radius)

            surface_angle = vane.equation(surface_radius) + shift

            self.surface_radius.append(surface_radius)
            self.surface_angle.append(surface_angle)
        self.surface_angle[-1] -= self._last_path_correction
