from __future__ import annotations

from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import g as gravity
from lirics.euclid import Vane, Cell, STEP
from lirics.leibniz import tabular_derivative, time_derivative, tabular_line_integral


@dataclass
class VelocityField:

    radius: NDArray[float64]
    angle: NDArray[float64]

    radial_component: NDArray[float64] = field(init=False)
    tangent_component: NDArray[float64] = field(init=False)

    d_radial_dr: NDArray[float64] = field(init=False)
    d_radial_dt: NDArray[float64] = field(init=False)

    d_tangent_dr: NDArray[float64] = field(init=False)
    d_tangent_dt: NDArray[float64] = field(init=False)

    def compute_components(
            self, domain: tuple[Vane, Cell], flow: float, step=STEP) -> None:

        vane, cell = domain

        self.radial_component = -flow/cell.flow_area(self.radius)

        self.tangent_component = (
            self.radius * self.radial_component * vane.derivative(self.radius, step))

    def compute_space_derivatives(self) -> None:

        self.d_radial_dr = tabular_derivative(
            self.radial_component, self.radius)

        self.d_tangent_dr = tabular_derivative(
            self.tangent_component, self.radius)

    def compute_time_derivatives(
            self, prior_velocity_field: VelocityField, time_step: float) -> None:

        self.d_radial_dt = time_derivative(
            self.radial_component, prior_velocity_field.radial_component, time_step)

        self.d_tangent_dt = time_derivative(
            self.tangent_component, prior_velocity_field.tangent_component, time_step)


def compute_pressure_derivatives(
        time: float, angular_velocity: float, density: float,
        velocity_field: VelocityField) -> tuple[NDArray[float64], NDArray[float64]]:

    radius, angle = velocity_field.radius, velocity_field.angle
    vr, vt = velocity_field.radial_component, velocity_field.tangent_component
    dvr_dt, dvt_dt = velocity_field.d_radial_dt, velocity_field.d_tangent_dt
    dvr_dr, dvt_dr = velocity_field.d_radial_dr, velocity_field.d_tangent_dr

    d_pressure_dr = density*(
        gravity*np.cos(angular_velocity*time + angle) -
        (dvr_dt + vr*dvr_dr - 2*vt*angular_velocity -
         vt**2/radius - angular_velocity**2*radius))

    d_pressure_dphi = density*radius*(
        -gravity*np.sin(angular_velocity*time + angle) -
        (dvt_dt + vr*dvt_dr + 2*vr*angular_velocity + vr*vt/radius))

    return d_pressure_dr, d_pressure_dphi
