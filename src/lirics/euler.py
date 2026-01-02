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


def main():
    """Usage example"""

    import matplotlib.pyplot as plt
    import pandas as pd
    from lirics.euclid import ArchVane

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    vane = ArchVane(
        start_radius=50e-3,
        end_radius=100e-3,
        thickness=3e-3,
        transition_radius=50e-3,
        end_adjacent_angle=np.deg2rad(45))
    cell = Cell(
        length=100e-3,
        hub_radius=50e-3,
        rim_radius=100e-3,
        amount=12,
        vane=vane)
    # frame = CircularFrame(
    #     length=100e-3,
    #     excentricity=15e-3,
    #     radius=150e-3)

    step = 0.1e-3
    partitions = 300
    radius = np.linspace(cell.hub_radius+1e-3, cell.rim_radius, partitions)

    prior_velocity_field = VelocityField(radius, vane.equation(radius))
    velocity_field = VelocityField(radius, vane.equation(radius))

    vane_angle = vane.equation(velocity_field.radius)
    vane_angle = np.rad2deg(vane_angle)
    # vane_deriv = vane.derivative(velocity_field.radius, step)
    vane_slope = vane.slope(velocity_field.radius, step)
    slope_angle = np.arctan(vane_slope)
    slope_angle = np.rad2deg(slope_angle)
    adjacent_angle = vane.adjacent_angle(velocity_field.radius, step)
    adjacent_angle = np.rad2deg(adjacent_angle)
    # clutter = cell.clutter_coeff(velocity_field.radius)

    prir_flow = -0.8e-3
    flow = -1e-3
    RPM = 1500
    angular_velocity = np.pi*RPM/30
    angular_step = np.deg2rad(1)

    # period = 2*np.pi/angular_velocity

    time_step = angular_step/angular_velocity
    prior_velocity_field.compute_components((vane, cell), prir_flow)
    velocity_field.compute_components((vane, cell), flow)
    velocity_field.compute_space_derivatives()
    velocity_field.compute_time_derivatives(prior_velocity_field, time_step)

    velocity_incline = np.arctan(velocity_field.tangent_component /
                                 velocity_field.radial_component)
    velocity_incline = np.rad2deg(velocity_incline) + vane_angle
    # now we're happy

    print(
        pd.DataFrame({
            "radius": velocity_field.radius*1e3,
            "radial": velocity_field.radial_component,
            "tangent": velocity_field.tangent_component,
            "r-deriv(r)": velocity_field.d_radial_dr,
            "t-deriv(r)": velocity_field.d_tangent_dr,
            "r-deriv(t)": velocity_field.d_radial_dr,
            "t-deriv(t)": velocity_field.d_tangent_dr,
        })
    )

    r_vane, phi_vane = (
        velocity_field.radius, vane.equation(velocity_field.radius))
    # phi_vane_approx = vane.approximation(r_vane)

    phi_full = np.linspace(0, 2*np.pi, 500)

    r_hub, r_rim = cell.hub_radius, cell.rim_radius
    x_hub, y_hub = r_hub*np.cos(phi_full), r_hub*np.sin(phi_full)
    x_rim, y_rim = r_rim*np.cos(phi_full), r_rim*np.sin(phi_full)
    x_vane, y_vane = r_vane*np.cos(phi_vane), r_vane*np.sin(phi_vane)
    # x_vane_approx, y_vane_approx = (
    #     r_vane*np.cos(phi_vane_approx), r_vane*np.sin(phi_vane_approx))

    fig, ax = plt.subplots()
    ax.plot(
        x_vane*1e3, y_vane*1e3,
        # x_vane_approx*1e3, y_vane_approx*1e3, ":",
        x_hub*1e3, y_hub*1e3, 'k',
        x_rim*1e3, y_rim*1e3, 'k')
    ax.set_aspect("equal")
    ax.set_title("Geometry")

    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")

    shift = 20
    ax.set_xlim(r_hub*1e3-shift, r_rim*1e3+shift)
    ax.set_ylim(np.min(y_vane)*1e3-shift, np.max(y_vane)*1e3+shift)

    fig, ax = plt.subplots()
    ax.plot(
        velocity_field.radius*1e3, velocity_field.radial_component,
        velocity_field.radius*1e3, velocity_field.tangent_component)
    ax.legend(["radial", "tangent"])
    ax.set_title("Velocity components")
    ax.set_xlabel("r, mm")
    ax.set_ylabel("v, m/s")
    ax.plot()

    fig, ax = plt.subplots()
    ax.plot(
        velocity_field.radius*1e3, velocity_field.d_radial_dr,
        velocity_field.radius*1e3, velocity_field.d_tangent_dr)
    ax.legend(["radial", "tangent"])
    ax.set_title("Velocity spatial derivatvies")
    ax.set_xlabel("r, mm")
    ax.set_ylabel("dv/dr, 1/s")
    ax.plot()

    fig, ax = plt.subplots()
    ax.plot(
        velocity_field.radius*1e3, velocity_field.d_radial_dt,
        velocity_field.radius*1e3, velocity_field.d_tangent_dt)
    ax.legend(["radial", "tangent"])
    ax.set_title("Velocity time derivatvies")
    ax.set_xlabel("r, mm")
    ax.set_ylabel("dv/dt, m/s^2")
    ax.plot()

    plt.show()


if __name__ == "__main__":
    main()
