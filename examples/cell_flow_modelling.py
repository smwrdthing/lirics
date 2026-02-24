import numpy as np
import matplotlib.pyplot as plt

from lirics import euclid
from lirics import euler
from lirics import descartes


RPM = 1500
OMEGA = np.pi*RPM/30
DENSITY = 1000

ANGLE_STEP = np.deg2rad(1)
TIME_STEP = ANGLE_STEP/OMEGA

# Geometry initizlaiztion & collection
vane = euclid.ArchVane(
    start_radius=100e-3,
    end_radius=200e-3,
    transition_radius=100e-3,
    thickness=5e-3,
    end_adjacent_angle=np.deg2rad(50))

cell = euclid.Cell(
    length=100e-3,
    hub_radius=vane.start_radius,
    rim_radius=vane.end_radius,
    amount=18,
    vane=vane)

domain = vane, cell

# Grid generation
R, PHI = euclid.mesh_cell(cell, vane, mesh_shape=(31, 31))
X, Y = descartes.polar_to_cartesian(R, PHI)

# Flow field initialziation
prior_flow_field = euler.FlowField(
    time=0.0,
    radius=R,
    angle=PHI,
    angular_velocity=OMEGA,
    fluid_density=DENSITY)

flow_field = euler.FlowField(
    time=prior_flow_field.time + TIME_STEP,
    radius=R,
    angle=PHI,
    angular_velocity=OMEGA,
    fluid_density=DENSITY)

# Free surface computations
Q = 1e-3
prior_flow_field.compute_velocity_components(domain, 0.0, TIME_STEP)
flow_field.compute_velocity_components(domain, Q, TIME_STEP)
flow_field.compute_velocity_space_derivatives()
flow_field.compute_velocity_time_derivatives(prior_flow_field)
flow_field.compute_pressure_derivatives()

r_init = 0.5*(vane.start_radius + vane.end_radius)
flow_field.track_surface(vane, r_init)

r_surface, phi_surface = flow_field.surface_radius, flow_field.surface_angle
r_surface, phi_surface = np.array(r_surface), np.array(phi_surface)
x_surface, y_surface = descartes.polar_to_cartesian(r_surface, phi_surface)

# Unit transition (m to mm)
X, Y = X*1e3, Y*1e3
x_surface, y_surface = x_surface*1e3, y_surface*1e3

# plotting
plt.rcParams["figure.dpi"] = 300

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title("Radial velocity in cell")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.plot(X, Y, "k", X.T, Y.T, "k", linewidth=0.5)
cf = ax.contourf(X, Y, flow_field.radial_velocity, levels=100)
fig.colorbar(cf, label="u [m/s]")

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title("Tangent velocity in cell")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.plot(X, Y, "k", X.T, Y.T, "k", linewidth=0.5)
cf = ax.contourf(X, Y, flow_field.tangent_velocity, levels=100)
fig.colorbar(cf, label="w [m/s]")

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title("Pressure partial wrt radius")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.plot(X, Y, "k", X.T, Y.T, "k", linewidth=0.5)
cf = ax.contourf(X, Y, flow_field.d_pressure_dr, levels=100)
fig.colorbar(cf, label=r"p'$_r$ [Pa/m]")

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title("Pessure partial wrt angle")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.plot(X, Y, "k", X.T, Y.T, "k", linewidth=0.5)
cf = ax.contourf(X, Y, flow_field.d_pressure_dphi, levels=100)
fig.colorbar(cf, label=r"p'$_r$ [Pa/rad]")

# plotting
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_title("Free surface in cell")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.plot(X, Y, "k", X.T, Y.T, "k", linewidth=0.5)
ax.plot(x_surface, y_surface)

plt.show()

plt.rcParams["figure.dpi"] = 100
